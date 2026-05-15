"""Raw Kokoro TTS model — loads nn.Modules directly, no kokoro pip package.

Architecture (StyleTTS2):
  bert          — PLBERT (AlbertModel) text encoder
  bert_encoder  — Linear projection BERT → hidden_dim
  predictor     — ProsodyPredictor (duration, F0, energy)
  text_encoder  — CNN + LSTM phoneme encoder
  decoder       — iSTFT vocoder (Generator)

All nn.Module definitions come from vendor's multitalk kokoro module.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import importlib
import importlib.util
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from transformers import AlbertConfig, AlbertModel

# Import nn.Module definitions from vendor's multitalk kokoro module directly.
# We create a fake package to support relative imports within those files,
# avoiding the heavy Wan2GP import chain (smplfitter, etc.).
_MULTITALK_DIR = Path(__file__).resolve().parents[1] / "wan" / "multitalk" / "kokoro"
_PKG_NAME = "_kokoro_raw_modules"


def _setup_kokoro_package():
    """Create a fake package and load all kokoro modules with relative import support."""
    import sys

    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]

    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [str(_MULTITALK_DIR)]
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg

    files = {
        "custom_stft": _MULTITALK_DIR / "custom_stft.py",
        "istftnet": _MULTITALK_DIR / "istftnet.py",
        "modules": _MULTITALK_DIR / "modules.py",
    }
    for name, path in files.items():
        fqn = f"{_PKG_NAME}.{name}"
        if fqn in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(fqn, str(path))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG_NAME
        sys.modules[fqn] = mod
        setattr(pkg, name, mod)

    # Execute in dependency order: custom_stft → istftnet → modules
    for name in ["custom_stft", "istftnet", "modules"]:
        fqn = f"{_PKG_NAME}.{name}"
        sys.modules[fqn].spec = importlib.util.spec_from_file_location(fqn, str(files[name]))
        sys.modules[fqn].spec.loader.exec_module(sys.modules[fqn])

    return pkg


_pkg = _setup_kokoro_package()
CustomAlbert = _pkg.modules.CustomAlbert
ProsodyPredictor = _pkg.modules.ProsodyPredictor
TextEncoder = _pkg.modules.TextEncoder
Decoder = _pkg.istftnet.Decoder

logger = logging.getLogger(__name__)


@dataclass
class KokoroOutput:
    audio: torch.FloatTensor
    pred_dur: Optional[torch.LongTensor] = None


class KokoroModel(nn.Module):
    """Kokoro TTS — 5 decomposed nn.Modules for mmgp VRAM management."""

    def __init__(self, config: dict, weights_path: str | Path):
        super().__init__()
        self.vocab = config["vocab"]
        self.context_length = config.get("context_length", 512)

        # 5 nn.Module components — each individually loadable/unloadable by mmgp
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=config["n_token"], **config["plbert"])
        )
        self.bert_encoder = nn.Linear(
            self.bert.config.hidden_size, config["hidden_dim"]
        )
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        # Load weights from .pth
        state_dicts = torch.load(weights_path, map_location="cpu", weights_only=True)
        for key, sd in state_dicts.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown component in weights: {key}")
            try:
                getattr(self, key).load_state_dict(sd)
            except RuntimeError:
                sd = {k[7:]: v for k, v in sd.items()}
                getattr(self, key).load_state_dict(sd, strict=False)

    @property
    def device(self) -> torch.device:
        return self.bert.device

    @torch.no_grad()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ) -> KokoroOutput:
        input_ids = list(
            filter(None, (self.vocab.get(p) for p in phonemes))
        )
        assert len(input_ids) + 2 <= self.context_length, (
            len(input_ids) + 2, self.context_length
        )

        device = ref_s.device
        input_ids_t = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        input_lengths = torch.LongTensor([input_ids_t.shape[-1]]).to(device)

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        bert_dur = self.bert(input_ids_t, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(input_ids_t.shape[1], device=device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids_t.shape[1], indices.shape[0]), device=device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids_t, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return KokoroOutput(audio=audio.cpu(), pred_dur=pred_dur.cpu())


def load_kokoro(
    model_dir: str | Path,
) -> tuple[KokoroModel, dict[str, nn.Module]]:
    """Load KokoroModel and return (model, pipe_dict) for mmgp.

    Args:
        model_dir: Directory containing config.json + kokoro-v1_0.pth

    Returns:
        (KokoroModel, pipe_dict) where pipe_dict maps component names to nn.Modules
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "kokoro-v1_0.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"kokoro-v1_0.pth not found at {weights_path}")

    with open(config_path) as f:
        config = json.load(f)

    model = KokoroModel(config, weights_path)

    # Decomposed pipe_dict for mmgp VRAM management
    pipe_dict = {
        "bert": model.bert,
        "bert_encoder": model.bert_encoder,
        "predictor": model.predictor,
        "text_encoder": model.text_encoder,
        "decoder": model.decoder,
    }

    return model, pipe_dict
