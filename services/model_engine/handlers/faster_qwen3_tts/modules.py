"""FasterQwen3-TTS raw nn.Module loading + mmgp setup.

Decomposes Qwen3TTSModel into:
- talker: Qwen3TTSTalker (28-layer transformer + text_projection + codec_head)
- code_predictor: predictor module (5-layer transformer + 15 lm_heads + 15 codec_embeds)
- speech_tokenizer: codec decoder (codes -> audio waveform)

The orchestrator calls forward() directly on each module.
No CUDA graphs — those are an optimization layer on top of this decomposition.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class FasterQwen3TTSModules:
    """Raw nn.Modules for FasterQwen3-TTS inference."""

    # Primary nn.Modules — managed by mmgp
    talker: Any           # talker nn.Module (has forward(), contains .model, .text_projection, .codec_head)
    code_predictor: Any   # code_predictor nn.Module (has .model, .lm_head, .small_to_mtp_projection)
    speech_tokenizer: Any # codec decoder nn.Module (has decode())

    # Extracted sub-module references for direct forward() calls in orchestrator
    talker_model: Any = None       # talker.model — 28-layer transformer backbone
    pred_model: Any = None         # code_predictor.model — 5-layer predictor transformer
    pred_projection: Any = None    # code_predictor.small_to_mtp_projection
    pred_lm_heads: Any = None      # code_predictor.lm_head — ModuleList[15]
    pred_codec_embeds: Any = None  # code_predictor.model.codec_embedding — ModuleList[15]
    talker_codec_embed: Any = None # talker.get_input_embeddings() — codec embedding
    talker_codec_head: Any = None  # talker.codec_head — output projection

    # Base model for preprocessing (NOT in pipe dict — no heavy params)
    base_model: Any = None

    # Config
    talker_config: Any = None
    pred_config: Any = None
    sample_rate: int = 24000

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
    ) -> FasterQwen3TTSModules:
        from faster_qwen3_tts.utils import suppress_flash_attn_warning

        with suppress_flash_attn_warning():
            from qwen_tts import Qwen3TTSModel

        logger.info("Loading Qwen3-TTS from %s", model_path)

        base_model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map="cpu",
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )

        # Extract nn.Modules from the model hierarchy
        inner = base_model.model
        talker = inner.talker
        code_predictor = talker.code_predictor
        speech_tokenizer = inner.speech_tokenizer

        # Sub-module references for direct forward() calls
        talker_model = talker.model
        pred_model = code_predictor.model
        pred_projection = code_predictor.small_to_mtp_projection
        pred_lm_heads = code_predictor.lm_head
        pred_codec_embeds = code_predictor.model.codec_embedding
        talker_codec_embed = talker.get_input_embeddings()
        talker_codec_head = talker.codec_head

        talker_config = inner.config.talker_config
        pred_config = code_predictor.model.config

        # Infer sample rate
        sample_rate = 24000
        if speech_tokenizer is not None:
            sr = getattr(speech_tokenizer, "sample_rate", None)
            if sr is not None:
                sample_rate = int(sr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move to device
        talker = talker.to(device)
        code_predictor = code_predictor.to(device)
        speech_tokenizer = speech_tokenizer.to(device)
        talker.eval()
        code_predictor.eval()
        speech_tokenizer.eval()

        # mmgp pipe dict — three independent nn.Modules
        pipe = {
            "talker": talker,
            "code_predictor": code_predictor,
            "speech_tokenizer": speech_tokenizer,
        }
        co_tenants = {
            "talker": ["code_predictor"],
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0
        logger.info(
            "Qwen3-TTS decomposed: talker + code_predictor + speech_tokenizer VRAM=%.0fMB",
            vram,
        )

        return cls(
            talker=talker,
            code_predictor=code_predictor,
            speech_tokenizer=speech_tokenizer,
            talker_model=talker_model,
            pred_model=pred_model,
            pred_projection=pred_projection,
            pred_lm_heads=pred_lm_heads,
            pred_codec_embeds=pred_codec_embeds,
            talker_codec_embed=talker_codec_embed,
            talker_codec_head=talker_codec_head,
            base_model=base_model,
            talker_config=talker_config,
            pred_config=pred_config,
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
            pipe=pipe,
            co_tenants=co_tenants,
        )
