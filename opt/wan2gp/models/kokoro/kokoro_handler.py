"""Kokoro TTS family handler — raw nn.Module decomposition, no kokoro pip package.

GPU service via mmgp full-RAM mode. Decomposed into 5 components:
  bert, bert_encoder, predictor, text_encoder, decoder

Phonemization via espeak-ng (phonemizer library), not misaki/spacy.

Wan2GP dependency (Amendment B): nn.Module definitions (ProsodyPredictor, TextEncoder,
Decoder, CustomAlbert) imported from Wan2GP's multitalk kokoro module via kokoro_model.py.
The handler and phonemizer are authored; only the raw Module classes come from Wan2GP.
"""
from __future__ import annotations

import base64
import io
import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.kokoro.kokoro_model import KokoroModel, load_kokoro
from models.kokoro.kokoro_phonemizer import phonemize, chunk_phonemes

logger = logging.getLogger(__name__)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["kokoro"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "kokoro"

    @staticmethod
    def query_family_infos():
        return {"kokoro": (302, "Kokoro TTS")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"audio_only": True, "image_outputs": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from models._shared import resolve_model_path
        model_path = resolve_model_path(
            "kokoro", "kokoro_path", model_def,
            check_file="config.json", quant=kwargs.get("quant"),
        )

        if not (model_path / "config.json").exists():
            # Download config + weights from HuggingFace on first use
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="hexgrad/Kokoro-82M",
                local_dir=str(model_path),
                allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/*.pt"],
            )

        kmodel, pipe_dict = load_kokoro(model_path)
        pipeline = _Pipeline(kmodel, model_path)

        return pipeline, pipe_dict

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": "Hello world"})


class _Pipeline:
    def __init__(self, kmodel: KokoroModel, model_path: Path):
        self.kmodel = kmodel
        self.model_path = model_path
        self._voice_cache: dict[str, torch.FloatTensor] = {}

    def generate(self, *, input_prompt="", voice="af_bella", speed=1.0,
                 lang_code="a", seed=-1, **kw):
        text = input_prompt or kw.get("text", "")
        if not text:
            raise ValueError("text required")

        voice_pack = self._load_voice(voice)  # [510, 1, 256]
        phonemes = phonemize(text, lang_code=lang_code)
        if not phonemes:
            raise ValueError(f"Phonemization produced empty output for: {text!r}")

        chunks = chunk_phonemes(phonemes)
        audio_parts = []

        for ps in chunks:
            # Select voice row matching phoneme length → [1, 256]
            ref_s = voice_pack[len(ps) - 1]
            output = self.kmodel(ps, ref_s, speed)
            audio_parts.append(output.audio.numpy())

        audio_data = np.concatenate(audio_parts) if audio_parts else np.array([])

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes((audio_data * 32767).astype("int16").tobytes())

        return {
            "status": "success",
            "data": base64.b64encode(buf.getvalue()).decode(),
            "media_type": "audio/wav",
        }

    def _load_voice(self, voice: str) -> torch.FloatTensor:
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        # Try local file first
        local = self.model_path / "voices" / f"{voice}.pt"
        if local.exists():
            pack = torch.load(local, weights_only=True)
            self._voice_cache[voice] = pack
            return pack

        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        voice_file = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename=f"voices/{voice}.pt",
        )
        pack = torch.load(voice_file, weights_only=True)
        self._voice_cache[voice] = pack
        return pack
