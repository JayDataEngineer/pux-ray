"""Faster-Whisper ASR handler — CTranslate2 backend, CPU inference.
 
No nn.Modules — Faster-Whisper uses CTranslate2 (C++ backend).
The handler provides the same BaseHandler interface with an empty pipe dict.
One system, one truth.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "faster_whisper": ModelVariant(
        name="faster_whisper", family="faster_whisper",
        display_name="Faster-Whisper Distil-Large-V3",
        vram_estimate_gb=0,
        defaults={"language": None, "beam_size": 5},
    ),
}


class FasterWhisperHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "asr" / "faster-whisper"

    def get_variant(self, model_type: str) -> ModelVariant:
        return VARIANTS["faster_whisper"]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        **kwargs,
    ) -> tuple:
        if not model_path.exists() or not any(model_path.iterdir()):
            raise FileNotFoundError(f"Faster-Whisper model not found at {model_path}")

        from faster_whisper import WhisperModel
        from .orchestrator import FasterWhisperOrchestrator

        model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
        orchestrator = FasterWhisperOrchestrator(model=model)

        return orchestrator, {}
