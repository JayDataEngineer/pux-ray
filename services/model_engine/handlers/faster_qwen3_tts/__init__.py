"""FasterQwen3-TTS handler — CUDA graph accelerated TTS.

Wraps the faster_qwen3_tts pip package. Uses CUDA graphs for 5x speedup.
Supports CustomVoice (9 speakers), voice cloning, and voice design.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "qwen3-tts-customvoice": ModelVariant(
        name="qwen3-tts-customvoice", family="faster_qwen3_tts",
        display_name="Faster Qwen3-TTS 1.7B (CustomVoice)",
        vram_estimate_gb=6,
        defaults={"voice": "Aiden", "language": "English"},
    ),
    "qwen3-tts-voiceclone": ModelVariant(
        name="qwen3-tts-voiceclone", family="faster_qwen3_tts",
        display_name="Faster Qwen3-TTS 1.7B (Voice Clone)",
        vram_estimate_gb=6,
        defaults={"language": "English"},
    ),
}


class FasterQwen3TTSHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "tts" / "qwen3-tts-12hz-1.7b-customvoice"

    def get_variant(self, model_type: str) -> ModelVariant:
        key = model_type if model_type in VARIANTS else "qwen3-tts-customvoice"
        return VARIANTS[key]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from .modules import FasterQwen3TTSModules
        from .orchestrator import FasterQwen3TTSOrchestrator

        modules = FasterQwen3TTSModules.load(model_path=model_path, dtype=dtype)
        orchestrator = FasterQwen3TTSOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
