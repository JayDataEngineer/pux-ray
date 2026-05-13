"""VibeVoice TTS handler — text-to-speech with voice cloning.
 
Uses VibeVoiceForConditionalGenerationInference (vibevoice/VibeVoice-7B).
Sub-modules in mmgp pipe dict, generate() on full model.
Supports multi-speaker TTS with optional voice cloning.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "vibevoice-tts": ModelVariant(
        name="vibevoice-tts", family="vibevoice_tts",
        display_name="VibeVoice TTS 7B",
        vram_estimate_gb=18,
        defaults={"language": "English", "max_tokens": 4096},
    ),
}


class VibeVoiceTTSHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "tts" / "vibevoice"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown VibeVoice TTS type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> tuple:
        from .modules import VibeVoiceTTSModules
        from .orchestrator import VibeVoiceTTSOrchestrator

        modules = VibeVoiceTTSModules.load(model_path=model_path, dtype=dtype)
        orchestrator = VibeVoiceTTSOrchestrator(modules)

        pipe = {"pipe": modules.pipe, "coTenantsMap": modules.co_tenants}
        return orchestrator, pipe
