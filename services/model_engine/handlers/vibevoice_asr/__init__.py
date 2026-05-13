"""VibeVoice ASR handler — speech-to-text with diarization.

Uses VibeVoiceASRForConditionalGeneration (microsoft/VibeVoice-ASR 7B).
Sub-modules in mmgp pipe dict, generate() on full model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "vibevoice-asr": ModelVariant(
        name="vibevoice-asr", family="vibevoice_asr",
        display_name="VibeVoice ASR 7B",
        vram_estimate_gb=16,
        defaults={"language": "english", "max_tokens": 512},
    ),
}


class VibeVoiceASRHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "asr" / "vibevoice-asr"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown VibeVoice ASR type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from .modules import VibeVoiceASRModules
        from .orchestrator import VibeVoiceASROrchestrator

        modules = VibeVoiceASRModules.load(model_path=model_path, dtype=dtype)
        orchestrator = VibeVoiceASROrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
