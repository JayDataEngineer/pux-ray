"""MOSS-SoundEffect handler — text-to-sound effect generation.

Decomposed into language_model (Qwen3-8B), emb_ext (16 VQ embeddings),
lm_heads (17 prediction heads), audio_tokenizer (CPU).
Note: generate() uses the full model due to delay pattern coupling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "moss-soundeffect": ModelVariant(
        name="moss-soundeffect", family="moss", display_name="MOSS-SoundEffect 8B",
        vram_estimate_gb=18,
        defaults={"max_tokens": 4096},
    ),
}


class MossHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "audio" / "moss-soundeffect"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown MOSS type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from .modules import MossModules
        from .orchestrator import MossOrchestrator

        modules = MossModules.load(model_path=model_path, dtype=dtype)
        orchestrator = MossOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
