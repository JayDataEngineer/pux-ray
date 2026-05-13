"""Kokoro TTS handler — 82M param CPU text-to-speech.

Decomposes KModel into bert, bert_encoder, predictor, text_encoder, decoder.
Runs on CPU — no GPU needed. Part of the unified model_engine system.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "kokoro": ModelVariant(
        name="kokoro", family="kokoro",
        display_name="Kokoro 82M TTS",
        vram_estimate_gb=0,
        defaults={"voice": "af_bella", "speed": 1.0},
    ),
}


class KokoroHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "tts" / "kokoro"

    def get_variant(self, model_type: str) -> ModelVariant:
        return VARIANTS["kokoro"]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> LoadResult:
        from .modules import KokoroModules
        from .orchestrator import KokoroOrchestrator

        modules = KokoroModules.load(model_path=model_path, dtype=dtype)
        orchestrator = KokoroOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
