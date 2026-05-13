"""See-Through handler — anime layer decomposition.

Decomposed into raw nn.Modules:
- LayerDiff: UNetFrameConditionModel, AutoencoderKL (SDXL VAE), TransparentVAE,
             CLIPTextModel (2x), CLIPTokenizer (2x), DPMSolverMultistepScheduler
- Marigold: UNetFrameConditionModel, AutoencoderKL (SDXL VAE), CLIPTextModel,
            CLIPTokenizer, DDIMScheduler

Architecture: Stage 1 (LayerDiff) → Stage 2 (Marigold depth) → Stage 3 (post-process)
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "see-through": ModelVariant(
        name="see-through", family="see_through", display_name="See-Through Layer Decomposition",
        vram_estimate_gb=6,
        defaults={"resolution": 1280, "steps": 30},
    ),
}


class SeeThroughHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "creative" / "see-through"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown See-Through type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from .modules import SeeThroughModules
        from .orchestrator import SeeThroughOrchestrator

        modules = SeeThroughModules.load(model_path=model_path)
        orchestrator = SeeThroughOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
