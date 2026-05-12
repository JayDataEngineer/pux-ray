"""HY-Motion handler — text-to-3D human motion generation.

Decomposed into motion_transformer (HunyuanMotionMMDiT), text_encoder
(Qwen3-8B + CLIP), body_model (WoodenMesh). Orchestrator runs ODE
denoising with explicit forward() calls.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "hy-motion-1.0": ModelVariant(
        name="hy-motion-1.0", family="hy_motion", display_name="HY-Motion 1.0",
        vram_estimate_gb=6,
        defaults={"duration": 3.0, "cfg_scale": 3.0, "output_format": "dict"},
    ),
    "hy-motion-1.0-lite": ModelVariant(
        name="hy-motion-1.0-lite", family="hy_motion", display_name="HY-Motion 1.0 Lite",
        vram_estimate_gb=4,
        defaults={"duration": 3.0, "cfg_scale": 3.0, "output_format": "dict"},
    ),
}


class HYMotionHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "motion" / model_type

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown HY-Motion type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from .modules import HYMotionModules
        from .orchestrator import HYMotionOrchestrator

        modules = HYMotionModules.load(model_path=model_path, model_type=model_type)
        orchestrator = HYMotionOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
