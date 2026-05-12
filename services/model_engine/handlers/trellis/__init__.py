"""TRELLIS.2 handler — image-to-3D mesh generation.

Decomposed to raw nn.Modules: 8 flow models + decoders + DINOv3 + BiRefNet.
Orchestrator calls .forward() directly on each module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "trellis": ModelVariant(
        name="trellis", family="trellis", display_name="TRELLIS.2 Image-to-3D",
        vram_estimate_gb=10,
        defaults={"steps": 12, "guidance": 7.5, "resolution": "1024_cascade",
                  "decimation": 50000, "texture_size": 4096},
    ),
}


class TrellisHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "3d" / "trellis"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown TRELLIS type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        precision: str = "bf16",
        **kwargs,
    ) -> LoadResult:
        from .modules import TrellisModules
        from .orchestrator import TrellisOrchestrator

        modules = TrellisModules.load(
            model_path=model_path,
            precision=precision,
        )
        orchestrator = TrellisOrchestrator(modules)

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
