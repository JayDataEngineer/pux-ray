"""AniGen handler — image-to-rigged-3D generation.
 
Decomposed into 6 nn.Modules: dinov2 (image encoder), dsine (normal estimation),
ss_flow_model (sparse structure diffusion), ss_decoder, slat_flow_model
(structured latent diffusion), slat_decoder (mesh + skin weights).
All FP32. Flash attention + pytorch3d patches applied.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "anigen": ModelVariant(
        name="anigen", family="anigen", display_name="AniGen Image-to-3D",
        vram_estimate_gb=10,
        defaults={"ss_steps": 25, "slat_steps": 25, "cfg_scale_ss": 7.5,
                  "cfg_scale_slat": 3.0, "texture_size": 1024},
    ),
}


class AniGenHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "3d" / "anigen"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown AniGen type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple:
        from .modules import AniGenModules
        from .orchestrator import AniGenOrchestrator

        modules = AniGenModules.load(model_path=model_path)
        orchestrator = AniGenOrchestrator(modules)

        pipe = {"pipe": modules.pipe, "coTenantsMap": modules.co_tenants}
        return orchestrator, pipe
