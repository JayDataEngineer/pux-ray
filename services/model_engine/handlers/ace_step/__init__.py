"""ACE-Step handler — text-to-music generation.

Loads raw nn.Modules individually (no pipeline wrapper), orchestrates
inference via AceStepOrchestrator, and exposes through ForgeService.

Supports v1, v1.5 (7 variants), and v1.5 XL with all Wan2GP features:
CoT metadata inference, audio code generation, reference audio, cover
blending, ODE/SDE denoising, and VAE temporal tiling.

Reference: Wan2GP's models/TTS/ace_step_handler.py decomposition.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)


# ── Model type → variant metadata ─────────────────────────────────────────────
# Same as before — defines defaults per model type

VARIANTS = {
    "ace_step_v1": ModelVariant(
        name="ace_step_v1", family="ace_step", display_name="ACE-Step v1",
        vram_estimate_gb=8,
        defaults={"steps": 60, "guidance_scale": 7.0, "sample_solver": "euler", "duration_seconds": 30},
    ),
    "ace_step_v1_5": ModelVariant(
        name="ace_step_v1_5", family="ace_step", display_name="ACE-Step 1.5 (base)",
        vram_estimate_gb=7,
        defaults={"steps": 8, "guidance_scale": 1.0, "alt_guidance_scale": 2.5,
                   "duration_seconds": 30, "temperature": 0.85, "top_p": 0.9},
    ),
    "ace_step_v1_5_turbo": ModelVariant(
        name="ace_step_v1_5_turbo", family="ace_step", display_name="ACE-Step 1.5 Turbo",
        vram_estimate_gb=7,
        defaults={"steps": 8, "guidance_scale": 1.0, "alt_guidance_scale": 2.5,
                   "duration_seconds": 30, "temperature": 0.85, "top_p": 0.9},
    ),
    "ace_step_v1_5_sft": ModelVariant(
        name="ace_step_v1_5_sft", family="ace_step", display_name="ACE-Step 1.5 SFT",
        vram_estimate_gb=7,
        defaults={"steps": 8, "guidance_scale": 1.0, "alt_guidance_scale": 2.5,
                   "duration_seconds": 30, "temperature": 0.85, "top_p": 0.9},
    ),
    "ace_step_v1_5_xl_turbo": ModelVariant(
        name="ace_step_v1_5_xl_turbo", family="ace_step", display_name="ACE-Step 1.5 XL Turbo",
        vram_estimate_gb=12,
        defaults={"steps": 8, "guidance_scale": 1.0, "alt_guidance_scale": 2.5,
                   "duration_seconds": 30, "temperature": 0.85, "top_p": 0.9},
    ),
}


class AceStepHandler(BaseHandler):
    """ACE-Step music generation handler.

    Loads raw nn.Modules via AceStepModules, wraps them with mmgp,
    and delegates inference to AceStepOrchestrator.
    """

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root / "audio" / "acestep"

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(
                f"Unknown ACE-Step type: {model_type}. Available: {list(VARIANTS.keys())}"
            )
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        quantize_transformer: bool = False,
        enable_lm: bool = True,
        **kwargs,
    ) -> LoadResult:
        """Load ACE-Step v1.5 as raw modules + orchestrator.

        The pipe dict (for mmgp) contains the raw nn.Modules.
        The pipeline object is the orchestrator (stateless, no pipeline class).
        """
        if model_type == "ace_step_v1":
            raise NotImplementedError("ACE-Step v1 loading not yet implemented")

        from .modules import AceStepModules

        modules = AceStepModules.load(
            model_path=model_path,
            model_type=model_type,
            dtype=dtype,
            enable_lm=enable_lm,
        )

        from .orchestrator import AceStepOrchestrator

        orchestrator = AceStepOrchestrator(modules)

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info(
            "ACE-Step loaded: modules=%s VRAM=%.0fMB",
            list(modules.pipe.keys()), vram,
        )

        return LoadResult(
            pipeline=orchestrator,
            pipe=modules.pipe,
            co_tenants=modules.co_tenants,
        )
