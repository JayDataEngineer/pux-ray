"""See-Through raw nn.Module decomposition.

Separates the two-stage pipeline into disjoint nn.Modules:
- LayerDiff: UNetFrameConditionModel, AutoencoderKL (SDXL VAE), TransparentVAE,
             CLIPTextModel (2x), CLIPTokenizer (2x), DPMSolverMultistepScheduler
- Marigold: UNetFrameConditionModel, AutoencoderKL (SDXL VAE), CLIPTextModel,
            CLIPTokenizer, DDIMScheduler

Both stages share the same SDXL VAE architecture but load from different pretrained repos,
so we keep separate VAE instances (VRAM impact is minimal: ~300MB each at bf16).
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

HUGGINGFACE_LAYERDIFF = "layerdifforg/seethroughv0.0.2_layerdiff3d"
HUGGINGFACE_MARIGOLD = "24yearsold/seethroughv0.0.1_marigold"


@dataclass
class SeeThroughModules:
    """All raw nn.Modules for See-Through inference."""

    # LayerDiff stage
    ld_unet: Any
    ld_vae: Any
    ld_trans_vae: Any
    ld_text_encoder: Any
    ld_text_encoder_2: Any
    ld_tokenizer: Any
    ld_tokenizer_2: Any
    ld_scheduler: Any

    # Marigold stage
    mg_unet: Any
    mg_vae: Any
    mg_text_encoder: Any
    mg_tokenizer: Any
    mg_scheduler: Any

    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    # Cached prompt embeddings
    _cached_prompt_embeds: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path = None,
        ld_pretrained: str = HUGGINGFACE_LAYERDIFF,
        mg_pretrained: str = HUGGINGFACE_MARIGOLD,
    ) -> SeeThroughModules:
        from services.compat import apply
        apply()

        from registry.config import Config
        cfg = Config()

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        # Add seethrough common to path
        seethrough_common = str(Path(cfg.project_root) / "vendor" / "seethrough" / "common")
        if seethrough_common not in sys.path:
            sys.path.insert(0, seethrough_common)

        logger.info("Loading See-Through modules...")

        # ---- Stage 1: LayerDiff ----
        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
        from modules.layerdiffuse.vae import TransparentVAE
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        from diffusers import DPMSolverMultistepScheduler

        logger.info("  Loading TransparentVAE...")
        trans_vae = TransparentVAE.from_pretrained(ld_pretrained, subfolder="trans_vae")

        logger.info("  Loading LayerDiff UNet...")
        ld_unet = UNetFrameConditionModel.from_pretrained(ld_pretrained, subfolder="unet")

        # Load the full pipeline to access text encoders, VAE
        logger.info("  Loading LayerDiff pipeline components...")
        ld_pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            ld_pretrained,
            trans_vae=trans_vae,
            unet=ld_unet,
            scheduler=None,
        )

        # Create the scheduler separately (pipeline constructor creates one with defaults)
        model_id = "frankjoshua/juggernautXL_version6Rundiffusion"
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            final_sigmas_type="zero",
            euler_at_final=True,
        )

        ld_vae = ld_pipeline.vae
        ld_text_encoder = ld_pipeline.text_encoder
        ld_text_encoder_2 = ld_pipeline.text_encoder_2
        ld_tokenizer = ld_pipeline.tokenizer
        ld_tokenizer_2 = ld_pipeline.tokenizer_2

        # Move to GPU in bf16
        for m in [ld_vae, ld_unet, trans_vae, ld_text_encoder, ld_text_encoder_2]:
            m.to(device="cuda", dtype=torch.bfloat16)
            m.eval()

        # ---- Stage 2: Marigold ----
        from modules.marigold.marigold_depth_pipeline import MarigoldDepthPipeline
        from diffusers import DDIMScheduler

        logger.info("  Loading Marigold UNet...")
        mg_unet = UNetFrameConditionModel.from_pretrained(mg_pretrained, subfolder="unet")

        logger.info("  Loading Marigold pipeline components...")
        mg_pipeline = MarigoldDepthPipeline.from_pretrained(mg_pretrained, unet=mg_unet)

        mg_vae = mg_pipeline.vae
        mg_text_encoder = mg_pipeline.text_encoder
        mg_tokenizer = mg_pipeline.tokenizer
        mg_scheduler = mg_pipeline.scheduler

        for m in [mg_vae, mg_unet, mg_text_encoder]:
            m.to(device="cuda", dtype=torch.bfloat16)
            m.eval()

        torch.cuda.empty_cache()
        gc.collect()

        pipe = {
            "ld_unet": ld_unet,
            "ld_vae": ld_vae,
            "ld_trans_vae": trans_vae,
            "ld_text_encoder": ld_text_encoder,
            "ld_text_encoder_2": ld_text_encoder_2,
            "mg_unet": mg_unet,
            "mg_vae": mg_vae,
            "mg_text_encoder": mg_text_encoder,
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2)
        logger.info("See-Through loaded: %d modules, VRAM=%.0fMB", len(pipe), vram)

        return cls(
            ld_unet=ld_unet,
            ld_vae=ld_vae,
            ld_trans_vae=trans_vae,
            ld_text_encoder=ld_text_encoder,
            ld_text_encoder_2=ld_text_encoder_2,
            ld_tokenizer=ld_tokenizer,
            ld_tokenizer_2=ld_tokenizer_2,
            ld_scheduler=scheduler,
            mg_unet=mg_unet,
            mg_vae=mg_vae,
            mg_text_encoder=mg_text_encoder,
            mg_tokenizer=mg_tokenizer,
            mg_scheduler=mg_scheduler,
            pipe=pipe,
            co_tenants={"ld_unet": [], "mg_unet": [], "ld_trans_vae": ["ld_vae"]},
        )
