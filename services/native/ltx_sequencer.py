"""LTX-Video sequencer via ltx-pipelines (Lightricks native).

NO diffusers. Uses ltx-pipelines package directly for:
  - TI2VidTwoStagesPipeline (dual-stage: half-res → 2x upscale)
  - ICLoraPipeline (video-to-video with IC-LoRA control)
  - Guiding vs Replacing latent injection
  - IC-LoRA spatial-temporal masking

Falls back to SGLang video API for standard generation.

Requires: pip install ltx-pipelines ltx-core
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

VAE_TEMPORAL_COMPRESSION = 8
SUPPORTED_PROFILES = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97]
COMMON_PROFILES = [9, 17, 33, 65, 97]


@dataclass
class KeyframeInput:
    """Keyframe for timeline injection."""
    image: Any               # PIL Image or tensor
    frame_index: int         # Frame position in output
    strength: float = 1.0    # [0, 1]
    mode: str = "guide"      # "guide" (smooth) or "replace" (hard cut)


@dataclass
class GenerationConfig:
    """LTX generation parameters."""
    prompt: str
    num_frames: int = 25
    num_inference_steps: int = 30
    guidance_scale: float = 6.0
    width: int = 768
    height: int = 512
    seed: int = -1
    negative_prompt: str = ""
    keyframes: List[KeyframeInput] = field(default_factory=list)
    # IC-LoRA
    ic_lora_mask: Optional[Any] = None
    ic_lora_strength: float = 1.0
    # Dual-stage
    use_two_stage: bool = True
    stage2_steps: int = 4
    # Quality
    fp8: bool = False


class LTXSequencer:
    """LTX-Video sequencer using ltx-pipelines (Lightricks native).

    For standard generation: delegates to SGLang's video API.
    For keyframe/IC-LoRA: uses ltx-pipelines directly.

    Usage:
        # Standard (via SGLang)
        seq = LTXSequencer(sglang_url="http://localhost:30010")
        video = seq.generate(GenerationConfig(prompt="dragon flying"))

        # Advanced keyframes (via ltx-pipelines)
        seq = LTXSequencer(ltx_model_path="/models/ltx-video")
        seq.load_pipeline()
        video = seq.generate(GenerationConfig(
            prompt="dragon flying",
            keyframes=[KeyframeInput(first_frame, 0, mode="guide", strength=0.8)],
        ))
    """

    def __init__(
        self,
        sglang_url: str = "http://localhost:30010",
        ltx_model_path: str | None = None,
        device: str = "cuda",
    ):
        self.sglang_url = sglang_url
        self.ltx_model_path = ltx_model_path
        self.device = torch.device(device)

        # ltx-pipelines components (lazy-loaded)
        self._pipeline = None
        self._vae = None
        self._dit = None
        self._loaded = False

    def load_pipeline(self) -> None:
        """Load ltx-pipelines for advanced features (keyframes, IC-LoRA).

        This is ONLY needed for keyframe injection or IC-LoRA masking.
        Standard generation uses SGLang and doesn't need this.
        """
        if self._loaded:
            return

        if not self.ltx_model_path:
            raise ValueError("ltx_model_path required for ltx-pipelines mode")

        logger.info("LTX: loading ltx-pipelines from %s", self.ltx_model_path)

        try:
            # Try importing ltx-pipelines components
            from ltx_video.pipelines import TI2VidTwoStagesPipeline
            from ltx_video.models.transformers import LTXVideoTransformer
            from ltx_video.models.vae import AutoencoderKLLTXVideo

            # Load components
            self._dit = LTXVideoTransformer.from_pretrained(
                self.ltx_model_path, subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            self._vae = AutoencoderKLLTXVideo.from_pretrained(
                self.ltx_model_path, subfolder="vae",
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            self._pipeline = TI2VidTwoStagesPipeline(
                dit_model=self._dit,
                vae=self._vae,
                device=self.device,
            )

            self._loaded = True
            logger.info("LTX: ltx-pipelines loaded (TI2VidTwoStagesPipeline)")

        except ImportError:
            logger.warning(
                "ltx-pipelines not installed. Install with:\n"
                "  pip install ltx-pipelines ltx-core\n"
                "  or: pip install git+https://github.com/Lightricks/ltx-pipelines.git\n"
                "Falling back to SGLang for all LTX generation."
            )
            self._loaded = False

    def generate(self, config: GenerationConfig) -> Any:
        """Generate video.

        If keyframes or IC-LoRA mask provided: uses ltx-pipelines (requires load_pipeline()).
        Otherwise: uses SGLang video API (faster, optimized kernels).
        """
        if config.keyframes or config.ic_lora_mask:
            return self._generate_advanced(config)
        else:
            return self._generate_sglang(config)

    def _generate_sglang(self, config: GenerationConfig) -> Any:
        """Standard generation via SGLang HTTP API."""
        import httpx

        body = {
            "model": "ltx-video",
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
        }
        if config.seed >= 0:
            body["seed"] = config.seed
        if config.negative_prompt:
            body["negative_prompt"] = config.negative_prompt

        logger.info("LTX: generating via SGLang (%d frames, %d steps)",
                    config.num_frames, config.num_inference_steps)

        with httpx.Client(timeout=600) as client:
            resp = client.post(f"{self.sglang_url}/v1/videos/generations", json=body)

        if resp.status_code != 200:
            raise RuntimeError(f"SGLang video generation failed: {resp.status_code} {resp.text[:200]}")

        return resp.json()

    def _generate_advanced(self, config: GenerationConfig) -> Any:
        """Advanced generation with keyframes via ltx-pipelines.

        Implements:
        1. Replacing latents (hard keyframe overwrite)
        2. Guiding latents (Gaussian decay additive signal)
        3. IC-LoRA attention masking
        4. Dual-stage execution (Stage 1 half-res → Stage 2 upscale)
        """
        if not self._loaded:
            self.load_pipeline()
            if not self._loaded:
                logger.warning("LTX: ltx-pipelines not available — falling back to SGLang")
                return self._generate_sglang(config)

        logger.info("LTX: advanced generation with %d keyframes", len(config.keyframes))

        # Use ltx-pipelines image_conditionings API for keyframe injection
        # The ltx-pipelines package provides native helpers:
        # - image_conditionings_by_replacing_latent
        # - image_conditionings_by_adding_guiding_latent

        try:
            from ltx_video.utils import (
                image_conditionings_by_replacing_latent,
                image_conditionings_by_adding_guiding_latent,
            )

            # Build image conditionings from keyframes
            replacing_keyframes = [kf for kf in config.keyframes if kf.mode == "replace"]
            guiding_keyframes = [kf for kf in config.keyframes if kf.mode == "guide"]

            image_conditionings = {}

            # Replacing latents — hard overwrite at specific frames
            for kf in replacing_keyframes:
                image_conditionings_by_replacing_latent(
                    image_conditionings,
                    image=kf.image,
                    frame_index=kf.frame_index,
                    vae=self._vae,
                    strength=kf.strength,
                )

            # Guiding latents — smooth additive influence
            for kf in guiding_keyframes:
                image_conditionings_by_adding_guiding_latent(
                    image_conditionings,
                    image=kf.image,
                    frame_index=kf.frame_index,
                    vae=self._vae,
                    strength=kf.strength,
                )

            # IC-LoRA mask processing
            if config.ic_lora_mask:
                image_conditionings["conditioning_attention_mask"] = self._process_mask(
                    config.ic_lora_mask, config
                )
                image_conditionings["conditioning_attention_strength"] = config.ic_lora_strength

            # Generate via ltx-pipelines with conditionings
            video = self._pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt or None,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                width=config.width,
                height=config.height,
                num_frames=config.num_frames,
                image_conditionings=image_conditionings,
                seed=config.seed if config.seed >= 0 else None,
            )

            return video

        except ImportError:
            logger.warning(
                "LTX: ltx-pipelines keyframe API not available.\n"
                "Install latest: pip install --upgrade ltx-pipelines\n"
                "Falling back to SGLang."
            )
            return self._generate_sglang(config)

    def _process_mask(self, mask_input, config: GenerationConfig):
        """Process IC-LoRA spatial-temporal mask.

        1. Load mask frames
        2. Grayscale (mean across channels)
        3. Normalize [0, 1]
        4. Downsample to latent space with causal temporal alignment
        5. Scale by conditioning strength (γ)
        """
        try:
            from PIL import Image
            import torch.nn.functional as F

            if isinstance(mask_input, Image.Image):
                mask_input = [mask_input]

            mask_frames = []
            for m in mask_input[:config.num_frames]:
                if isinstance(m, Image.Image):
                    m = m.resize((config.width, config.height), Image.LANCZOS)
                    arr = np.array(m.convert("RGB")).astype(np.float32)
                    grayscale = arr.mean(axis=2) / 255.0
                    mask_frames.append(grayscale)

            if not mask_frames:
                return None

            mask_tensor = torch.from_numpy(np.stack(mask_frames))
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            # Downsample to latent dimensions
            latent_h = config.height // 32  # LTX spatial compression
            latent_w = config.width // 32
            num_latent_frames = (config.num_frames - 1) // VAE_TEMPORAL_COMPRESSION + 1

            mask_down = F.interpolate(
                mask_tensor, size=(config.num_frames, latent_h, latent_w),
                mode="trilinear", align_corners=False,
            )

            # Causal temporal compression
            mask_latent = mask_down[:, :, ::VAE_TEMPORAL_COMPRESSION][:, :, :num_latent_frames]
            mask_latent = mask_latent * config.ic_lora_strength

            return mask_latent

        except Exception as e:
            logger.warning("LTX: mask processing failed: %s", e)
            return None

    @staticmethod
    def get_nearest_profile(frames: int) -> int:
        """Map frame count to nearest discrete profile."""
        for p in COMMON_PROFILES:
            if p >= frames:
                return p
        return COMMON_PROFILES[-1]

    @staticmethod
    def pad_latents(latent: torch.Tensor, target_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad latents to discrete profile with attention mask."""
        B, C, F, H, W = latent.shape
        pad_len = target_frames - F

        if pad_len > 0:
            padding = torch.zeros(B, C, pad_len, H, W,
                                  dtype=latent.dtype, device=latent.device)
            padded = torch.cat([latent, padding], dim=2)
        else:
            padded = latent

        mask = torch.zeros(B, 1, target_frames, H, W,
                           dtype=latent.dtype, device=latent.device)
        mask[:, :, :F] = 1.0

        return padded, mask

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Release ltx-pipelines resources."""
        self._pipeline = None
        self._dit = None
        self._vae = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LTX: unloaded")
