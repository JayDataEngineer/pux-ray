"""SGLang configuration manager — replaces mmGP AND diffusers VRAM APIs.

SGLang handles ALL VRAM management internally:
  - Model loading (from_pretrained internally)
  - VRAM offloading (--performance-mode memory/speed)
  - Quantization (--quantization fp8/gguf)
  - Attention backends (SageAttention, FlashAttention — baked into SGLang)
  - Cache acceleration (Cache-DiT, TeaCache — built into SGLang)
  - Sleep/wake for scale-to-zero

This module maps model names to optimal SGLang serve flags.
NO diffusers APIs. NO group_offload. NO layerwise_casting in our code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig:
    """Configuration for serving a model via SGLang."""
    performance_mode: str = "auto"       # auto | speed | memory
    quantization: Optional[str] = None   # None | fp8 | gguf | modelopt_fp8
    dtype: str = "bfloat16"
    server_warmup: bool = True
    extra_args: dict = None

    def to_args(self) -> list[str]:
        """Convert to sglang serve CLI arguments."""
        args = [
            "--performance-mode", self.performance_mode,
            "--dtype", self.dtype,
        ]
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        if not self.server_warmup:
            args.append("--skip-server-warmup")
        return args


# ─── Per-model SGLang configurations ───────────────────────────────────────────

# Image models — small enough for speed mode on 24GB
IMAGE_CONFIGS = {
    "z-image": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    "z-image-turbo": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    "flux2-klein-4b": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    "anima": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    "qwen-image": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    # FLUX.1 is large — needs memory mode on 24GB
    "flux-schnell": SGLangConfig(performance_mode="memory", quantization="fp8"),
    "flux-dev": SGLangConfig(performance_mode="memory", quantization="fp8"),
}

# Video models
VIDEO_CONFIGS = {
    "ltx-video": SGLangConfig(performance_mode="speed", dtype="bfloat16"),
    "wan-t2v": SGLangConfig(performance_mode="memory", quantization="fp8"),
    "wan-i2v": SGLangConfig(performance_mode="memory", quantization="fp8"),
}

ALL_CONFIGS = {**IMAGE_CONFIGS, **VIDEO_CONFIGS}


def get_config(model_name: str) -> SGLangConfig:
    """Get the SGLang configuration for a model."""
    return ALL_CONFIGS.get(model_name, SGLangConfig())


def release_vram(sglang_url: str = "http://localhost:30010") -> bool:
    """Tell SGLang to release VRAM (sleep mode).

    SGLang moves all weights to CPU and clears CUDA cache.
    VRAM drops to ~250-400MB (CUDA context only).
    """
    import httpx
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{sglang_url}/release_memory_occupation",
                json={"tags": ["weights", "cache"]},
            )
            logger.info("SGLang: VRAM released (sleep mode)")
            return resp.status_code == 200
    except Exception as e:
        logger.warning("SGLang: sleep failed: %s", e)
        return False


def wake_vram(sglang_url: str = "http://localhost:30010") -> bool:
    """Tell SGLang to wake from sleep (restore VRAM).

    Moves weights back from CPU to GPU. Takes ~0.5s for FP8 models
    over PCIe Gen4.
    """
    import httpx
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{sglang_url}/resume_memory_occupation")
            logger.info("SGLang: VRAM restored (wake mode)")
            return resp.status_code == 200
    except Exception as e:
        logger.warning("SGLang: wake failed: %s", e)
        return False
