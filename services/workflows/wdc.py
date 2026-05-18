"""WhatDreamsCost workflow functions — LTX Video conditioning orchestration.

Each function maps to one WDC ComfyUI workflow JSON. WDC workflows are
conditioning strategies on top of LTX Video (already a Wan2GP built-in).

Workflows available:
  ltx_fflf_2stage  — Image-to-video with first/last frame conditioning
  ltx_fflf_3stage  — Image-to-video with 3-stage first/last frame + upscale
  ltx_audio        — Image-to-video with audio conditioning
  timeline         — Multi-shot timeline video
"""
from __future__ import annotations

import logging
from typing import Any

from services.workflows.base import get_service

logger = logging.getLogger(__name__)


WORKFLOWS = [
    {"id": "wdc/ltx-fflf-2stage",
     "description": "Image-to-video with first/last frame conditioning"},
    {"id": "wdc/ltx-fflf-3stage",
     "description": "Image-to-video with 3-stage first/last frame + upscale"},
    {"id": "wdc/ltx-audio",
     "description": "Image-to-video with audio conditioning"},
    {"id": "wdc/timeline",
     "description": "Multi-shot timeline video (LTX Director)"},
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


def ltx_fflf_2stage(
    prompt: str,
    first_frame_b64: str,
    last_frame_b64: str | None = None,
    seed: int = 42,
    width: int = 768,
    height: int = 512,
    frames: int = 97,
    **kwargs: Any,
) -> dict[str, Any]:
    """Image-to-video with first/last frame conditioning.

    First frame is the start frame. Optional last frame is the end frame.
    LTX Video interpolates between them over `frames` frames.
    """
    svc = get_service()
    svc.load("ltx2")

    infer_kwargs: dict[str, Any] = {
        "input_prompt": prompt,
        "image_b64": first_frame_b64,
        "seed": seed,
        "width": width,
        "height": height,
        "frame_num": frames,
        "sampling_steps": 24,
        "guide_scale": 3.0,
        "fps": 24,
    }

    if last_frame_b64:
        infer_kwargs["image_end_b64"] = last_frame_b64

    infer_kwargs.update(kwargs)
    result = svc.infer(infer_kwargs)
    return result


def ltx_fflf_3stage(
    prompt: str,
    first_frame_b64: str,
    last_frame_b64: str | None = None,
    seed: int = 42,
    width: int = 768,
    height: int = 512,
    frames: int = 97,
    **kwargs: Any,
) -> dict[str, Any]:
    """Image-to-video with 3-stage generation (3x upscale).

    Same as 2-stage but enables intermediate spatial upscaling
    for higher resolution output.
    """
    infer_kwargs: dict[str, Any] = dict(
        prompt=prompt,
        first_frame_b64=first_frame_b64,
        last_frame_b64=last_frame_b64,
        seed=seed,
        width=width,
        height=height,
        frames=frames,
        upscale_factor=3.0,
        **kwargs,
    )
    return ltx_fflf_2stage(**infer_kwargs)


def ltx_audio(
    prompt: str,
    first_frame_b64: str,
    audio_b64: str,
    seed: int = 42,
    width: int = 768,
    height: int = 512,
    frames: int = 97,
    **kwargs: Any,
) -> dict[str, Any]:
    """Image-to-video with audio conditioning.

    Audio drives video timing for lip-sync or rhythmic generation.
    """
    svc = get_service()
    svc.load("ltx2")

    infer_kwargs: dict[str, Any] = {
        "input_prompt": prompt,
        "image_b64": first_frame_b64,
        "audio_b64": audio_b64,
        "seed": seed,
        "width": width,
        "height": height,
        "frame_num": frames,
        "sampling_steps": 24,
        "guide_scale": 3.0,
        "fps": 24,
    }

    infer_kwargs.update(kwargs)
    result = svc.infer(infer_kwargs)
    return result


def timeline(
    segments: list[dict[str, Any]],
    seed: int = 42,
    fps: int = 24,
    **kwargs: Any,
) -> dict[str, Any]:
    """Multi-shot timeline video (LTX Director style).

    Each segment defines a shot with prompt, duration, and optional
    camera guidance. Rendered as a single continuous video.
    """
    svc = get_service()
    svc.load("ltx2")

    infer_kwargs: dict[str, Any] = {
        "input_prompt": segments[0].get("prompt", "") if segments else "",
        "seed": seed,
        "fps": fps,
        "segments": segments,
    }

    infer_kwargs.update(kwargs)
    result = svc.infer(infer_kwargs)
    return result
