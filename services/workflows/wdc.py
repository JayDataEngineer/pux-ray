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
    width: int = 768,
    height: int = 512,
    **kwargs: Any,
) -> dict[str, Any]:
    """Multi-shot timeline video (LTX Director style).

    Each segment is an independent video generation call:
      {"prompt": "...", "frames": 97,
       "first_frame_b64": "...",   # optional
       "last_frame_b64": "..."}    # optional

    LTXDirector's timeline/camera-guide logic lives in a ComfyUI custom node;
    the Wan2GP LTX handler has no multi-shot or segment concept. Each segment
    is generated separately and returned as a batch.

    Returns:
        dict with "segments" list, each containing "data" (base64 video bytes),
        "media_type", "prompt", "frames", and "segment_index".
    """
    svc = get_service()
    svc.load("ltx2")

    results = []
    for seg_idx, segment in enumerate(segments):
        seg_prompt = segment.get("prompt", "")
        seg_frames = segment.get("frames", 97)
        seg_seed = seed + seg_idx

        infer_kwargs: dict[str, Any] = {
            "input_prompt": seg_prompt,
            "seed": seg_seed,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_num": seg_frames,
            "sampling_steps": 24,
            "guide_scale": 3.0,
        }

        if segment.get("first_frame_b64"):
            infer_kwargs["image_b64"] = segment["first_frame_b64"]
        if segment.get("last_frame_b64"):
            infer_kwargs["image_end_b64"] = segment["last_frame_b64"]

        infer_kwargs.update(kwargs)
        result = svc.infer(infer_kwargs)

        if result.get("status") == "ok":
            results.append({
                "segment_index": seg_idx,
                "prompt": seg_prompt,
                "frames": seg_frames,
                "data": result["data"],
                "media_type": result.get("media_type", "video/mp4"),
                "fps": fps,
            })
        else:
            results.append({
                "segment_index": seg_idx,
                "error": result.get("error", "unknown"),
            })

    return {
        "status": "ok",
        "segments": results,
        "total": len(results),
    }
