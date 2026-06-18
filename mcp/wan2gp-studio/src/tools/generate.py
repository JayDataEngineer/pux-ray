"""Unified inference tool — one MCP tool, one code path.

Routes to /v1/run which handles:
  - Named DAG pipelines: {"pipeline": "vnccs/pose-edit", "params": {...}}
  - Single service calls: {"service": "native", "model": "z_image", ...}

Provide either pipeline or service, not both. Params are passed through.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

# All known DAG pipeline IDs and their parameters
_PIPELINE_CATALOG = {
    "tech-noir/generate": {
        "description": "Z-Image character generation from text prompt",
        "params": {"prompt": "string (required)", "quality": "turbo|standard", "seed": "int", "width": "int", "height": "int"},
    },
    "tech-noir/video": {
        "description": "LTX Video assembly from image + prompt",
        "params": {"image_b64": "string", "prompt": "string", "seed": "int", "fps": "int", "frames": "int", "width": "int", "height": "int"},
    },
    "tech-noir/trellis": {
        "description": "TRELLIS 3D model generation from image",
        "params": {"image_b64": "string", "seed": "int", "resolution": "string"},
    },
    "vnccs/pose-edit": {
        "description": "Character + BodyMesh pose → posed character image",
        "params": {"character_image_b64": "string", "rotations": "dict", "model_rotation_y": "float", "seed": "int"},
    },
    "vnccs/char-sheet": {
        "description": "Text → character base sheet (Z-Image + QWEN refine)",
        "params": {"prompt": "string", "seed": "int"},
    },
    "vnccs/clone": {
        "description": "Reference character → cloned variant",
        "params": {"character_character_image_b64": "string", "prompt": "string"},
    },
    "lance/gemx-mesh": {
        "description": "GEM-X video-based SOMA mesh extraction — video → per-frame 77-joint poses",
        "params": {"video_path": "string (required)", "cache_key": "string"},
    },
    "lance/extract-frames": {
        "description": "Extract frames from video via ffmpeg",
        "params": {"video_path": "string", "output_dir": "string", "max_frames": "int"},
    },
    "lance/pair-frames": {
        "description": "Pair video frames for training (Frame_A ≠ Frame_B, prevents identity mapping)",
        "params": {"frames_dir": "string", "frame_offset": "int", "seed": "int"},
    },
    "lance/full-pipeline": {
        "description": "Complete training data generation — videos → kohya_ss control_dirs dataset. "
                       "GEM-X mesh + DWPose skeleton + frame pairing. No teacher model.",
        "params": {"video_dir": "string (required)", "output_dir": "string",
                   "max_pairs": "int", "frame_offset": "int", "seed": "int"},
    },
}


async def run(
    pipeline: Annotated[str | None, Field(
        description=f"DAG pipeline to execute. One of: {', '.join(sorted(_PIPELINE_CATALOG))}",
        default=None,
    )] = None,
    service: Annotated[str | None, Field(
        description="Service name from the registry. Use list_models to discover.",
        default=None,
    )] = None,
    params: Annotated[dict[str, Any] | None, Field(
        description="Parameters passed through. For pipelines: DAG-specific keys. "
                    "For services: model, prompt, text, image_b64, audio_b64, seed, "
                    "steps, guidance, width, height, frames, negative_prompt, voice. "
                    "All optional — sensible defaults applied server-side.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Run any DAG pipeline or registered service. Single entry point for all inference.

    Use pipeline= for multi-step DAGs (pose-edit, generate, video).
    Use service= + model= for single-model calls (native with z_image, ltx2, etc).

    Returns standard dict with status and data/media_type fields.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    if pipeline:
        # DAG pipeline call
        payload = {"pipeline": pipeline, "params": params or {}}
    elif service:
        # Single service call
        payload = {"service": service, **(params or {})}
    else:
        raise ValueError("Provide 'pipeline' or 'service' (not both)")

    return await client.invoke(payload)
