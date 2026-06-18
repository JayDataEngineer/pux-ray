"""Video Director MCP tool — VACE video generation via OMNI-vLLM.

Calls the workflow engine with the appropriate VACE workflow spec
(vace_base, vace_fast, or vace_lightning), waits for completion,
and returns the video artifact as base64.

The workflow engine handles pool resolution via inference_pools.yaml,
which routes wan-vace to the omni-vllm Tier B pool.
"""
from __future__ import annotations

import base64
from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field

from ..workflow_client import WorkflowClient


# ─── VACE mode -> workflow spec mapping ───────────────────────────────────

VACE_SPECS = {
    "base": "vace_base",
    "fast": "vace_fast",
    "lightning": "vace_lightning",
}


def _wf(ctx: Context) -> WorkflowClient:
    client = ctx.lifespan_context.get("workflow_client")
    if client is None:
        raise RuntimeError("Workflow client not initialized")
    return client


def _mode_to_spec(mode: str) -> str:
    spec = VACE_SPECS.get(mode)
    if not spec:
        valid = ", ".join(VACE_SPECS.keys())
        raise ValueError(f"Unknown VACE mode '{mode}'. Valid: {valid}")
    return spec


def _steps_for_mode(mode: str) -> int:
    return {"base": 25, "fast": 10, "lightning": 4}.get(mode, 10)


def _map_params(mode: str, args: dict) -> dict:
    """Map MCP tool arguments to VACE workflow inputs."""
    inputs = {
        "prompt": args.get("prompt", ""),
        "negative_prompt": args.get("negative_prompt", ""),
        "width": int(args.get("width", 640)),
        "height": int(args.get("height", 480)),
        "frames": int(args.get("frames", 81)),
        "steps": int(args.get("steps", 0)) or _steps_for_mode(mode),
        "seed": int(args.get("seed", -1)),
        "fps": int(args.get("fps", 24)),
        "guidance": str(args.get("guidance", 5.0)),
    }
    cond_image = args.get("cond_image_b64") or args.get("cond_image") or args.get("image_b64", "")
    if cond_image:
        inputs["cond_image"] = cond_image
    if not inputs["negative_prompt"]:
        inputs["negative_prompt"] = ""
    return inputs


# ─── Tool ────────────────────────────────────────────────────────────────


async def video_generate(
    mode: Annotated[str, Field(
        description="VACE quality mode: 'base' (25 steps), 'fast' (10 steps, 2.5x speedup), 'lightning' (4 steps — pending)",
        default="fast",
    )] = "fast",
    prompt: Annotated[str, Field(description="Text prompt describing the video content")] = "",
    cond_image_b64: Annotated[str | None, Field(
        description="Optional base64-encoded conditioning image for Image-to-Video mode. Omit for Text-to-Video.",
    )] = None,
    negative_prompt: Annotated[str, Field(description="Negative prompt to exclude unwanted content", default="")] = "",
    width: Annotated[int, Field(description="Video width in pixels (max 832 for 24GB VRAM)", default=640, ge=256, le=1280)] = 640,
    height: Annotated[int, Field(description="Video height in pixels (max 480 for 24GB VRAM)", default=480, ge=256, le=720)] = 480,
    frames: Annotated[int, Field(description="Number of frames (33-121, more = longer video)", default=81, ge=9, le=201)] = 81,
    fps: Annotated[int, Field(description="Frames per second (16, 24, or 30)", default=24)] = 24,
    seed: Annotated[int, Field(description="Random seed (-1 for random)", default=-1)] = -1,
    guidance: Annotated[float, Field(description="CFG guidance scale (3.0-7.0)", default=5.0, ge=1.0, le=15.0)] = 5.0,
    steps: Annotated[int | None, Field(
        description="Override denoising steps. Defaults: base=25, fast=10, lightning=4.",
        default=None, ge=1, le=50,
    )] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Generate video using VACE on OMNI-vLLM.

    Creates a video from a text prompt (Text-to-Video) or from a text prompt
    plus conditioning image (Image-to-Video). Uses the vLLM-Omni Tier B pool
    with Wan2.1 VACE 14B FP8.

    Returns base64-encoded MP4 video data with generation metadata.

    Quality/speed tradeoffs:
      - base (25 steps): full quality, ~165s for 33 frames
      - fast (10 steps): 2.5x speedup, ~72s for 33 frames
      - lightning (4 steps): fastest, pending compatible checkpoint
      - Teacache (server-side): ~70% speedup when enabled on container

    Known good VRAM configurations (24GB RTX 4090):
      - 640x480 x 81 frames x 25 steps: works (~510s)
      - 640x480 x 33 frames x 25 steps: works (~165s)
      - 704x480 x 33 frames: works (~120s)
      - 832x480 x 33 frames: edge of VRAM, may OOM
    """
    logger.info("video_generate: mode={} prompt={:.60s} {}x{} frames={}",
                mode, prompt, width, height, frames)

    if not prompt:
        return {"status": "error", "error": "Prompt is required"}

    spec_name = _mode_to_spec(mode)
    inputs = _map_params(mode, {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
        "fps": fps,
        "guidance": guidance,
        "cond_image_b64": cond_image_b64,
    })

    client = _wf(ctx)
    try:
        run = await client.run_and_wait(spec_name, inputs)
    except Exception as e:
        logger.error("VACE workflow failed: {}", e)
        return {"status": "error", "error": f"Workflow error: {e}"}

    status = run.get("status")
    if status != "completed":
        step_states = run.get("step_states", {})
        errors = []
        for sid, ss in step_states.items():
            if ss.get("status") == "failed":
                errors.append(f"{sid}: {ss.get('error', 'unknown')}")
        return {
            "status": "error",
            "error": f"Workflow {status}: {'; '.join(errors) or 'unknown error'}",
        }

    # Fetch video artifact from the 'generate' step
    video_b64 = None
    try:
        video_bytes = await client.get_artifact_data(
            spec_name, run["run_id"], "generate", "output.mp4",
        )
        video_b64 = base64.b64encode(video_bytes).decode()
    except Exception as e:
        logger.warning("Could not fetch video artifact: {}", e)

    # Gather metadata
    step_info = run.get("step_states", {}).get("generate", {})
    elapsed_s = step_info.get("duration_ms", 0) / 1000 if step_info else 0

    logger.info("VACE gen done: {} KB video, {:.1f}s",
                len(video_b64 or "") // 4 * 3 // 1024, elapsed_s)

    return {
        "status": "success" if video_b64 else ("error" if status == "failed" else "unknown"),
        "video_data": video_b64,
        "format": "mp4",
        "run_id": run.get("run_id"),
        "mode": mode,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "elapsed_s": round(elapsed_s, 1),
    }
