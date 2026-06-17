"""VACE video step executors — vLLM-Omni backend.

Step types:
  vace_generate  — Full Wan2.2 VACE video generation (T2V or I2V)

Calls the Omni API video sync endpoint at the URL resolved by the inference
pool system (services.inference.dispatch). The pool system's
inference_pools.yaml declares wan-vace on the omni-vllm pool with
api.generate pointing at /v1/videos/sync (set in the launcher).

Uses multipart form-data (required by the video API) to send all
parameters. Returns raw MP4 bytes directly (no base64 wrapping).

Supports:
  - Text-to-video (prompt only)
  - Image-to-video (prompt + conditioning image)
  - TeaCache: server-side toggle via OMNI_TEACACHE_THRESH env var
    (set=0.01 on container for ~70% speedup with negligible quality loss)
  - Base mode (18+ steps), Fast mode (6-12 steps)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiohttp

from . import StepExecutor, StepContext, StepResult
from services.inference.dispatch import resolve_step

logger = logging.getLogger(__name__)

# Legacy fallback map — only used if the pool config is unavailable.
# The pool system is the source of truth for routing.
_FALLBACK_BASES = {
    "wan2.1-vace-14b-fp8-diffusers": "http://omni-14b-vace-fp8:8000",
    "wan2.1-vace-14b-fp8":           "http://omni-14b-vace-fp8:8000",
    "wan2.1-vace-14b-fp8-lightning": "http://omni-14b-vace-lightning:8000",
}
_FALLBACK_MODEL_PATHS = {
    "wan2.1-vace-14b-fp8-diffusers": "/models/vace-fp8",
    "wan2.1-vace-14b-fp8":           "/models/vace-fp8",
    "wan2.1-vace-14b-fp8-lightning": "/models/vace-fp8-lightning",
}
_DEFAULT_BASE = "http://omni-14b-vace-fp8:8000"
_DEFAULT_MODEL_PATH = "/models/vace-fp8"
OMNI_TIMEOUT = 600  # 10 min for long generations


def _resolve_api_base(model_id: str) -> tuple[str, str]:
    """Resolve model → (api_base_url, model_path).

    The base URL comes from the pool system (services.inference.dispatch).
    The in-container model path is the bind-mount target declared by the
    launch script (e.g. /models/vace-fp8), which isn't in the YAML — the
    legacy map below stays authoritative for that.

    Falls back to the legacy map for both fields if the pool config is
    unavailable.
    """
    base_url: str | None = None
    try:
        # wan-vace in the YAML uses model name 'wan-vace'; accept the
        # legacy HuggingFace-style ids too by mapping to the canonical name.
        canonical = "wan-vace" if "vace" in model_id else model_id
        plan = resolve_step(service=None, model=canonical, action="generate")
        if plan:
            # Strip the /v1/... path to get the base URL.
            base_url = plan[0].url.rsplit("/v1/", 1)[0]
    except (ValueError, FileNotFoundError) as e:
        logger.warning(
            "Pool resolution failed for vace %r — using legacy map: %s",
            model_id, e
        )

    # Model path is the in-container mount, declared by the launch script.
    # Not derivable from YAML — the legacy map stays authoritative here.
    path = _FALLBACK_MODEL_PATHS.get(model_id, _DEFAULT_MODEL_PATH)
    if base_url is None:
        base_url = _FALLBACK_BASES.get(model_id, _DEFAULT_BASE)
    return base_url, path


class VaceGenerateStep(StepExecutor):
    """Wan2.2 VACE video generation via vLLM-Omni video sync API.

    Uses POST /v1/videos/sync (multipart form-data) which returns raw
    MP4 bytes — no base64 encoding, no PNG wrapping.

    Supports three quality tiers, all served from the same PROVEN
    direct-cast FP8 base model (wan2.1-vace-14b-fp8-diffusers):
      - vace_base:    25 steps, full quality (~165s for 33-frame 640x480)
                       with TeaCache (0.01): ~49s (70% speedup)
      - vace_fast:    10 steps, ~2.5x speedup (~72s)
                       with TeaCache (0.01): ~35s
      - vace_lightning: PENDING — needs compatible Wan2.1-VACE distillation

    TeaCache is a SERVER-SIDE toggle (OMNI_TEACACHE_THRESH env var on the
    container). When the server has TeaCache enabled, all requests benefit
    automatically. The step executor has no per-request TeaCache control.

    Routes to the correct Omni API endpoint based on the *_model parameter.

    Params:
      input_prompt: Text prompt
      image_b64: Base64 conditioning image (omit for T2V)
      n_prompt: Negative prompt
      seed: Random seed (-1 for random)
      fps: Frames per second
      frame_num: Number of frames
      width/height: Video resolution
      sampling_steps: Denoising steps (10 for Fast, 25 for Base)
      guide_scale: CFG guidance scale
      *_model: Model identifier (auto-injected from workflow spec)
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        prompt = params.get("input_prompt", "")
        image_b64 = params.get("image_b64", "")
        n_prompt = params.get("n_prompt", "")
        seed = params.get("seed", -1)
        fps = params.get("fps", 16)
        frame_num = params.get("frame_num", 33)
        width = params.get("width", 640)
        height = params.get("height", 480)
        steps = params.get("sampling_steps", 25)
        guidance = params.get("guide_scale", 5.0)
        model_id = params.get("_model", "wan2.1-vace-14b-fp8-diffusers")

        # Resolve Omni API base URL + in-container model path via the pool system.
        api_base, model_path = _resolve_api_base(model_id)
        api_url = f"{api_base}/v1/videos/sync"

        logger.info(
            "VACE gen [%s]: prompt=%.60s frames=%d steps=%d size=%dx%d",
            model_id, prompt, frame_num, steps, width, height,
        )

        # Build multipart form-data
        form = aiohttp.FormData()
        form.add_field("model", model_path)
        form.add_field("prompt", prompt)
        form.add_field("width", str(width))
        form.add_field("height", str(height))
        form.add_field("num_frames", str(frame_num))
        form.add_field("num_inference_steps", str(steps))
        form.add_field("fps", str(fps))
        form.add_field("guidance_scale", str(guidance))

        if seed >= 0:
            form.add_field("seed", str(seed))
        if n_prompt:
            form.add_field("negative_prompt", n_prompt)
        if image_b64:
            # Wrap base64 image as a data URI for the video API
            image_ref = json.dumps({"image_url": f"data:image/png;base64,{image_b64}"})
            form.add_field("image_reference", image_ref)

        logger.info("Calling Omni video sync API at %s", api_url)

        # Call Omni video sync API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                data=form,
                timeout=aiohttp.ClientTimeout(total=OMNI_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Omni video API error {resp.status}: {err_text[:500]}")
                video_bytes = await resp.read()

                # Read inference time from response headers
                inference_time_s = resp.headers.get("X-Inference-Time-S")
                peak_memory_mb = resp.headers.get("X-Peak-Memory-MB")

        elapsed = time.monotonic() - t0

        metadata = {
            "elapsed_s": round(elapsed, 2),
            "frames": frame_num,
            "steps": steps,
            "width": width,
            "height": height,
            # Mode bucketing: <=4 = lightning (when available), <=12 = fast,
            # else base. Useful for frontend UX and timing estimates.
            "mode": "lightning" if steps <= 4 else ("fast" if steps <= 12 else "base"),
        }
        if inference_time_s:
            metadata["inference_time_s"] = float(inference_time_s)
        if peak_memory_mb:
            metadata["peak_memory_mb"] = float(peak_memory_mb)

        logger.info(
            "VACE gen done: %.1fs (model %.1fs), %d KB MP4 output",
            elapsed,
            float(inference_time_s) if inference_time_s else 0.0,
            len(video_bytes) // 1024,
        )

        return StepResult(
            data=video_bytes,
            media_type="video/mp4",
            metadata=metadata,
        )
