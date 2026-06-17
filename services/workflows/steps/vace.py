"""VACE video step executors — vLLM-Omni backend.

Step types:
  vace_generate  — Full Wan2.2 VACE video generation (T2V or I2V)

Calls the Omni API video sync endpoint at
  http://<host>:8000/v1/videos/sync

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

logger = logging.getLogger(__name__)

# Model → API server routing.
# - wan2.1-vace-14b-fp8-diffusers: the PROVEN direct-cast FP8 model served by
#   scripts/run_omni_14b.sh (container: omni-14b-vace-fp8). Used by both
#   vace_base (25 steps) and vace_fast (10 steps) workflows.
# - wan2.1-vace-14b-fp8: legacy alias for the same container.
# - wan2.1-vace-14b-fp8-lightning: PENDING. The LightX2V LoRA at
#   hf://lightx2v/Wan2.2-Distill-Loras targets Wan2.2-I2V-A14B (not VACE),
#   so the previous merge produced washed-out output. Routing entry kept
#   so the workflow IaC is ready when a compatible distillation lands.
OMNI_ENDPOINTS = {
    "wan2.1-vace-14b-fp8-diffusers": ("http://omni-14b-vace-fp8:8000",        "/models/vace-fp8"),
    "wan2.1-vace-14b-fp8":           ("http://omni-14b-vace-fp8:8000",        "/models/vace-fp8"),
    "wan2.1-vace-14b-fp8-lightning": ("http://omni-14b-vace-lightning:8000",  "/models/vace-fp8-lightning"),
}
DEFAULT_BASE = "http://omni-14b-vace-fp8:8000"
OMNI_TIMEOUT = 600  # 10 min for long generations


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

        # Resolve API base URL + model path
        ep_info = OMNI_ENDPOINTS.get(model_id)
        if ep_info:
            api_base = ep_info[0]
            model_path = ep_info[1]
        else:
            api_base = DEFAULT_BASE
            model_path = "/models/vace-fp8"

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
