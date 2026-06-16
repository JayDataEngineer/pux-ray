"""VACE video step executors — vLLM-Omni backend.

Step types:
  vace_generate  — Full Wan2.2 VACE video generation (T2V or I2V)

Calls the Omni API server at http://omni-14b-vace:8000/v1/images/generations.
The Omni container must be running separately (see scripts/run_omni_14b.sh).

Supports:
  - Text-to-video (prompt only)
  - Image-to-video (prompt + conditioning image)
  - Base mode (18+ steps) and Lightning mode (4 steps)
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Any

import aiohttp
from PIL import Image

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)

OMNI_API_URL = "http://omni-14b-vace:8000/v1/images/generations"
OMNI_TIMEOUT = 600  # 10 min for long generations


class VaceGenerateStep(StepExecutor):
    """Wan2.2 VACE video generation via vLLM-Omni.

    Params:
      input_prompt: Text prompt
      image_b64: Base64 conditioning image (omit for T2V)
      n_prompt: Negative prompt
      seed: Random seed (-1 for random)
      fps: Frames per second
      frame_num: Number of frames
      width/height: Video resolution
      sampling_steps: Denoising steps (4 for Lightning, 18+ for Base)
      guide_scale: CFG guidance scale
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        prompt = params.get("input_prompt", "")
        image_b64 = params.get("image_b64", "")
        n_prompt = params.get("n_prompt", "")
        seed = params.get("seed", -1)
        fps = params.get("fps", 16)
        frame_num = params.get("frame_num", 33)
        width = params.get("width", 832)
        height = params.get("height", 480)
        steps = params.get("sampling_steps", 18)
        guidance = params.get("guide_scale", 5.0)

        # Build request body
        body = {
            "model": "/models/vace-fp8",
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "num_frames": frame_num,
            "steps": steps,
            "guidance_scale": float(guidance),
            "fps": fps,
        }
        if seed >= 0:
            body["seed"] = seed
        if image_b64:
            body["image"] = image_b64
        if n_prompt:
            body["negative_prompt"] = n_prompt

        logger.info(
            "VACE gen: prompt=%.60s frames=%d steps=%d size=%dx%d",
            prompt, frame_num, steps, width, height,
        )

        # Call Omni API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OMNI_API_URL, json=body, timeout=aiohttp.ClientTimeout(total=OMNI_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Omni API error {resp.status}: {err_text[:500]}")
                result = await resp.json()

        elapsed = time.monotonic() - t0

        # Extract base64 video data
        data_list = result.get("data", [])
        if not data_list:
            raise RuntimeError(f"Omni returned no data: {result}")

        b64_json = data_list[0].get("b64_json", "")
        if not b64_json:
            raise RuntimeError(f"Omni returned no b64_json: {data_list[0]}")

        video_bytes = base64.b64decode(b64_json)

        # The Omni API returns a PNG frame — wrap as MP4 if needed
        # For now, return raw bytes (client can interpret based on content-type)
        metadata = {
            "elapsed_s": round(elapsed, 2),
            "frames": frame_num,
            "steps": steps,
            "width": width,
            "height": height,
            "mode": "lightning" if steps <= 4 else "base",
        }
        logger.info("VACE gen done: %.1fs, %d KB output", elapsed, len(video_bytes) // 1024)

        return StepResult(
            data=video_bytes,
            media_type="video/mp4",
            metadata=metadata,
        )
