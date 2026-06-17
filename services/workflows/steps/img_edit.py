"""Qwen-Image-Edit step executor — vLLM-Omni backend.

Step types:
  img_edit  — Qwen-Image-Edit-2511 image editing (instruction-based)

Calls the Omni image edits API at
  http://<host>:8000/v1/images/edits

Uses multipart form-data (OpenAI DALL-E compatible) to send the input
image and parameters. Returns base64-encoded PNG bytes.

Supports:
  - Single-image editing (Qwen-Image-Edit / Qwen-Image-Edit-2511)
  - Multi-image editing (Qwen-Image-Edit-2511)
  - Cache-DiT acceleration (server-side, enabled via --cache-backend)
  - Layerwise CPU offload for 24GB VRAM
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import aiohttp

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)

# Model → API server routing.
# All Qwen-Image-Edit models served from the same container.
OMNI_ENDPOINTS = {
    "qwen-image-edit-2511": ("http://omni-qwen-img-edit:8000", "/models/qwen-img-edit"),
    "qwen-image-edit":      ("http://omni-qwen-img-edit:8000", "/models/qwen-img-edit"),
}
DEFAULT_BASE = "http://omni-qwen-img-edit:8000"
OMNI_TIMEOUT = 300  # 5 min for slow layerwise-offload inference


class ImageEditStep(StepExecutor):
    """Qwen-Image-Edit image editing via vLLM-Omni image edits API.

    Uses POST /v1/images/edits (multipart form-data, OpenAI DALL-E compatible)
    which returns JSON with base64-encoded image.

    Supports:
      - qwen-image-edit-2511: Latest 20B MMDiT with built-in LoRA support,
        multi-image editing, improved consistency.
      - qwen-image-edit: Original single-image editing model.

    Server-side optimizations (configured in launch script):
      - Cache-DiT backend: ~2.38x speedup (block-level caching + TaylorSeer)
      - Layerwise CPU offload: enables 20B model on 24GB VRAM
      - VAE tiling/slicing: prevents OOM at high resolutions

    Params:
      input_prompt: Edit instruction text
      image_b64: Base64-encoded input image (PNG)
      image_b64_2: Optional second image for multi-image editing
      n_prompt: Negative prompt
      seed: Random seed (-1 for random)
      width/height: Output image resolution
      sampling_steps: Denoising steps
      guidance_scale: CFG guidance scale
      mask_b64: Optional base64-encoded mask (white=edit region)
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        prompt = params.get("input_prompt", "")
        image_b64 = params.get("image_b64", "")
        image_b64_2 = params.get("image_b64_2", "")  # multi-image input
        n_prompt = params.get("n_prompt", "")
        seed = params.get("seed", -1)
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("sampling_steps", 40)
        guidance = params.get("guidance_scale", 1.5)
        mask_b64 = params.get("mask_b64", "")
        model_id = params.get("_model", "qwen-image-edit-2511")

        # Resolve API base URL
        ep_info = OMNI_ENDPOINTS.get(model_id)
        api_base = ep_info[0] if ep_info else DEFAULT_BASE
        api_url = f"{api_base}/v1/images/edits"

        logger.info(
            "Image edit [%s]: prompt=%.60s size=%dx%d steps=%d",
            model_id, prompt, width, height, steps,
        )

        # Build multipart form-data
        form = aiohttp.FormData()
        form.add_field("model", model_id)
        form.add_field("prompt", prompt)
        form.add_field("size", f"{width}x{height}")
        form.add_field("num_inference_steps", str(steps))
        form.add_field("guidance_scale", str(guidance))

        if seed >= 0:
            form.add_field("seed", str(seed))
        if n_prompt:
            form.add_field("negative_prompt", n_prompt)
        if mask_b64:
            form.add_field("mask_image", mask_b64)

        # Add image(s) — decode base64 and send as raw file bytes
        # (The /v1/images/edits endpoint expects multipart file upload)
        if image_b64:
            raw_bytes = base64.b64decode(image_b64)
            form.add_field("image", raw_bytes, filename="input.png",
                           content_type="image/png")
        if image_b64_2:
            raw_bytes_2 = base64.b64decode(image_b64_2)
            form.add_field("image", raw_bytes_2, filename="input2.png",
                           content_type="image/png")

        logger.info("Calling Omni image edits API at %s", api_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                data=form,
                timeout=aiohttp.ClientTimeout(total=OMNI_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(
                        f"Omni image edit API error {resp.status}: {err_text[:500]}"
                    )
                result = await resp.json()

        elapsed = time.monotonic() - t0

        # Parse response (OpenAI DALL-E compatible format)
        b64_json = result.get("data", [{}])[0].get("b64_json", "")
        if not b64_json:
            raise RuntimeError(
                f"Omni image edit response missing b64_json: {json.dumps(result)[:300]}"
            )

        image_bytes = base64.b64decode(b64_json)

        metadata = {
            "elapsed_s": round(elapsed, 2),
            "width": width,
            "height": height,
            "steps": steps,
            "size_bytes": len(image_bytes),
        }

        logger.info(
            "Image edit done: %.1fs, %d KB output",
            elapsed, len(image_bytes) // 1024,
        )

        return StepResult(
            data=image_bytes,
            media_type="image/png",
            metadata=metadata,
        )
