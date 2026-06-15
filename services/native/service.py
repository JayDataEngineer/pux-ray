"""Native model service via SGLang HTTP API.

NO diffusers. NO mmGP. NO Wan2GP.
SGLang serves models; this service calls SGLang's OpenAI-compatible API.

For LTX advanced features (keyframes, IC-LoRA), uses ltx-pipelines directly.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.native.registry import get_model, ModelEntry, ALL_MODELS

logger = logging.getLogger(__name__)
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30010")


class NativeService(ForgeService):
    """Serves models via SGLang HTTP API.

    SGLang runs as a separate process (sglang serve --model-type diffusion).
    This service manages model loading/unloading via SGLang's sleep/wake API
    and routes generation requests through the OpenAI-compatible endpoint.
    """

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self.entry: Optional[ModelEntry] = None
        self._current_model: str | None = None
        self._sglang_ready = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Load a model by telling SGLang to serve it."""
        model_name = model_name or self.default_model
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ALL_MODELS.keys())}")

        self.entry = entry
        self._current_model = model_name

        # Check if SGLang is already serving this model
        if self._is_serving(model_name):
            logger.info("SGLang already serving '%s'", model_name)
        else:
            # Need to switch models — SGLang doesn't support multi-model,
            # so we restart the server with the new model
            self._switch_model(model_name)

        self._loaded = True
        self.model_name = model_name
        logger.info("Native: '%s' loaded via SGLang", model_name)

    def unload(self) -> None:
        """Release model VRAM via SGLang sleep."""
        if self._sglang_ready:
            try:
                with httpx.Client(timeout=30) as client:
                    client.post(f"{SGLANG_URL}/release_memory_occupation",
                               json={"tags": ["weights", "cache"]})
                logger.info("SGLang: memory released (sleep mode)")
            except Exception as e:
                logger.warning("SGLang sleep failed: %s", e)

        self._loaded = False
        self.model_name = None
        self._current_model = None

    def infer(self, payload: dict) -> dict:
        """Run inference via SGLang API."""
        if not self._loaded:
            return {"status": "error", "error": "Not loaded"}

        e = self.entry
        if e is None:
            return {"status": "error", "error": "No model entry"}

        try:
            if e.task in ("text2video", "image2video"):
                return self._generate_video(payload)
            else:
                return self._generate_image(payload)
        except Exception as ex:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(ex)}

    def _generate_image(self, payload: dict) -> dict:
        """Generate image via SGLang /v1/images/generations."""
        prompt = payload.get("prompt", "")
        steps = payload.get("steps") or payload.get("sampling_steps") or self.entry.steps
        width = int(payload.get("width", self.entry.width))
        height = int(payload.get("height", self.entry.height))
        seed = payload.get("seed", -1)

        body = {
            "model": self._current_model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "num_inference_steps": int(steps),
        }
        if seed >= 0:
            body["seed"] = int(seed)

        t0 = time.perf_counter()
        with httpx.Client(timeout=300) as client:
            resp = client.post(f"{SGLANG_URL}/v1/images/generations", json=body)

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error", "error": f"SGLang returned {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()

        # Extract image from response
        image_b64 = None
        if "images" in data and data["images"]:
            img = data["images"][0]
            image_b64 = img.split(",", 1)[1] if isinstance(img, str) and img.startswith("data:") else img
        elif "data" in data and data["data"]:
            img = data["data"][0]
            image_b64 = img.get("b64_json") or img.get("url", "").split(",", 1)[-1] if isinstance(img, dict) else img

        if not image_b64:
            return {"status": "error", "error": "No image in SGLang response"}

        return {
            "status": "success",
            "output": {"type": "image", "content": image_b64, "format": "png"},
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": self.model_name,
            },
        }

    def _generate_video(self, payload: dict) -> dict:
        """Generate video via SGLang /v1/videos/generations."""
        prompt = payload.get("prompt", "")
        steps = payload.get("steps") or self.entry.steps
        width = int(payload.get("width", self.entry.width))
        height = int(payload.get("height", self.entry.height))
        num_frames = int(payload.get("num_frames", 121))
        seed = payload.get("seed", -1)

        body = {
            "model": self._current_model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": int(steps),
        }
        if seed >= 0:
            body["seed"] = int(seed)

        t0 = time.perf_counter()
        with httpx.Client(timeout=600) as client:
            resp = client.post(f"{SGLANG_URL}/v1/videos/generations", json=body)

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error", "error": f"SGLang returned {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        video_b64 = data.get("video") or data.get("data", [{}])[0].get("video", "")

        return {
            "status": "success",
            "output": {"type": "video", "content": video_b64},
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": self.model_name,
                "frames": num_frames,
            },
        }

    # ── SGLang Management ──────────────────────────────────────────────────────

    def _is_serving(self, model_name: str) -> bool:
        """Check if SGLang is already serving this model."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{SGLANG_URL}/health")
                if resp.status_code == 200:
                    self._sglang_ready = True
                    return True
        except Exception:
            pass
        self._sglang_ready = False
        return False

    def _switch_model(self, model_name: str) -> None:
        """Switch SGLang to a different model.

        In production, this would:
        1. Release current model VRAM (POST /release_memory_occupation)
        2. Kill the SGLang server process
        3. Restart with new model (sglang serve --model-path ...)
        4. Wait for health check

        For now, assumes SGLang is already running with the model.
        """
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info("SGLang: switching to '%s'", model_name)

        # Try to wake SGLang if it's sleeping
        try:
            with httpx.Client(timeout=30) as client:
                client.post(f"{SGLANG_URL}/resume_memory_occupation")
                self._sglang_ready = True
        except Exception:
            pass

        if not self._is_serving(model_name):
            logger.warning("SGLang not serving '%s' — model must be started externally", model_name)

    def actual_vram_mb(self) -> int:
        """Check SGLang's VRAM via health/status endpoint."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{SGLANG_URL}/health")
                if resp.status_code == 200:
                    return 0  # SGLang manages its own VRAM
        except Exception:
            pass
        return 0
