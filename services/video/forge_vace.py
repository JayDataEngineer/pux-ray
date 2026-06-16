"""Forge adapter for VACE video service.

Routes VACE video-editing requests to the standalone DiffSynth-Studio container
via HTTP. The VACE server runs in a separate Kubernetes pod, reachable via
vace-service:8082 (headless) or localhost:8082 (sidecar).

Why this exists separately from SGLang:
  SGLang's diffusion engine cannot ingest VCU (Video Condition Unit) tensors.
  VACE requires masked source-video latents + alpha mask channels + reference
  layout frames as extra input channels — incompatible with SGLang's compiled
  CUDA graph. DiffSynth-Studio handles VCU natively + ships TeaCache, tiled
  VAE, and AutoWrappedModule (mmGP-equivalent block streaming).

Claims full GPU in forge ledger for accurate eviction — VACE pipelines saturate
VRAM during the denoise loop and cannot coexist with other GPU services.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

# Try K8s headless service first, fall back to localhost (sidecar / dev)
VACE_URL = os.environ.get("VACE_URL", "http://vace-service:8082")

# Video generation is slow (sub-minute even with optimizations) — generous timeout.
# The forge caller already handles its own backpressure, so we just need to
# outlive the longest single-request denoise loop (~120s worst case on 4090).
REQUEST_TIMEOUT_S = 600


class VaceForgeService(ForgeService):
    """Calls the standalone VACE video server via HTTP."""

    service_name = "vace"
    default_model = "wan-vace-fun-a14b"
    persistence = Persistence.TRANSIENT
    vram_mb = 24576  # Full GPU — VACE cannot coexist with other GPU services

    def __init__(self):
        super().__init__()
        self._healthy = False
        self._vace_url: str | None = None
        self._current_model: str | None = None

    def _try_health(self) -> bool:
        """Check if VACE server is reachable. Caches the working URL."""
        for url in [VACE_URL, "http://localhost:8082", "http://127.0.0.1:8082"]:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(f"{url}/health")
                    if resp.status_code == 200:
                        self._vace_url = url
                        return True
            except Exception:
                continue
        return False

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Load a specific VACE model in the container."""
        model_name = model_name or self.default_model
        self._current_model = model_name

        if self._try_health():
            self._healthy = True
            logger.info("VACE: server reachable at %s", self._vace_url)
            # Pre-load the model (warm start — first /generate will skip the load)
            try:
                with httpx.Client(timeout=300) as client:
                    client.post(f"{self._vace_url}/load", json={"model": model_name})
                    logger.info("VACE: model '%s' pre-loaded", model_name)
            except Exception as e:
                logger.warning("VACE: pre-load failed (%s) — will lazy-load on first request", e)
        else:
            logger.warning("VACE: server not reachable — will retry on first request")

        self._loaded = True

    def unload(self) -> None:
        """Tell VACE server to release VRAM."""
        if self._vace_url:
            try:
                with httpx.Client(timeout=30) as client:
                    client.post(f"{self._vace_url}/release")
                logger.info("VACE: model released, GPU freed")
            except Exception as e:
                logger.warning("VACE: release failed (%s)", e)
        self._loaded = False
        self._healthy = False
        self._current_model = None

    def infer(self, payload: dict) -> dict:
        """Generate video via VACE HTTP API."""
        prompt = payload.get("prompt") or payload.get("input_prompt", "")
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        if not self._vace_url or not self._try_health():
            return {"status": "error", "error": "VACE server not reachable"}

        model = payload.get("model", self._current_model or self.default_model)
        body = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": payload.get("negative_prompt") or payload.get("n_prompt", ""),
            "vace_video": payload.get("vace_video") or payload.get("input_video"),
            "vace_video_mask": payload.get("vace_video_mask") or payload.get("input_mask"),
            "vace_reference_image": payload.get("vace_reference_image") or payload.get("input_image"),
            "vace_scale": float(payload.get("vace_scale", 1.0)),
            "width": int(payload.get("width", 832)),
            "height": int(payload.get("height", 480)),
            "num_frames": int(payload.get("num_frames", 81)),
            "seed": int(payload.get("seed", -1)),
            "steps": int(payload.get("steps", payload.get("sampling_steps", 0)) or 0),
            "cfg": float(payload.get("cfg", payload.get("guide_scale", 5.0))),
            "fps": int(payload.get("fps", 15)),
            "tea_cache_l1_thresh": payload.get("tea_cache_l1_thresh"),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
                resp = client.post(f"{self._vace_url}/generate", json=body)
        except httpx.ConnectError:
            return {"status": "error", "error": "VACE server not reachable"}
        except httpx.ReadTimeout:
            return {"status": "error", "error": f"VACE timed out after {REQUEST_TIMEOUT_S}s"}

        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return {
                "status": "error",
                "error": f"VACE returned {resp.status_code}: {resp.text[:300]}",
            }

        data = resp.json()
        self._current_model = data.get("model", model)
        metrics = data.get("metrics", {})
        metrics["forge_latency_s"] = round(elapsed, 2)

        return {
            "status": "success",
            "output": {
                "type": "video",
                "content": data.get("video", ""),
                "format": "mp4",
                "fps": data.get("fps", 15),
            },
            "metrics": metrics,
        }

    def actual_vram_mb(self) -> int:
        """Report full GPU allocation when loaded."""
        return self.vram_mb if self._loaded else 0
