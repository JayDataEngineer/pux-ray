"""Forge adapter for MOSS audio service.

Routes audio requests to the standalone MOSS container via HTTP.
The MOSS server runs in a separate Docker container on port 8081.
Supports multiple MOSS model variants with automatic switching.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

MOSS_URL = os.environ.get("MOSS_URL", "http://localhost:8081")


class MossForgeService(ForgeService):
    """Calls the standalone MOSS audio server via HTTP.

    Supports model switching: moss-soundeffect, moss-tts, etc.
    Each request specifies which model to use. The MOSS server
    handles unload/load/switch internally.
    """

    service_name = "moss"
    default_model = "moss-soundeffect"
    persistence = Persistence.TRANSIENT
    # Claim full GPU — MOSS runs in a separate container with exclusive GPU access.
    vram_mb = 24576  # Full RTX 4090 VRAM

    def __init__(self):
        super().__init__()
        self._healthy = False
        self._current_model: str | None = None

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Load a specific MOSS model. Switches if different model is loaded."""
        model_name = model_name or self.default_model

        try:
            # Tell MOSS server to load this model (pre-load for faster first request)
            with httpx.Client(timeout=120) as client:
                resp = client.post(f"{MOSS_URL}/load", json={"model": model_name})
                if resp.status_code == 200:
                    self._healthy = True
                    self._current_model = model_name
                    self._loaded = True
                    logger.info("MOSS: model '%s' loaded via server", model_name)
                else:
                    # Server might not support /load yet — just check health
                    resp = client.get(f"{MOSS_URL}/health")
                    self._healthy = resp.status_code == 200
                    self._loaded = True
        except Exception as e:
            logger.warning("MOSS: server not reachable at %s: %s", MOSS_URL, e)
            # Mark as loaded anyway — first generate call will trigger load
            self._loaded = True
            self._current_model = model_name

    def unload(self) -> None:
        """Tell MOSS server to release VRAM and free the GPU."""
        try:
            with httpx.Client(timeout=30) as client:
                client.post(f"{MOSS_URL}/release")
            logger.info("MOSS: model released, GPU freed")
        except Exception:
            pass
        self._loaded = False
        self._current_model = None

    def infer(self, payload: dict) -> dict:
        """Generate audio via MOSS HTTP API.

        Passes the model name so the server can switch if needed.
        """
        prompt = payload.get("prompt") or payload.get("input_prompt", "")
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        model = payload.get("model", self._current_model or self.default_model)

        body = {
            "model": model,
            "prompt": prompt,
            "seconds": float(payload.get("seconds", payload.get("duration_seconds", 10.0))),
            "seed": int(payload.get("seed", 0)),
            "steps": int(payload.get("steps", payload.get("sampling_steps", 100))),
            "cfg": float(payload.get("cfg", payload.get("guide_scale", 4.0))),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=300) as client:
                resp = client.post(f"{MOSS_URL}/generate", json=body)
        except httpx.ConnectError:
            return {"status": "error", "error": f"MOSS server not reachable at {MOSS_URL}"}

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error", "error": f"MOSS returned {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        # Track which model is actually loaded
        self._current_model = data.get("model", model)

        return {
            "status": "success",
            "output": {
                "type": "audio",
                "content": data.get("audio", ""),
                "format": "wav",
                "sample_rate": data.get("sample_rate", 48000),
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": model,
                "duration_s": data.get("duration_s"),
            },
        }

    def actual_vram_mb(self) -> int:
        """Report full GPU allocation when loaded."""
        return self.vram_mb if self._loaded else 0
