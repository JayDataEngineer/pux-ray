"""Forge adapter for MOSS audio service.

Routes audio requests to the standalone MOSS container via HTTP.
The MOSS server runs in a separate Kubernetes pod, reachable via
moss-service:8081 (headless service) or localhost:8081 (sidecar).

Supports model switching: moss-soundeffect-v2, moss-tts, etc.
Claims full GPU in forge ledger for accurate eviction.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

# Try K8s service first, fall back to localhost (sidecar)
MOSS_URL = os.environ.get("MOSS_URL", "http://moss-service:8081")


class MossForgeService(ForgeService):
    """Calls the standalone MOSS audio server via HTTP."""

    service_name = "moss"
    default_model = "moss-soundeffect-v2"
    persistence = Persistence.TRANSIENT
    vram_mb = 24576  # Full GPU — MOSS gets exclusive access

    def __init__(self):
        super().__init__()
        self._healthy = False
        self._current_model: str | None = None

    def _try_health(self) -> bool:
        """Check if MOSS server is reachable."""
        for url in [MOSS_URL, "http://localhost:8081", "http://127.0.0.1:8081"]:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(f"{url}/health")
                    if resp.status_code == 200:
                        self._moss_url = url
                        return True
            except Exception:
                continue
        return False

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Load a specific MOSS model."""
        model_name = model_name or self.default_model
        self._current_model = model_name

        if self._try_health():
            self._healthy = True
            logger.info("MOSS: server reachable at %s", getattr(self, "_moss_url", MOSS_URL))
            # Pre-load the model
            try:
                with httpx.Client(timeout=120) as client:
                    client.post(f"{self._moss_url}/load", json={"model": model_name})
                    logger.info("MOSS: model '%s' pre-loaded", model_name)
            except Exception:
                pass  # Model will load on first generate
        else:
            logger.warning("MOSS: server not reachable — will try on first request")

        self._loaded = True

    def unload(self) -> None:
        """Tell MOSS server to release VRAM."""
        url = getattr(self, "_moss_url", MOSS_URL)
        try:
            with httpx.Client(timeout=30) as client:
                client.post(f"{url}/release")
            logger.info("MOSS: model released, GPU freed")
        except Exception:
            pass
        self._loaded = False
        self._current_model = None

    def infer(self, payload: dict) -> dict:
        """Generate audio via MOSS HTTP API."""
        prompt = payload.get("prompt") or payload.get("input_prompt", "")
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        # Find reachable server
        if not getattr(self, "_moss_url", None) or not self._try_health():
            return {"status": "error", "error": "MOSS server not reachable"}

        model = payload.get("model", self._current_model or self.default_model)
        body = {
            "model": model,
            "prompt": prompt,
            "seconds": float(payload.get("seconds", payload.get("duration_seconds", 3.0))),
            "seed": int(payload.get("seed", 0)),
            "steps": int(payload.get("steps", payload.get("sampling_steps", 50))),
            "cfg": float(payload.get("cfg", payload.get("guide_scale", 4.0))),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=300) as client:
                resp = client.post(f"{self._moss_url}/generate", json=body)
        except httpx.ConnectError:
            return {"status": "error", "error": f"MOSS server not reachable"}

        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return {"status": "error", "error": f"MOSS returned {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
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
                "gen_time_s": data.get("generation_time_s"),
            },
        }

    def actual_vram_mb(self) -> int:
        """Report full GPU allocation when loaded."""
        return self.vram_mb if self._loaded else 0
