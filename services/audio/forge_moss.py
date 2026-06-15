"""Forge adapter for MOSS audio service.

Routes audio requests to the standalone MOSS container via HTTP.
The MOSS server runs in a separate Docker container on port 8081.
"""
from __future__ import annotations

import logging
import os

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

MOSS_URL = os.environ.get("MOSS_URL", "http://localhost:8081")


class MossForgeService(ForgeService):
    """Calls the standalone MOSS audio server via HTTP."""

    service_name = "moss"
    default_model = "moss-soundeffect"
    persistence = Persistence.TRANSIENT
    vram_mb = 0  # Self-managed in separate container

    def __init__(self):
        super().__init__()
        self._healthy = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Check MOSS server health — model loads on first request."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{MOSS_URL}/health")
                self._healthy = resp.status_code == 200
                self._loaded = True
                logger.info("MOSS: server healthy at %s", MOSS_URL)
        except Exception as e:
            logger.warning("MOSS: server not reachable at %s: %s", MOSS_URL, e)
            self._loaded = True  # Allow requests — server might start later

    def unload(self) -> None:
        """Tell MOSS server to release VRAM."""
        try:
            with httpx.Client(timeout=30) as client:
                client.post(f"{MOSS_URL}/release")
        except Exception:
            pass
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        """Generate audio via MOSS HTTP API."""
        import time

        prompt = payload.get("prompt") or payload.get("input_prompt", "")
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        body = {
            "prompt": prompt,
            "seconds": float(payload.get("seconds", 10.0)),
            "seed": int(payload.get("seed", 0)),
            "steps": int(payload.get("steps", 100)),
            "cfg": float(payload.get("cfg", 4.0)),
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
                "model": "moss-soundeffect",
                "duration_s": data.get("duration_s"),
            },
        }
