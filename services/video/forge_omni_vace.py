"""Forge adapter for Omni VACE — Wan2.2 via vLLM-Omni.

Calls the Omni FastAPI server via HTTP, same pattern as forge_moss.py.
Claims full GPU in forge ledger. Supports "base" and "turbo" profiles.
"""
from __future__ import annotations
import logging, os, time
import httpx
from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)
OMNI_URL = os.environ.get("OMNI_URL", "http://omni-vace-service:8083")
TIMEOUT_S = 600

class OmniVaceForgeService(ForgeService):
    service_name = "omni-vace"
    default_model = "wan-vace-fun-a14b"
    persistence = Persistence.TRANSIENT
    vram_mb = 24576

    def __init__(self):
        super().__init__()
        self._omni_url: str | None = None

    def _try_health(self) -> bool:
        for url in [OMNI_URL, "http://localhost:8083", "http://127.0.0.1:8083"]:
            try:
                with httpx.Client(timeout=5) as c:
                    if c.get(f"{url}/health").status_code == 200:
                        self._omni_url = url
                        return True
            except: continue
        return False

    def load(self, model_name=None, quant=None):
        if self._try_health() and self._omni_url:
            try:
                with httpx.Client(timeout=120) as c:
                    c.post(f"{self._omni_url}/load")
                logger.info("Omni VACE: model pre-loaded")
            except: pass
        self._loaded = True

    def unload(self):
        if self._omni_url:
            try:
                with httpx.Client(timeout=30) as c:
                    c.post(f"{self._omni_url}/release")
            except: pass
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        prompt = payload.get("prompt") or payload.get("input_prompt", "")
        if not prompt:
            return {"status": "error", "error": "no prompt"}
        if not self._omni_url or not self._try_health():
            return {"status": "error", "error": "server unreachable"}

        body = {
            "prompt": prompt,
            "negative_prompt": payload.get("negative_prompt", ""),
            "width": int(payload.get("width", 832)),
            "height": int(payload.get("height", 480)),
            "num_frames": int(payload.get("num_frames", 81)),
            "steps": int(payload.get("steps", payload.get("sampling_steps", 0)) or 18),
            "cfg": float(payload.get("cfg", payload.get("guide_scale", 5.0))),
            "seed": int(payload.get("seed", -1)),
            "fps": int(payload.get("fps", 16)),
            "profile": payload.get("profile", "base"),
            "reference_image": (payload.get("reference_image") or
                                payload.get("vace_reference_image")),
            "source_video": (payload.get("source_video") or
                             payload.get("vace_video")),
            "source_mask": (payload.get("source_mask") or
                            payload.get("vace_video_mask")),
            "last_image": payload.get("last_image"),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=TIMEOUT_S) as c:
                resp = c.post(f"{self._omni_url}/generate", json=body)
        except httpx.ConnectError:
            return {"status": "error", "error": "server unreachable"}
        except httpx.ReadTimeout:
            return {"status": "error", "error": f"timeout after {TIMEOUT_S}s"}
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        metrics = data.get("metrics", {})
        metrics["forge_latency_s"] = round(elapsed, 2)
        return {
            "status": "success",
            "output": data.get("output", {}),
            "metrics": metrics,
        }

    def actual_vram_mb(self) -> int:
        return self.vram_mb if self._loaded else 0
