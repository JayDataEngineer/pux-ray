"""Kimodo Viser demo — Forge service wrapping the interactive 3D motion authoring UI.

Follows the ComfyUI subprocess pattern: start kimodo_demo as a subprocess,
proxy HTTP + WebSocket requests to it. Runs on port 18470.
"""
from __future__ import annotations

import logging

from services.forge_base import ForgeService
from services.forge_subprocess import ForgeSubprocessMixin

logger = logging.getLogger(__name__)


class KimodoDemoService(ForgeSubprocessMixin, ForgeService):
    vram_mb = 17_408
    service_name = "kimodo_demo"
    default_model = "kimodo-soma-rp"

    PORT = 18470

    def load(self, model_name: str, quant: str | None = None) -> None:
        variant = model_name or self.default_model
        self.start_subprocess(
            cmd=[
                "python3", "-m", "kimodo.scripts.demo",
                "--model", variant,
                "--port", str(self.PORT),
                "--host", "0.0.0.0",
            ],
            port=self.PORT,
            health_path="/",
            timeout=600,
            cwd="/opt/kimodo",
        )
        self.model_name = model_name
        self._loaded = True

    def unload(self) -> None:
        self.stop_subprocess()
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        """Proxy requests to the Viser demo's HTTP API."""
        if "path" in payload:
            try:
                result = self._call(
                    method=payload.get("method", "GET"),
                    path=payload["path"],
                    json=payload.get("body"),
                    params=payload.get("params"),
                    timeout=payload.get("timeout", 600),
                )
                return {"status": "ok", "result": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        return {"status": "error", "error": "payload must contain 'path'"}
