"""Kimodo Viser demo — Forge service wrapping the interactive 3D motion authoring UI.

Follows the ComfyUI subprocess pattern: start kimodo_demo as a subprocess,
proxy HTTP + WebSocket requests to it. Runs on port 18470 (matches Traefik ingress).
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

        # Resolve local checkpoint dir from model registry.
        # Kimodo expects CHECKPOINT_DIR/<display_name>/ with config.yaml + weights.
        # The model registry has the full path (e.g. /models/avatar/kimodo/Kimodo-SOMA-RP-v1.1/);
        # CHECKPOINT_DIR is the parent (e.g. /models/avatar/kimodo/).
        checkpoint_dir = "/models/avatar/kimodo"
        try:
            from registry.models import ModelRegistry
            reg = ModelRegistry()
            model_path = reg.get_path("avatar", variant)
            from pathlib import Path
            parent = Path(model_path).parent
            if parent.is_dir():
                checkpoint_dir = str(parent)
        except Exception:
            pass

        self.start_subprocess(
            cmd=[
                "kimodo_demo",
                "--model", variant,
            ],
            port=self.PORT,
            health_path="/",
            timeout=600,
            cwd="/opt/kimodo",
            env={
                "SERVER_PORT": str(self.PORT),
                "SERVER_NAME": "0.0.0.0",
                "CHECKPOINT_DIR": checkpoint_dir,
                "TEXT_ENCODER_MODE": "local",
                # Point HuggingFace cache at PVC so LLM2Vec finds the
                # downloaded Llama + adapter models without internet access.
                # The hub cache (models--Org--Model/) is populated by the
                # download script so from_pretrained() finds files locally.
                "HF_HOME": "/models/.hf_cache",
                "HF_HUB_CACHE": "/models/cache/huggingface",
                "TRANSFORMERS_CACHE": "/models/cache/huggingface",
                # The diffusion model uses CHECKPOINT_DIR (local path).
                # LLM2Vec uses the HF hub cache populated by the download script.
                # Do NOT set HF_HUB_OFFLINE=1 — the hub cache may need to
                # verify refs/blobs which requires a brief metadata check.
                # With the hub cache populated, no large downloads happen.
            },
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
