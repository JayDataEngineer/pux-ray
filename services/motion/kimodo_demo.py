"""Kimodo Viser demo — Forge service wrapping the interactive 3D motion authoring UI.

Follows the ComfyUI subprocess pattern: start kimodo_demo as a subprocess,
proxy HTTP + WebSocket requests to it. Runs on port 18470 (matches Traefik ingress).

Uses _run_kimodo.py (in this same directory) to monkey-patch transformers +
huggingface_hub before kimodo loads, avoiding network calls that crash on
air-gapped pods (transformers' _patch_mistral_regex calls model_info()).
"""
from __future__ import annotations

import logging
from pathlib import Path

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.forge_subprocess import ForgeSubprocessMixin

logger = logging.getLogger(__name__)

_WRAPPER_SCRIPT = Path(__file__).resolve().parent / "_run_kimodo.py"


class KimodoDemoService(ForgeSubprocessMixin, ForgeService):
    vram_mb = 17_408
    service_name = "kimodo_demo"
    default_model = "kimodo-soma-rp"
    persistence = Persistence.PIPELINE_LOCKED

    PORT = 18470

    def load(self, model_name: str, quant: str | None = None) -> None:
        variant = model_name or self.default_model

        # Resolve local checkpoint dir from model registry.
        # Kimodo expects CHECKPOINT_DIR/<display_name>/ with config.yaml + weights.
        checkpoint_dir = "/models/avatar/kimodo"
        try:
            from registry.models import ModelRegistry
            reg = ModelRegistry()
            model_path = reg.get_path("motion", variant)
            parent = Path(model_path).parent
            if parent.is_dir():
                checkpoint_dir = str(parent)
        except Exception:
            pass

        if not _WRAPPER_SCRIPT.exists():
            raise FileNotFoundError(f"Kimodo wrapper not found: {_WRAPPER_SCRIPT}")

        # Resolve kimodo package directory for cwd.
        # K8s mounts vendor/ → /opt/vendor, so kimodo is at /opt/vendor/kimodo.
        # Local dev has it at <project>/vendor/kimodo.
        # Fallback to /opt/kimodo for legacy setups.
        _VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "kimodo"
        cwd = "/opt/vendor/kimodo"  # K8s mount path
        if _VENDOR_DIR.is_dir():
            cwd = str(_VENDOR_DIR)  # Local dev

        self.start_subprocess(
            cmd=[
                "python3", str(_WRAPPER_SCRIPT),
            ],
            port=self.PORT,
            health_path="/",
            timeout=600,
            cwd=cwd,
            env={
                "KIMODO_MODEL": variant,
                "SERVER_PORT": str(self.PORT),
                "SERVER_NAME": "0.0.0.0",
                "CHECKPOINT_DIR": checkpoint_dir,
                "TEXT_ENCODER_MODE": "local",
                "HF_HOME": "/models/.hf_cache",
                "HF_HUB_CACHE": "/models/cache/huggingface",
                "TRANSFORMERS_CACHE": "/models/cache/huggingface",
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
                if payload.get("raw"):
                    kwargs: dict = {
                        "method": payload.get("method", "GET"),
                        "path": payload["path"],
                        "params": payload.get("params"),
                        "timeout": payload.get("timeout", 600),
                    }
                    # Forward binary POST bodies (screenshots, exports)
                    if payload.get("body_b64"):
                        import base64
                        kwargs["content"] = base64.b64decode(payload["body_b64"])
                        if payload.get("content_type"):
                            kwargs["headers"] = {"content-type": payload["content_type"]}
                    return self._call_raw_full(**kwargs)
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
