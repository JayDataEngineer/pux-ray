"""Kimodo Viser demo — Forge service wrapping the interactive 3D motion authoring UI.

Follows the ComfyUI subprocess pattern: start kimodo_demo as a subprocess,
proxy HTTP + WebSocket requests to it. Runs on port 18470 (matches Traefik ingress).

Uses a wrapper script that patches transformers + huggingface_hub to avoid
network calls during model loading (tokenizer's is_base_mistral check, etc.).
"""
from __future__ import annotations

import logging
import textwrap

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.forge_subprocess import ForgeSubprocessMixin

logger = logging.getLogger(__name__)

# Wrapper script that monkey-patches transformers before kimodo loads.
# Fixes: transformers' _patch_mistral_regex calls model_info() which hits
# the network and fails on air-gapped pods. Patch it to skip the check.
_WRAPPER_SCRIPT = textwrap.dedent("""\
    import os, sys

    # ── Patch 1: Kill all network calls from huggingface_hub ──
    # transformers' _patch_mistral_regex calls model_info() which hits the
    # network. On air-gapped pods this triggers OfflineModeIsEnabled.
    # Replace model_info at the module level so ALL imports see the patch.
    class _FakeModelInfo:
        tags = []
        library_name = None
        def __init__(self, *a, **kw): pass

    import huggingface_hub
    import huggingface_hub.hf_api as _hfapi
    _hfapi.model_info = lambda *a, **kw: _FakeModelInfo()
    huggingface_hub.model_info = _hfapi.model_info

    # ── Patch 2: Set offline mode so transformers uses local cache ──
    # With HF_HUB_CACHE set and model_info patched to be a no-op,
    # transformers should resolve cached models automatically.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ── Start kimodo ──
    import kimodo.demo
    model = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
    sys.argv = ["kimodo_demo", "--model", model]
    kimodo.demo.main()
""")


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
            from pathlib import Path
            parent = Path(model_path).parent
            if parent.is_dir():
                checkpoint_dir = str(parent)
        except Exception:
            pass

        # Write wrapper script to /tmp (same machine as the subprocess)
        wrapper_path = "/tmp/kimodo_wrapper.py"
        try:
            with open(wrapper_path, "w") as f:
                f.write(_WRAPPER_SCRIPT)
            logger.info("Wrote kimodo wrapper to %s", wrapper_path)
        except Exception as e:
            logger.error("Failed to write wrapper: %s", e)
            raise

        self.start_subprocess(
            cmd=[
                "python3", wrapper_path,
            ],
            port=self.PORT,
            health_path="/",
            timeout=600,
            cwd="/opt/kimodo",
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
                    return self._call_raw_full(
                        method=payload.get("method", "GET"),
                        path=payload["path"],
                        params=payload.get("params"),
                        timeout=payload.get("timeout", 600),
                    )
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
