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

    # ── Patch 2: Resolve repo IDs to local cached snapshot paths ──
    # from_pretrained("org/model") triggers network checks in transformers.
    # If the model is cached locally, resolve to the snapshot path so
    # transformers sees it as a local directory and skips network calls.
    from huggingface_hub import scan_cache_dir
    _cache_dir = os.environ.get("HF_HUB_CACHE", "")
    _repo_to_local = {}
    if _cache_dir:
        try:
            for _repo in scan_cache_dir(_cache_dir).repos:
                for _rev in _repo.revisions:
                    _repo_to_local[_repo.repo_id] = str(_rev.snapshot_path)
                    break
        except Exception:
            pass

    if _repo_to_local:
        import transformers
        _orig_auto_from = transformers.AutoTokenizer.from_pretrained
        @classmethod
        def _patched_auto_from(cls, name_or_path, *a, **kw):
            name_or_path = _repo_to_local.get(name_or_path, name_or_path)
            return _orig_auto_from(name_or_path, *a, **kw)
        transformers.AutoTokenizer.from_pretrained = _patched_auto_from

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
