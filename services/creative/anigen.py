"""AniGen — 3D character generation from a single image (ForgeService).

Delegates to model_engine AniGenHandler for nn.Module decomposition
with explicit forward() calls. Full Euler ODE loop, CFG blending,
geodesic smooth noise — all in the orchestrator.
"""
from __future__ import annotations

import base64
import gc
import io
import logging
from pathlib import Path

import torch
from PIL import Image

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

VARIANT = "anigen"


class AniGenService(ForgeService):
    vram_mb = 12_000
    service_name = "anigen"
    default_model = "anigen"

    def __init__(self):
        super().__init__()
        self._modules = None
        self._orchestrator = None
        self._handle = None

    def load(self, model_name: str = "anigen") -> None:
        from services.model_engine.handlers.anigen.modules import AniGenModules
        from services.model_engine.handlers.anigen.orchestrator import AniGenOrchestrator

        model_path = self._resolve_model_path()

        self._modules = AniGenModules.load(model_path=model_path)
        self._orchestrator = AniGenOrchestrator(self._modules)

        self.model_name = model_name
        self._loaded = True

        # Warm-up: run once with a tiny input to fill CUDA graphs
        if torch.cuda.is_available():
            try:
                dummy = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
                buf = io.BytesIO()
                dummy.save(buf, format="PNG")
                self._handle = {"image": base64.b64encode(buf.getvalue()).decode(), "steps": 1}
                self._orchestrator(self._handle)
                torch.cuda.synchronize()
            except Exception:
                logger.warning("AniGen warm-up failed (non-critical)", exc_info=True)

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("AniGen loaded (VRAM=%.0fMB)", vram)

    def unload(self) -> None:
        self._modules = None
        self._orchestrator = None
        self._handle = None
        self.model_name = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def infer(self, payload: dict) -> dict:
        result = self._orchestrator(payload)
        return result

    def _resolve_model_path(self) -> Path:
        from registry.models import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.get_path("3d", "anigen")
        if not model_path.is_dir():
            raise FileNotFoundError(f"AniGen model not found at {model_path}")
        return model_path
