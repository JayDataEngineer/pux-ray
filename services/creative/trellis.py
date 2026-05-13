"""TRELLIS — Image-to-3D generation (ForgeService).

Delegates to model_engine TrellisHandler for nn.Module decomposition
with explicit forward(). SparseTensor denoising, SSD texture generation,
marching cubes meshing — all in the orchestrator.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


class TrellisService(ForgeService):
    vram_mb = 10_240
    service_name = "trellis"
    default_model = "trellis"

    def __init__(self):
        super().__init__()
        self._modules = None
        self._orchestrator = None

    def load(self, model_name: str = "trellis") -> None:
        from services.model_engine.handlers.trellis.modules import TrellisModules
        from services.model_engine.handlers.trellis.orchestrator import TrellisOrchestrator

        model_path = self._resolve_model_path()

        self._modules = TrellisModules.load(model_path=model_path)
        self._orchestrator = TrellisOrchestrator(self._modules)

        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("TRELLIS loaded (VRAM=%.0fMB)", vram)

    def unload(self) -> None:
        self._modules = None
        self._orchestrator = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def infer(self, payload: dict) -> dict:
        return self._orchestrator(payload)

    def _resolve_model_path(self) -> Path:
        from registry.models import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.get_path("3d", "trellis")
        if not model_path.is_dir():
            raise FileNotFoundError(f"TRELLIS model not found at {model_path}")
        return model_path
