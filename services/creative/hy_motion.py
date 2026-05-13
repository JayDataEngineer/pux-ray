"""HY-Motion — Text-to-motion generation (ForgeService).

Delegates to model_engine HYMotionHandler for nn.Module decomposition
with explicit forward(). MotionCLIP encoder, MusicTransformer decoder,
3D pose regression — all in the orchestrator.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


class HYMotionService(ForgeService):
    vram_mb = 6_144
    service_name = "hy_motion"
    default_model = "hy-motion-1.0"

    def __init__(self):
        super().__init__()
        self._modules = None
        self._orchestrator = None

    def load(self, model_name: str = "hy-motion-1.0") -> None:
        from services.model_engine.handlers.hy_motion.modules import HYMotionModules
        from services.model_engine.handlers.hy_motion.orchestrator import HYMotionOrchestrator

        model_path = self._resolve_model_path()

        self._modules = HYMotionModules.load(
            model_path=model_path,
            model_type=model_name,
        )
        self._orchestrator = HYMotionOrchestrator(self._modules)

        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("HY-Motion loaded (VRAM=%.0fMB)", vram)

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
        model_path = registry.get_path("motion", "hy-motion-1.0")
        if not model_path.is_dir():
            raise FileNotFoundError(f"HY-Motion model not found at {model_path}")
        return model_path
