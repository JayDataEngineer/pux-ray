"""MOSS-SoundEffect — Text-to-sound effect generation (ForgeService).

Delegates to model_engine MossHandler for nn.Module decomposition
with explicit generate loop. Delay pattern sampling embedded in
model.generate() — sub-modules go into mmgp pipe dict for VRAM management.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


class MossService(ForgeService):
    vram_mb = 16_000
    service_name = "moss_soundeffect"
    default_model = "moss"

    def __init__(self):
        super().__init__()
        self._modules = None
        self._orchestrator = None

    def load(self, model_name: str = "moss") -> None:
        from services.model_engine.handlers.moss.modules import MossModules
        from services.model_engine.handlers.moss.orchestrator import MossOrchestrator

        model_path = self._resolve_model_path()

        self._modules = MossModules.load(model_path=model_path)
        self._orchestrator = MossOrchestrator(self._modules)

        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("MOSS loaded (VRAM=%.0fMB)", vram)

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
        model_path = registry.get_path("audio", "moss-soundeffect")
        if not model_path.is_dir():
            raise FileNotFoundError(f"MOSS model not found at {model_path}")
        return model_path
