"""Batch executor — loads models, runs task lists, manages mmgp lifecycle.

The executor owns the GPU. It loads models via handlers, registers pipe dicts
with mmgp, and runs task lists sequentially with models persisting in memory.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import torch
from mmgp import offload

from services.model_engine.base_handler import BaseHandler, LoadResult

logger = logging.getLogger(__name__)


class ModelExecutor:
    """Manages model lifecycle and batch task execution.

    One executor per GPU. Models stay loaded between tasks via mmgp.
    """

    def __init__(self, models_root: Path, mmgp_profile: int = 1):
        self._models_root = models_root
        self._mmgp_profile = mmgp_profile
        self._loaded: dict[str, LoadResult] = {}
        self._handlers: dict[str, BaseHandler] = {}

    def register_handler(self, family: str, handler: BaseHandler) -> None:
        """Register a handler for a model family."""
        self._handlers[family] = handler
        logger.info("Registered handler for %s: %s", family, handler.__class__.__name__)

    def _resolve_handler(self, model_type: str) -> tuple[BaseHandler, str]:
        """Find the handler and model_type for a given type string."""
        for family, handler in self._handlers.items():
            if model_type in handler.supported_types():
                return handler, model_type
        raise ValueError(
            f"No handler for model type '{model_type}'. "
            f"Registered families: {list(self._handlers.keys())}"
        )

    def _resolve_model_path(self, model_type: str) -> Path:
        """Resolve model weights directory from model type."""
        # TODO: integrate with registry.models.ModelRegistry
        # For now, convention: models_root/<model_type>/
        path = self._models_root / model_type
        if not path.is_dir():
            # Try family-level directory
            for family, handler in self._handlers.items():
                if model_type in handler.supported_types():
                    path = self._models_root / family
                    if path.is_dir():
                        return path
            raise FileNotFoundError(f"No model weights found for {model_type}")
        return path

    def load(self, model_type: str, **kwargs) -> LoadResult:
        """Load a model and register with mmgp.

        If already loaded, returns cached result.
        """
        if model_type in self._loaded:
            return self._loaded[model_type]

        handler, resolved_type = self._resolve_handler(model_type)
        model_path = self._resolve_model_path(resolved_type)

        logger.info("Loading %s from %s (mmgp profile=%d)",
                     model_type, model_path, self._mmgp_profile)

        result = handler.load_model(resolved_type, model_path, **kwargs)

        # Register pipe dict with mmgp
        offload.profile(
            result.pipe,
            profile_no=self._mmgp_profile,
            coTenantsMap=result.co_tenants,
        )

        self._loaded[model_type] = result

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (VRAM=%.0fMB, active models=%d)",
                     model_type, vram, len(self._loaded))

        return result

    def unload(self, model_type: str | None = None) -> None:
        """Unload a specific model or all models."""
        if model_type:
            if model_type in self._loaded:
                del self._loaded[model_type]
        else:
            self._loaded.clear()

        offload.flush_torch_caches()
        gc.collect()
        torch.cuda.empty_cache()

    def infer(self, model_type: str, payload: dict) -> Any:
        """Run inference on a loaded model."""
        if model_type not in self._loaded:
            self.load(model_type)

        result = self._loaded[model_type]
        return result.pipeline(payload)

    def status(self) -> dict:
        """Current state of the executor."""
        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        return {
            "loaded_models": list(self._loaded.keys()),
            "active_vram_mb": vram,
            "mmgp_profile": self._mmgp_profile,
            "registered_families": list(self._handlers.keys()),
        }
