"""Model executor — loads models into a shared mmgp pool, runs inference.

One executor per GPU. Multiple model families coexist. mmgp handles all
VRAM/RAM placement automatically — no manual budgets or eviction needed.
"""
from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from mmgp import offload

from services.model_engine.base_handler import BaseHandler, LoadResult

logger = logging.getLogger(__name__)


class ModelExecutor:
    """Loads models into a shared mmgp pool and runs inference."""

    def __init__(self, models_root: Path, mmgp_profile: int = 1):
        self._models_root = models_root
        self._mmgp_profile = mmgp_profile
        self._handlers: dict[str, BaseHandler] = {}
        self._loaded: OrderedDict[str, LoadResult] = OrderedDict()
        self._pool: dict[str, torch.nn.Module] = {}
        self._co_tenants: dict[str, list[str]] = {}

    def register_handler(self, family: str, handler: BaseHandler) -> None:
        self._handlers[family] = handler

    def _resolve_handler(self, model_type: str) -> tuple[BaseHandler, str]:
        for family, handler in self._handlers.items():
            if model_type in handler.supported_types():
                return handler, model_type
        raise ValueError(
            f"No handler for '{model_type}'. "
            f"Registered: {list(self._handlers.keys())}"
        )

    def _resolve_model_path(self, model_type: str) -> Path:
        for family, handler in self._handlers.items():
            if model_type in handler.supported_types():
                return handler.resolve_path(model_type, self._models_root)
        raise FileNotFoundError(f"No model weights found for {model_type}")

    def _family_for(self, model_type: str) -> str:
        for family, handler in self._handlers.items():
            if model_type in handler.supported_types():
                return family
        return "unknown"

    def _reprofile(self) -> None:
        if not self._pool:
            return
        offload.profile(
            self._pool,
            profile_no=self._mmgp_profile,
            coTenantsMap=self._co_tenants,
        )

    def load(self, model_type: str, **kwargs) -> LoadResult:
        if model_type in self._loaded:
            self._loaded.move_to_end(model_type)
            return self._loaded[model_type]

        handler, _ = self._resolve_handler(model_type)
        model_path = self._resolve_model_path(model_type)
        family = self._family_for(model_type)

        logger.info("Loading %s from %s", model_type, model_path)
        result = handler.load_model(model_type, model_path, **kwargs)

        # Merge pipe into shared pool with family-prefixed keys
        for name, module in result.pipe.items():
            key = f"{family}/{name}"
            self._pool[key] = module
            if name in result.co_tenants:
                self._co_tenants[key] = [f"{family}/{t}" for t in result.co_tenants[name]]

        self._reprofile()
        self._loaded[model_type] = result

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (%d pool modules, VRAM=%.0fMB)", model_type, len(self._pool), vram)
        return result

    def unload(self, model_type: str | None = None) -> None:
        if model_type:
            if model_type not in self._loaded:
                return
            del self._loaded[model_type]
            # Remove that model's modules from pool
            # (requires tracking — rebuild pool from remaining loaded)
            family = self._family_for(model_type)
            # Can't easily remove by family prefix without tracking keys
            # so rebuild the pool from scratch
            self._rebuild_pool()
        else:
            self._loaded.clear()
            self._pool.clear()
            self._co_tenants.clear()

        if self._pool:
            self._reprofile()

        offload.flush_torch_caches()
        gc.collect()
        torch.cuda.empty_cache()

    def _rebuild_pool(self) -> None:
        """Rebuild pool from currently loaded models (after removal)."""
        self._pool.clear()
        self._co_tenants.clear()
        for model_type, result in self._loaded.items():
            family = self._family_for(model_type)
            for name, module in result.pipe.items():
                key = f"{family}/{name}"
                self._pool[key] = module
                if name in result.co_tenants:
                    self._co_tenants[key] = [f"{family}/{t}" for t in result.co_tenants[name]]

    def infer(self, model_type: str, payload: dict) -> Any:
        if model_type not in self._loaded:
            self.load(model_type)
        self._loaded.move_to_end(model_type)
        return self._loaded[model_type].pipeline(payload)

    def status(self) -> dict:
        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        return {
            "loaded_models": list(self._loaded.keys()),
            "pool_modules": len(self._pool),
            "active_vram_mb": vram,
            "mmgp_profile": self._mmgp_profile,
            "registered_families": list(self._handlers.keys()),
        }
