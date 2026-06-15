"""ForgeProxy — routes workflow calls through Forge's VRAM ledger.

Workflow functions call:
    svc = get_service()  # bare NativeService singleton or ForgeProxy
    svc.load("z-image-turbo")
    svc.infer({...})

With ForgeProxy they call the same methods, but VRAM is tracked by the Forge.
First load goes through Forge's full lifecycle (_do_load).
Model swaps call the adapter directly and reconcile VRAM with the Forge ledger.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.forge import ForgeCore

logger = logging.getLogger(__name__)


class ForgeProxy:
    """Drop-in replacement for NativeService with Forge VRAM tracking."""

    def __init__(self, forge_core: ForgeCore):
        self._forge = forge_core
        self._native_loaded = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        if not self._native_loaded:
            self._forge._do_load("native", model=model_name, quant=quant)
            self._native_loaded = True
        else:
            adapter = self._forge._services["native"]
            before = adapter.vram_mb
            adapter.load(model_name, quant=quant)
            after = adapter.vram_mb
            diff = after - before
            self._forge._vram_allocations["native"] = after
            self._forge._vram_free_mb -= diff
            logger.info("ForgeProxy: swapped to model=%s vram_delta=%dMB", model_name, diff)

    def unload(self) -> None:
        if self._native_loaded:
            self._forge._do_unload("native")
            self._native_loaded = False

    def infer(self, payload: dict) -> dict:
        adapter = self._forge._services["native"]
        return adapter.infer(payload)

    @property
    def _loaded_model(self) -> str | None:
        adapter = self._forge._services.get("native")
        if adapter and hasattr(adapter, "_svc"):
            return adapter._svc.model_name
        return None
