"""ForgeProxy — drop-in replacement for Wan2GPService that routes through Forge's VRAM ledger.

Workflow functions currently call:
    svc = get_service()  # bare Wan2GPService singleton
    svc.load("z_image")
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
    """Drop-in replacement for Wan2GPService with Forge VRAM tracking."""

    def __init__(self, forge_core: ForgeCore):
        self._forge = forge_core
        self._wan2gp_loaded = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        if not self._wan2gp_loaded:
            self._forge._do_load("wan2gp", model=model_name, quant=quant)
            self._wan2gp_loaded = True
        else:
            adapter = self._forge._services["wan2gp"]
            before = adapter.vram_mb
            adapter.load(model_name, quant=quant)
            after = adapter.vram_mb
            diff = after - before
            self._forge._vram_allocations["wan2gp"] = after
            self._forge._vram_free_mb -= diff
            logger.info("ForgeProxy: swapped to model=%s vram_delta=%dMB", model_name, diff)

    def unload(self) -> None:
        if self._wan2gp_loaded:
            self._forge._do_unload("wan2gp")
            self._wan2gp_loaded = False

    def infer(self, payload: dict) -> dict:
        adapter = self._forge._services["wan2gp"]
        return adapter.infer(payload)

    @property
    def _loaded_model(self) -> str | None:
        adapter = self._forge._services.get("wan2gp")
        if adapter and hasattr(adapter, "_svc"):
            return adapter._svc._loaded_model
        return None
