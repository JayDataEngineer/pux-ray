"""Forge adapter for NativeDiffusersService.

Same interface as Wan2GPForgeService — wraps the native service for
forge integration with VRAM reporting and auto-evict support.
"""
from __future__ import annotations

import gc
import logging

import torch

from services.forge_base import ForgeService
from services.native.service import NativeDiffusersService

logger = logging.getLogger(__name__)


class NativeForgeService(ForgeService):
    """ForgeService adapter around NativeDiffusersService.

    Reports VRAM via torch.cuda.memory_allocated() diff so the forge's
    eviction logic works correctly.
    """

    service_name = "native"
    default_model = "z-image-turbo"

    def __init__(self):
        super().__init__()
        self._svc = NativeDiffusersService()
        self._vram_mb = 0

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        before = self._get_vram_mb()
        self._svc.load(model_name, quant=quant)
        self._loaded = True
        self._vram_mb = max(0, self._get_vram_mb() - before)
        logger.info("Native adapter: model=%s vram=%dMB", model_name, self._vram_mb)

    def unload(self) -> None:
        self._svc.unload()
        self._loaded = False
        self._vram_mb = 0
        gc.collect()

    def infer(self, payload: dict) -> dict:
        return self._svc.infer(payload)

    @staticmethod
    def _get_vram_mb() -> int:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return 0
