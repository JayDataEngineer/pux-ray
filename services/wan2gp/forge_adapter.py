"""Wan2GP ForgeService adapter — wraps Wan2GPService for forge integration.

Reports VRAM via torch.cuda.memory_allocated() diff so the forge's eviction
logic works correctly. mmgp handles all internal VRAM management; this just
gives the forge a signal for when to evict.
"""
from __future__ import annotations

import gc
import logging

import torch

from services.forge_base import ForgeService
from services.wan2gp.deployment import Wan2GPService

logger = logging.getLogger(__name__)


class Wan2GPForgeService(ForgeService):
    """ForgeService adapter around Wan2GPService — reports real VRAM usage."""

    service_name = "wan2gp"
    default_model = "wan/t2v"

    def __init__(self):
        super().__init__()
        self._svc = Wan2GPService()
        self._vram_baseline = _get_vram_mb()
        self._vram_mb = 0

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        try:
            before = _get_vram_mb()
        except Exception:
            before = self._vram_baseline
        self._svc.load(model_name, quant=quant)
        self._loaded = True
        try:
            self._vram_mb = max(0, _get_vram_mb() - before)
            logger.info("Wan2GP adapter: model=%s vram=%dMB", model_name, self._vram_mb)
        except Exception:
            pass

    def unload(self) -> None:
        self._svc.unload()
        self._loaded = False
        self._vram_mb = 0
        gc.collect()

    def infer(self, payload: dict) -> dict:
        return self._svc.infer(payload)


def _get_vram_mb() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
    return 0
