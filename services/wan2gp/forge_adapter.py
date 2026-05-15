"""Wan2GP ForgeService adapter — wraps Wan2GPService for forge integration.

vram_mb=0 because mmgp handles all VRAM management internally.
The forge's eviction logic delegates to Wan2GPService.load/unload/infer.
"""
from __future__ import annotations

import gc
import logging

from services.forge_base import ForgeService
from services.wan2gp.deployment import Wan2GPService

logger = logging.getLogger(__name__)


class Wan2GPForgeService(ForgeService):
    """ForgeService adapter around Wan2GPService.

    Wan2GPService already has load/unload/infer with the right signatures.
    This adapter just bridges the interface and sets vram_mb=0 (mmgp self-manages).
    """

    vram_mb = 0
    service_name = "wan2gp"
    default_model = "wan/t2v-14B"

    def __init__(self):
        super().__init__()
        self._svc = Wan2GPService()

    def load(self, model_name: str | None = None) -> None:
        model_name = model_name or self.default_model
        self._svc.load(model_name)
        self._loaded = True

    def unload(self) -> None:
        self._svc.unload()
        self._loaded = False
        gc.collect()

    def infer(self, payload: dict) -> dict:
        return self._svc.infer(payload)
