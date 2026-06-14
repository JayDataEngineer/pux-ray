"""Forge adapter — same interface as Wan2GPForgeService."""
from __future__ import annotations
import gc, logging, torch
from services.forge_base import ForgeService
from services.native.service import NativeService

logger = logging.getLogger(__name__)

class NativeForgeService(ForgeService):
    service_name = "native"
    default_model = "z-image-turbo"

    def __init__(self):
        super().__init__()
        self._svc = NativeService()
        self._vram_mb = 0

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        before = self._vram_allocated()
        self._svc.load(model_name or self.default_model, quant)
        self._loaded = True
        self._vram_mb = max(0, self._vram_allocated() - before)

    def unload(self) -> None:
        self._svc.unload()
        self._loaded = False
        self._vram_mb = 0

    def infer(self, payload: dict) -> dict:
        return self._svc.infer(payload)

    @staticmethod
    def _vram_allocated() -> int:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return 0
