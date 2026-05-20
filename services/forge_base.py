"""ForgeService — base class for all Forge-managed GPU services.

Services implement three methods:
    load(model_name)   — load model into VRAM
    unload()           — release model from VRAM
    infer(payload)     — run inference, return result dict

No Starlette, no TNAP, no Governor leases.
Services take dicts and return dicts. HTTP stays at the boundary.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Optional

from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)


class ForgeService:
    """Base class for Forge-managed services.

    Attributes:
        vram_mb: Estimated VRAM footprint in MB. 0 = dynamic/CPU (no tracking).
        service_name: Unique identifier for scheduling and logging.
        default_model: Model name used when none specified.
    """

    vram_mb: int = 0
    service_name: str = ""
    default_model: str = ""
    persistence: Persistence = Persistence.TRANSIENT

    def __init__(self):
        self._loaded: bool = False
        self.model_name: Optional[str] = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Load model into VRAM. Blocking — Forge runs this in a thread."""
        raise NotImplementedError(f"{self.__class__.__name__}.load() not implemented")

    def unload(self) -> None:
        """Release model from VRAM. Safe to call even if not loaded."""
        self._loaded = False
        self.model_name = None
        gc.collect()

    def infer(self, payload: dict) -> dict:
        """Run inference. Takes a dict, returns a dict."""
        raise NotImplementedError(f"{self.__class__.__name__}.infer() not implemented")

    def is_loaded(self) -> bool:
        return self._loaded

    def actual_vram_mb(self) -> int:
        """Report actual VRAM usage after load. Override for subprocess services."""
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        except ImportError:
            pass
        return 0
