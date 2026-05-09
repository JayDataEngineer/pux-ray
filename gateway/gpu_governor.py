"""GPU Governor — VRAM lease manager for single-node RTX 4090.

Replaces the old GPUScheduler and MasterRouter with a proactive eviction
model. Every heavy GPU service requests a lease before loading. If VRAM is
tight, the Governor evicts the current holder BEFORE the new service loads,
eliminating the OOM window.

Lightweight services (faster_qwen3_tts, index_tts, vibevoice_cpp_gpu, vibevoice_cpp_cpu) coexist
without leases — they're small enough to share. Only heavy hitters (trellis,
ace_step, moss_soundeffect, anigen, see_through, comfyui, hy_motion, llm)
coordinate through the Governor.

Usage in a deployment::

    governor = ray.get_actor("gpu_governor")
    await governor.acquire.remote("trellis", 8192)
    # ... load model ...
    await governor.release.remote("trellis")
"""

from __future__ import annotations

import logging
from typing import Optional

import ray

logger = logging.getLogger(__name__)

# Heavy services that need exclusive GPU coordination.
# Values are estimated VRAM in MB (conservative — includes peak during inference).
HEAVY_SERVICES: dict[str, int] = {
    "trellis": 10_240,          # ~8GB + overhead
    "ace_step": 8_192,          # ~6GB + overhead
    "moss_soundeffect": 18_432, # ~16GB + overhead
    "anigen": 10_240,           # ~8GB + overhead
    "see_through": 6_144,       # ~4GB + overhead
    "comfyui": 14_336,          # ~12GB + overhead
    "hy_motion": 6_144,         # ~4GB + overhead
    "llm": 20_480,              # ~18GB + overhead (27B Q6_K)
}

TOTAL_VRAM_MB = 24_576
RESERVED_MB = 2_048  # OS + CUDA driver + lightweight services overhead
AVAILABLE_MB = TOTAL_VRAM_MB - RESERVED_MB


@ray.remote
class GPUGovernor:
    """VRAM lease manager. One heavy service at a time with explicit eviction."""

    def __init__(self):
        self._holder: Optional[str] = None
        self._pending_eviction: Optional[str] = None

    async def acquire(self, service: str) -> dict:
        """Request GPU for a heavy service. Evicts current holder if any.

        Returns immediately if service already holds the lease.
        If another service holds it, triggers eviction first.
        """
        vram = HEAVY_SERVICES.get(service, 8_192)

        if self._holder == service:
            return {"granted": True, "evicted": None}

        evicted = self._holder
        if evicted:
            logger.info("Governor: evicting %s for %s (need ~%dMB)", evicted, service, vram)
            await self._evict(evicted)

        self._holder = service
        logger.info("Governor: lease granted to %s (%dMB)", service, vram)
        return {"granted": True, "evicted": evicted}

    async def release(self, service: str) -> None:
        """Release GPU lease. No-op if service doesn't hold lease."""
        if self._holder == service:
            logger.info("Governor: lease released by %s", service)
            self._holder = None

    async def park(self, service: str) -> None:
        """Mark service as parked (CPU). Lease released, but state kept for fast swap."""
        if self._holder == service:
            logger.info("Governor: parking %s", service)
            self._holder = None

    async def unpark(self, service: str, vram_mb: int) -> dict:
        """Request GPU for a parked service. Evicts current if needed."""
        return await self.acquire(service)

    async def status(self) -> dict:
        return {
            "holder": self._holder,
            "holder_vram_mb": HEAVY_SERVICES.get(self._holder, 0) if self._holder else 0,
            "free_mb": AVAILABLE_MB,
            "total_mb": TOTAL_VRAM_MB,
            "reserved_mb": RESERVED_MB,
        }

    async def _evict(self, service: str) -> None:
        """Tell a service to unload and wait for confirmation."""
        from ray import serve

        logger.info("Governor: evicting %s — getting deployment handle", service)
        try:
            handle = serve.get_deployment_handle(service, app_name=service)
        except Exception as e:
            logger.warning("Governor: cannot get handle for %s (may be scaled to 0): %s", service, e)
            self._holder = None
            return

        try:
            await handle.unload_model.remote()
            logger.info("Governor: eviction of %s complete", service)
        except Exception as e:
            logger.warning("Governor: eviction remote call to %s failed (replica may be gone): %s", service, e)
        self._holder = None
