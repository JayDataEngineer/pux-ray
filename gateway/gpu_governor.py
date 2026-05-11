"""GPU Governor — VRAM lease manager for single-node RTX 4090.

Coordination model:
  - The Master Router (services/creative/master_router.py) claims num_gpus: 1.0
    and performs explicit _load/_unload for heavy services (trellis, ace_step,
    comfyui, hy_motion, moss_soundeffect, anigen, see_through, llm).
    It manages GPU state independently — the Governor cannot directly evict
    models loaded inside the Master Router.
  - The Governor manages lightweight GPU services that use fractional GPU
    claims (faster_qwen3_tts, index_tts, vibevoice_cpp_gpu).
  - The API Ingress calls governor.acquire() for any service with needs_gpu=True.
    For heavy services, acquire() records the lease but eviction is best-effort
    (heavy services live inside master_router, not standalone deployments).

HEAVY_SERVICES here provides VRAM estimates for logging. It does NOT need to
match MasterRouter's HEAVY_SERVICES exactly — the router's set includes
Tier 2 services (vibevoice_microsoft, vibevoice_community_tts, phi4mm) that
are not yet deployed.
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

        # Heavy services live inside master_router — unload through it
        if service in HEAVY_SERVICES:
            try:
                router = serve.get_deployment_handle("master_router", app_name="forge")
                await router.unload_service.remote(service)
                logger.info("Governor: evicted %s via master_router", service)
            except Exception as e:
                logger.warning("Governor: failed to evict %s via master_router: %s", service, e)
            self._holder = None
            return

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
