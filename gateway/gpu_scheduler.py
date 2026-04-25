"""GPU Scheduler - coordinates model swaps across all GPU deployments.

Only one GPU model can run at a time on a single-GPU machine.
This actor serializes the load/unload sequence to prevent VRAM conflicts.

Uses handle.options(method_name=...).remote() for Serve 2.x compatibility.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Optional

import ray
from ray import serve

logger = logging.getLogger(__name__)


@ray.remote
class GPUScheduler:
    """Coordinates GPU model swaps. Ensures unload-then-load sequencing.

    Usage:
        scheduler = ray.get_actor("gpu_scheduler")
        await scheduler.acquire_gpu.remote("llm", "qwen3.5-27b")
        # ... use the LLM ...
        await scheduler.acquire_gpu.remote("tts", "qwen-tts")  # auto-unloads LLM
    """

    def __init__(self):
        self.current_service: Optional[str] = None
        self.current_model: Optional[str] = None
        self._lock = asyncio.Lock()

    async def acquire_gpu(self, service_name: str, model_name: str) -> bool:
        """Acquire the GPU for a service/model. Unloads current if different."""
        async with self._lock:
            # Already loaded?
            if (self.current_service == service_name
                    and self.current_model == model_name):
                handle = self._get_handle(service_name)
                if handle:
                    try:
                        loaded = await handle.options(
                            method_name="is_loaded"
                        ).remote()
                        if loaded:
                            return True
                    except Exception:
                        pass

            # Unload current
            if self.current_service and self.current_service != service_name:
                await self._unload_current()

            # Also unload target if it has a different model
            if (self.current_service == service_name
                    and self.current_model != model_name):
                handle = self._get_handle(service_name)
                if handle:
                    try:
                        await handle.options(
                            method_name="unload_model"
                        ).remote()
                    except Exception as e:
                        logger.warning("Failed to unload %s: %s", service_name, e)

            # Load requested model
            handle = self._get_handle(service_name)
            if not handle:
                raise ValueError(f"Unknown service: {service_name}")

            logger.info("Loading %s/%s on GPU", service_name, model_name)
            await handle.options(method_name="load_model").remote(model_name)

            self.current_service = service_name
            self.current_model = model_name
            logger.info("GPU now running %s/%s", service_name, model_name)
            return True

    def _get_handle(self, service_name: str) -> Any:
        """Get a Serve deployment handle by service name."""
        app_name = service_name  # app_name matches service name in deploy
        try:
            return serve.get_deployment_handle(service_name, app_name)
        except Exception:
            return None

    async def _unload_current(self) -> None:
        """Unload the currently loaded GPU model."""
        if not self.current_service:
            return

        handle = self._get_handle(self.current_service)
        if handle:
            logger.info("Unloading %s/%s from GPU",
                       self.current_service, self.current_model)
            try:
                await handle.options(method_name="unload_model").remote()
            except Exception as e:
                logger.warning("Error unloading %s: %s", self.current_service, e)

        self.current_service = None
        self.current_model = None

    async def release_gpu(self) -> None:
        """Explicitly unload the current model and free GPU."""
        async with self._lock:
            await self._unload_current()
            logger.info("GPU released")

    async def status(self) -> dict:
        """Get current GPU allocation status."""
        return {
            "current_service": self.current_service,
            "current_model": self.current_model,
        }
