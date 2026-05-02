"""GPU Scheduler - coordinates model swaps across all GPU deployments.

Only one GPU model can run at a time on a single-GPU machine.
This actor serializes the load/unload sequence to prevent VRAM conflicts.

Handles two types of GPU services:
- Docker workers (TRELLIS, AniGen, VibeVoice): start/stop containers
- Ray Serve deployments (LLM, IndexTTS, etc.): handle-based load/unload
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import ray
from ray import serve

logger = logging.getLogger(__name__)

# Docker worker config: service_name -> {port, profile}
DOCKER_WORKERS = {
    "trellis": {"port": 18401, "profile": "trellis"},
    "anigen": {"port": 18402, "profile": "anigen"},
    "vibevoice": {"port": 18403, "profile": "vibevoice"},
}

# Path to the Docker Compose file for workers
COMPOSE_FILE = Path(__file__).resolve().parent.parent / "infra" / "docker" / "compose.workers.yaml"


@ray.remote
class GPUScheduler:
    """Coordinates GPU model swaps. Ensures unload-then-load sequencing.

    Usage:
        scheduler = ray.get_actor("gpu_scheduler")
        await scheduler.acquire_gpu("trellis", "trellis")
        # ... use TRELLIS ...
        await scheduler.acquire_gpu("llm", "qwen3.5-27b")
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
                if await self._check_healthy(service_name):
                    return True

            # Unload current
            if self.current_service:
                await self._unload_current()

            # Load requested service
            if service_name in DOCKER_WORKERS:
                await self._start_worker(service_name)
            else:
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
        try:
            return serve.get_deployment_handle(service_name, service_name)
        except Exception:
            return None

    async def _unload_current(self) -> None:
        """Unload the currently loaded GPU model."""
        if not self.current_service:
            return

        svc = self.current_service

        if svc in DOCKER_WORKERS:
            await self._stop_worker(svc)
        else:
            handle = self._get_handle(svc)
            if handle:
                logger.info("Unloading %s/%s from GPU", svc, self.current_model)
                try:
                    await handle.options(method_name="unload_model").remote()
                except Exception as e:
                    logger.warning("Error unloading %s: %s", svc, e)

        self.current_service = None
        self.current_model = None

    async def release_gpu(self) -> None:
        """Explicitly unload the current model and free GPU."""
        async with self._lock:
            await self._unload_current()
            logger.info("GPU released")

    # ── Docker worker lifecycle ───────────────────────────────────────────

    async def _start_worker(self, service_name: str) -> None:
        """Start a Docker worker container and wait for it to be healthy."""
        cfg = DOCKER_WORKERS[service_name]
        profile = cfg["profile"]
        port = cfg["port"]

        logger.info("Starting Docker worker: %s (profile=%s, port=%d)",
                     service_name, profile, port)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._docker_compose_up,
            profile,
        )

        # Wait for health check
        import httpx
        deadline = asyncio.get_event_loop().time() + 120
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        logger.info("Worker %s is healthy", service_name)
                        return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(2)

        raise RuntimeError(f"Worker {service_name} failed to become healthy within 120s")

    async def _stop_worker(self, service_name: str) -> None:
        """Stop a Docker worker container."""
        cfg = DOCKER_WORKERS[service_name]
        profile = cfg["profile"]

        logger.info("Stopping Docker worker: %s", service_name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._docker_compose_stop,
            profile,
        )

    @staticmethod
    def _docker_compose_up(profile: str) -> None:
        """Run docker compose up for a profile (blocking)."""
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE),
             "--profile", profile, "up", "-d", "--wait"],
            capture_output=True, text=True, timeout=120,
        )

    @staticmethod
    def _docker_compose_stop(profile: str) -> None:
        """Run docker compose stop for a profile (blocking)."""
        worker = f"{profile}-worker"
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE),
             "stop", worker],
            capture_output=True, text=True, timeout=60,
        )

    # ── Health checks ─────────────────────────────────────────────────────

    async def _check_healthy(self, service_name: str) -> bool:
        """Check if a service is currently healthy."""
        if service_name in DOCKER_WORKERS:
            import httpx
            port = DOCKER_WORKERS[service_name]["port"]
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    return resp.status_code == 200
            except (httpx.ConnectError, httpx.TimeoutException):
                return False
        else:
            handle = self._get_handle(service_name)
            if handle:
                try:
                    return await handle.options(
                        method_name="is_loaded"
                    ).remote()
                except Exception:
                    return False
            return False

    async def status(self) -> dict:
        """Get current GPU allocation status."""
        return {
            "current_service": self.current_service,
            "current_model": self.current_model,
        }
