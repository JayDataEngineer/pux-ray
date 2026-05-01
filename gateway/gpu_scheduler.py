"""GPU Scheduler - coordinates model swaps across all GPU deployments.

Only one GPU model can run at a time on a single-GPU machine.
This actor serializes the load/unload sequence to prevent VRAM conflicts.

For Docker worker containers, the scheduler starts/stops the container
and waits for the health endpoint before returning.

For in-process Ray Serve deployments, the scheduler uses handle-based
load/unload as before.

Uses handle.options(method_name=...).remote() for Serve 2.x compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, Optional

import ray
from ray import serve

logger = logging.getLogger(__name__)

# Docker Compose file for worker containers
_COMPOSE_FILE = "/home/user/Documents/programs/ray/infra/docker/compose.workers.yaml"

# Services that run in Docker containers (HTTP workers)
_DOCKER_WORKERS = {
    "trellis": {"profile": "trellis", "port": 18401},
    "anigen": {"profile": "anigen", "port": 18402},
    "vibevoice": {"profile": "vibevoice", "port": 18403},
}


@ray.remote
class GPUScheduler:
    """Coordinates GPU model swaps. Ensures unload-then-load sequencing.

    Usage:
        scheduler = ray.get_actor("gpu_scheduler")
        await scheduler.acquire_gpu("llm", "qwen3.5-27b")
        # ... use the LLM ...
        await scheduler.acquire_gpu("trellis", "trellis")  # starts container, unloads LLM
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

            # Unload current (Docker worker or in-process)
            if self.current_service:
                await self._unload_current()

            # Also unload target if it has a different model (in-process only)
            if service_name not in _DOCKER_WORKERS:
                handle = self._get_handle(service_name)
                if handle:
                    try:
                        await handle.options(method_name="is_loaded").remote()
                        loaded = await asyncio.wait_for(
                            handle.options(method_name="is_loaded").remote(),
                            timeout=10,
                        )
                        if loaded and self.current_service == service_name:
                            await handle.options(
                                method_name="unload_model"
                            ).remote()
                    except Exception:
                        pass

            # Load requested service
            if service_name in _DOCKER_WORKERS:
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
        app_name = service_name  # app_name matches service name in deploy
        try:
            return serve.get_deployment_handle(service_name, app_name)
        except Exception:
            return None

    async def _unload_current(self) -> None:
        """Unload the currently loaded GPU model."""
        if not self.current_service:
            return

        svc = self.current_service
        if svc in _DOCKER_WORKERS:
            await self._stop_worker(svc)
        else:
            handle = self._get_handle(svc)
            if handle:
                logger.info("Unloading %s/%s from GPU",
                           svc, self.current_model)
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

    async def status(self) -> dict:
        """Get current GPU allocation status."""
        return {
            "current_service": self.current_service,
            "current_model": self.current_model,
        }

    # ------------------------------------------------------------------
    # Docker worker lifecycle
    # ------------------------------------------------------------------

    async def _start_worker(self, service_name: str) -> None:
        """Start a Docker worker container and wait for it to be healthy."""
        worker = _DOCKER_WORKERS[service_name]
        profile = worker["profile"]
        port = worker["port"]

        logger.info("Starting Docker worker: %s (profile=%s, port=%d)",
                    service_name, profile, port)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._docker_compose_up,
            profile,
        )

        # Wait for the health endpoint
        healthy = await self._poll_worker_health(port, timeout=180)
        if not healthy:
            raise RuntimeError(
                f"Worker {service_name} did not become healthy within 180s"
            )

        logger.info("Docker worker %s is healthy", service_name)

    async def _stop_worker(self, service_name: str) -> None:
        """Stop a Docker worker container."""
        worker = _DOCKER_WORKERS[service_name]
        profile = worker["profile"]
        container_name = f"{profile}-worker"

        logger.info("Stopping Docker worker: %s", service_name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._docker_compose_stop,
            container_name,
        )

    async def _check_healthy(self, service_name: str) -> bool:
        """Check if a service is currently healthy."""
        if service_name in _DOCKER_WORKERS:
            port = _DOCKER_WORKERS[service_name]["port"]
            return await self._poll_worker_health(port, timeout=5)
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

    async def _poll_worker_health(self, port: int, timeout: int = 30) -> bool:
        """Poll a worker's /health endpoint until it returns 200."""
        import httpx

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(2)
        return False

    @staticmethod
    def _docker_compose_up(profile: str) -> None:
        """Run docker compose up (blocking)."""
        subprocess.run(
            [
                "docker", "compose",
                "-f", _COMPOSE_FILE,
                "--profile", profile,
                "up", "-d",
            ],
            capture_output=True, text=True, timeout=120,
            check=True,
        )

    @staticmethod
    def _docker_compose_stop(container_name: str) -> None:
        """Run docker compose stop (blocking)."""
        subprocess.run(
            [
                "docker", "compose",
                "-f", _COMPOSE_FILE,
                "stop", container_name,
            ],
            capture_output=True, text=True, timeout=60,
            check=False,  # don't raise if already stopped
        )
