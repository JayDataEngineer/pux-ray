"""Idle model auto-unload.

Background task that periodically checks when models were last used
and unloads them if they've been idle beyond the configured timeout.

Each service with a model registers itself via watch(service).
The watcher calls service.close() and resets service._loaded when idle.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from ..settings import get_settings


class IdleWatcher:
    """Tracks model services and unloads them after idle timeout."""

    def __init__(self):
        self._services: dict[str, object] = {}
        self._last_used: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def watch(self, name: str, service: object) -> None:
        """Register a service for idle tracking."""
        self._services[name] = service
        self._last_used[name] = time.monotonic()

    def touch(self, name: str) -> None:
        """Mark a service as recently used."""
        if name in self._last_used:
            self._last_used[name] = time.monotonic()

    async def start(self) -> None:
        """Start the background eviction loop."""
        settings = get_settings()
        if settings.idle_timeout <= 0:
            logger.info("Idle auto-unload disabled (MEDIA_IDLE_TIMEOUT=0)")
            return

        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Idle watcher started (timeout={settings.idle_timeout}s, "
            f"check every 300s)"
        )

    async def stop(self) -> None:
        """Stop the background eviction loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Idle watcher stopped")

    async def _loop(self) -> None:
        """Background loop that checks idle models every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._evict_idle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Idle watcher error: {e}")

    async def _evict_idle(self) -> None:
        """Unload models that have been idle beyond the timeout."""
        settings = get_settings()
        timeout = settings.idle_timeout
        if timeout <= 0:
            return

        now = time.monotonic()
        evicted = []

        for name, last_used in list(self._last_used.items()):
            service = self._services.get(name)
            if service is None:
                continue

            idle_seconds = now - last_used

            # Only evict if the service has a model loaded
            is_loaded = getattr(service, "_loaded", False)
            if not is_loaded:
                continue

            # Don't evict services that had a load error (they won't re-load)
            has_error = getattr(service, "_load_error", None)
            if has_error:
                continue

            if idle_seconds >= timeout:
                logger.info(
                    f"Unloading idle model '{name}' "
                    f"(idle for {idle_seconds:.0f}s >= {timeout}s)"
                )
                try:
                    await service.close()
                    evicted.append(name)
                except Exception as e:
                    logger.warning(f"Failed to unload idle model '{name}': {e}")

        if evicted:
            logger.info(f"Evicted {len(evicted)} idle model(s): {', '.join(evicted)}")


# Singleton
_watcher: IdleWatcher | None = None


def get_idle_watcher() -> IdleWatcher:
    global _watcher
    if _watcher is None:
        _watcher = IdleWatcher()
    return _watcher
