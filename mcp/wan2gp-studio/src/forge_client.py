"""HTTP client for the Forge GPU inference gateway."""
from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

DEFAULT_FORGE_URL = "http://tech-noir-ray-serve-svc.ai-services:8000/forge"
DEFAULT_TIMEOUT = 300.0  # video generation is slow


class ForgeClient:
    """Async HTTP client for the Forge /forge endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.environ.get("FORGE_URL", DEFAULT_FORGE_URL)).rstrip("/")
        self.timeout = timeout or float(os.environ.get("FORGE_TIMEOUT", DEFAULT_TIMEOUT))
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an inference request to the Forge.

        The payload must include 'service' key (e.g. "wan2gp").
        All other keys are passed through to the service handler.
        """
        client = await self._get_client()
        logger.info("Forge invoke: service={} model={}",
                     payload.get("service"), payload.get("model", "default"))
        resp = await client.post(self.base_url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict[str, Any]:
        """Get Forge GPU status (VRAM, loaded services)."""
        client = await self._get_client()
        resp = await client.get(self.base_url)
        resp.raise_for_status()
        return resp.json()

    async def list_models(self) -> dict[str, Any]:
        """Discover available wan2gp models via Forge status.

        The Forge doesn't have a dedicated model list endpoint,
        so we return the status which includes loaded services.
        Model families are derived from the wan2gp service configuration.
        """
        return await self.status()
