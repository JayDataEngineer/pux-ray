"""HTTP client for the Tech Noir API ingress.

Routes through the same dispatch pipeline as external clients
(/v1/run), so the MCP gets service registry lookup, model
resolution, and proper routing — no duplicated logic.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

DEFAULT_API_URL = "http://tech-noir-ray-serve-svc.ai-services:8000"
DEFAULT_TIMEOUT = 600.0  # LLM load after eviction can take >300s


class ForgeClient:
    """Async HTTP client for the Tech Noir API.

    Uses /v1/run (unified dispatch) and /v1/models (model discovery)
    instead of hitting the Forge directly. This gives the MCP the same
    service registry, model resolution, and routing as external clients.
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.environ.get("FORGE_URL", DEFAULT_API_URL)).rstrip("/")
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
        """Send an inference request via /v1/run.

        The payload must include 'service' key (e.g. "wan2gp").
        All other keys are passed through as-is.
        """
        # Drop None values — unresolved template placeholders
        payload = {k: v for k, v in payload.items() if v is not None}

        client = await self._get_client()
        logger.info("API invoke: service={} model={}",
                     payload.get("service"), payload.get("model", "default"))
        resp = await client.post(f"{self.base_url}/v1/run", json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            text = resp.text[:500]
            logger.error("ForgeClient: non-JSON response: {}", text)
            return {"status": "error", "error": text or f"HTTP {resp.status_code}"}

    async def status(self) -> dict[str, Any]:
        """Get Forge GPU status (VRAM, loaded services)."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/status")
        resp.raise_for_status()
        return resp.json()

    async def list_models(self, category: str | None = None) -> dict[str, Any]:
        """Get the full model catalog from the ingress."""
        client = await self._get_client()
        url = f"{self.base_url}/v1/models"
        if category:
            url += f"?category={category}"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_services(self) -> dict[str, Any]:
        """Get all registered services from the ingress."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/v1/services")
        resp.raise_for_status()
        return resp.json()

    async def get_service(self, service_name: str) -> dict[str, Any]:
        """Get detailed info about a specific service."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/v1/services/{service_name}")
        resp.raise_for_status()
        return resp.json()
