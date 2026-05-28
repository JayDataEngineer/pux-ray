"""Admin tools — GPU service lifecycle management.

- load_service: Preload a model/service on GPU
- unload_services: Release all GPU memory
"""
from __future__ import annotations

from typing import Annotated

from fastmcp import Context
from pydantic import Field


def _forge(ctx: Context) -> object:
    fc = ctx.lifespan_context.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


async def load_service(
    service: Annotated[str, Field(
        description="Service to preload on GPU. Use list_services to discover names. "
                    "E.g. 'wan2gp', 'llm', 'comfyui', 'kimodo_demo'.",
    )],
    model: Annotated[str | None, Field(
        description="Specific model variant to load (optional, uses service default if omitted).",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Preload a service/model on GPU. Blocks until loaded.

    Some models are large and take 30-60s to load. Check forge_status afterward
    to verify VRAM usage. Only one GPU service can be loaded at a time (VRAM shared).
    """
    forge = _forge(ctx)

    payload = {"service": service}
    if model:
        payload["model"] = model

    # Use the admin/load gateway endpoint
    import httpx
    import os
    base_url = os.environ.get("FORGE_URL", "http://tech-noir-ray-serve-svc.ai-services:8000").rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.post(f"{base_url}/admin/load", json=payload)
        resp.raise_for_status()
        return resp.json()


async def unload_services(
    ctx: Context | None = None,
) -> dict:
    """Release all GPU memory by unloading every loaded service.

    Use this when switching between large models that don't fit in VRAM together.
    Returns the freed GPU status.
    """
    import httpx
    import os
    base_url = os.environ.get("FORGE_URL", "http://tech-noir-ray-serve-svc.ai-services:8000").rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.post(f"{base_url}/admin/unload", json={})
        resp.raise_for_status()
        return resp.json()
