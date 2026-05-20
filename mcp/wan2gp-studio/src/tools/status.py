"""Status and model discovery tools.

Gets live data from the ingress instead of hardcoding model families.
The ingress serves /v1/models from the service registry — single source
of truth.
"""
from __future__ import annotations

from typing import Annotated

from fastmcp import Context
from pydantic import Field


async def list_models(
    ctx: Context | None = None,
) -> dict:
    """List all available models from the live service registry.

    Returns the same model catalog that /v1/models serves, grouped
    by category with GPU requirements and descriptions.
    """
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        catalog = await forge.list_models()
    except Exception as e:
        return {"status": "error", "error": f"Failed to fetch models: {e}"}

    # Enrich with GPU status
    gpu_status = {}
    try:
        gpu_status = await forge.status()
    except Exception:
        pass

    return {"catalog": catalog, "gpu_status": gpu_status}


async def forge_status(
    detailed: Annotated[bool, Field(
        description="Include per-service VRAM breakdown",
    )] = False,
    ctx: Context | None = None,
) -> dict:
    """Check GPU status, VRAM usage, and currently loaded services."""
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        status = await forge.status()
    except Exception as e:
        return {"status": "error", "error": f"Failed to reach Forge: {e}"}

    if not detailed:
        status.pop("gpu_nodes", None)

    return status
