"""Status, model discovery, and service registry tools.

Gets live data from the ingress instead of hardcoding model families.
The ingress serves /v1/models from the service registry — single source
of truth.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field


async def list_models(
    category: Annotated[str | None, Field(
        description="Filter by category (e.g. 'tts', 'image', 'video', '3d', 'audio', 'llm').",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """List all available models from the live service registry.

    Returns model catalog with categories, GPU requirements, output types, and descriptions.
    Optionally filter by category to narrow results.
    """
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        catalog = await forge.list_models(category=category)
    except Exception as e:
        return {"status": "error", "error": f"Failed to fetch models: {e}"}

    # Enrich with GPU status
    gpu_status = {}
    try:
        gpu_status = await forge.status()
    except Exception:
        pass

    return {"catalog": catalog, "gpu_status": gpu_status}


async def list_services(
    ctx: Context | None = None,
) -> dict:
    """List all registered services with deployment info and model aliases.

    Different from list_models — shows the service registry structure
    (deployment targets, model aliases) rather than the model catalog.
    """
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        services = await forge.list_services()
        # The ingress returns a list; wrap in dict for FastMCP structured output
        if isinstance(services, list):
            return {"services": services}
        return services
    except Exception as e:
        return {"status": "error", "error": f"Failed to fetch services: {e}"}


async def get_service(
    service_name: Annotated[str, Field(
        description="Service name to inspect (e.g. 'wan2gp', 'kokoro', 'llm').",
    )],
    ctx: Context | None = None,
) -> dict:
    """Get detailed info about a specific service: models, params, capabilities."""
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        return await forge.get_service(service_name)
    except Exception as e:
        return {"status": "error", "error": f"Failed to fetch service info: {e}"}


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


async def list_pipelines(
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List all available DAG pipelines with their parameter schemas.

    Returns each pipeline's ID, description, and parameter definitions
    (name, type, default, required). Use this to dynamically build forms.
    """
    import inspect
    from gateway.routes.workflows import _WORKFLOW_REGISTRY

    pipelines = []
    for pipeline_id, fn in sorted(_WORKFLOW_REGISTRY.items()):
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            if name in ("kwargs", "args"):
                continue
            ptype = "string"
            if param.annotation != inspect.Parameter.empty:
                ann = str(param.annotation)
                if "int" in ann: ptype = "integer"
                elif "float" in ann: ptype = "number"
                elif "bool" in ann: ptype = "boolean"
                elif "dict" in ann: ptype = "object"
            params.append({
                "name": name,
                "type": ptype,
                "required": param.default == inspect.Parameter.empty,
                "default": None if param.default == inspect.Parameter.empty else param.default,
            })
        pipelines.append({
            "id": pipeline_id,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "params": params,
        })
    return pipelines
