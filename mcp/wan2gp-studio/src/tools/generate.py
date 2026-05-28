"""Generation tool — thin passthrough to /v1/run.

One tool, one code path. The service registry defines what services exist.
The service handlers define what params they accept. The MCP just passes it
through.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field


async def run(
    service: Annotated[str, Field(
        description="Service name from the registry. Use list_models to discover.",
    )],
    params: Annotated[dict[str, Any] | None, Field(
        description="Parameters passed through to the service. Common: model, prompt, "
                    "text, image_b64, audio_b64, seed, steps, guidance, width, height, "
                    "frames, negative_prompt, voice, language. All optional.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Run any registered service. Single entry point for all inference.

    Use list_models to discover available services, categories, and models.
    Pass whatever keys the service needs — they go straight through.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")
    payload = {"service": service, **(params or {})}
    return await client.invoke(payload)
