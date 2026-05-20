"""Wan2GP Studio MCP Server.

Wraps the Forge GPU inference gateway, exposing wan2gp's 20+ model families
as MCP tools for AI assistants (Claude Desktop, ChatGPT) and the web UI.

Tools:
- generate_video: Text/image-to-video (wan, hunyuan, ltx2)
- generate_image: Text-to-image (flux, qwen)
- generate_3d: Image-to-3D mesh (trellis, anigen)
- generate_audio: TTS + sound effects
- list_models: Discover available model families
- forge_status: GPU/VRAM usage and loaded services
"""
from __future__ import annotations

import os

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from loguru import logger

from .forge_client import ForgeClient


@lifespan
async def service_lifespan(server: FastMCP):
    """Initialize Forge client on startup, cleanup on shutdown."""
    forge = ForgeClient()
    logger.info("Wan2GP Studio MCP server starting")
    logger.info("Forge URL: {}", forge.base_url)
    try:
        yield {"forge_client": forge}
    finally:
        await forge.close()
        logger.info("Shutdown complete")


mcp = FastMCP(
    name="wan2gp-studio",
    instructions=(
        "GPU inference server. Use `run` to call any registered service with "
        "any parameters. Use `list_models` to discover available services and "
        "models. Use `forge_status` to check GPU/VRAM state."
    ),
    lifespan=service_lifespan,
)


# ========== TOOL REGISTRATION ==========

from .tools.generate import run
from .tools.status import list_models, forge_status

mcp.add_tool(run)
mcp.add_tool(list_models)
mcp.add_tool(forge_status)


# ========== ASGI APP (for uvicorn) ==========

app = mcp.http_app(stateless_http=True)


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Wan2GP Studio MCP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")

    mcp.run(transport="http", host=host, port=port)
