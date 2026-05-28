"""Wan2GP Studio MCP Server.

Wraps the Forge GPU inference gateway and Workflow Engine, exposing all AI
operations as MCP tools for AI assistants and the web UI.

Forge tools:
- run: Invoke any registered GPU service
- list_models: Discover available model families
- forge_status: GPU/VRAM usage and loaded services

Workflow tools:
- workflow_list_specs: List available pipeline specs
- workflow_get_spec: Get spec details and input schema
- workflow_start_run: Start a new run (manual by default)
- workflow_get_run: Get run status and step states
- workflow_cancel_run: Cancel a running workflow
- workflow_execute_step: Execute a single step in isolation
- workflow_approve_step: Approve a waiting step
- workflow_rerun_step: Rerun from a specific step
"""
from __future__ import annotations

import os

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from loguru import logger

from .forge_client import ForgeClient
from .workflow_client import WorkflowClient


@lifespan
async def service_lifespan(server: FastMCP):
    """Initialize clients on startup, cleanup on shutdown."""
    forge = ForgeClient()
    workflow = WorkflowClient()
    logger.info("Wan2GP Studio MCP server starting")
    logger.info("Forge URL: {}", forge.base_url)
    logger.info("Workflow URL: {}", workflow.base_url)
    try:
        yield {"forge_client": forge, "workflow_client": workflow}
    finally:
        await forge.close()
        await workflow.close()
        logger.info("Shutdown complete")


mcp = FastMCP(
    name="wan2gp-studio",
    instructions=(
        "GPU inference and workflow orchestration server. "
        "Use `run` to call any registered GPU service. "
        "Use `list_models` to discover available services and models. "
        "Use `forge_status` to check GPU/VRAM state. "
        "Use `workflow_list_specs` to discover pipelines. "
        "Use `workflow_start_run` to create a manual run, then "
        "`workflow_execute_step` to run steps one at a time. "
        "Use `workflow_get_run` to check status and artifacts."
    ),
    lifespan=service_lifespan,
)


# ========== TOOL REGISTRATION ==========

from .tools.generate import run
from .tools.status import list_models, forge_status
from .tools.workflow import (
    workflow_list_specs,
    workflow_get_spec,
    workflow_start_run,
    workflow_get_run,
    workflow_cancel_run,
    workflow_execute_step,
    workflow_approve_step,
    workflow_rerun_step,
)

mcp.add_tool(run)
mcp.add_tool(list_models)
mcp.add_tool(forge_status)

mcp.add_tool(workflow_list_specs)
mcp.add_tool(workflow_get_spec)
mcp.add_tool(workflow_start_run)
mcp.add_tool(workflow_get_run)
mcp.add_tool(workflow_cancel_run)
mcp.add_tool(workflow_execute_step)
mcp.add_tool(workflow_approve_step)
mcp.add_tool(workflow_rerun_step)


# ========== ASGI APP (for uvicorn) ==========

app = mcp.http_app(stateless_http=True)


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Wan2GP Studio MCP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")

    mcp.run(transport="http", host=host, port=port)
