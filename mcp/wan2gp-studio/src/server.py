"""Wan2GP Studio MCP Server.

Wraps the Forge GPU inference gateway and Workflow Engine, exposing all AI
operations as MCP tools for AI assistants and the web UI.

Forge tools:
- run: Invoke any registered GPU service
- list_models: Discover available model families (with optional category filter)
- list_services: List all registered services with deployment info
- get_service: Get detailed info about a specific service
- forge_status: GPU/VRAM usage and loaded services

TTS tools:
- tts_speak: Generate speech (kokoro, qwen3_tts, moss_voicegenerator)
- tts_voices: List available TTS engines and voice presets

Audio tools:
- transcribe: Speech-to-text (faster_whisper CPU, vibevoice GPU)
- generate_sound: Text-to-sound-effect (MOSS-SoundEffect 8B)
- generate_music: Text-to-music (ACE-Step 1.5)

LLM tools:
- chat: Send messages to the LLM (llama.cpp GPU)
- llm_configure: Configure LLM model, system prompt, hardware settings

Admin tools:
- load_service: Preload a model on GPU
- unload_services: Release all GPU memory

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
from .apps.registry import get_app_html, list_apps, APPS, MCP_APP_MIME


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
        "GPU inference and workflow orchestration server.\n\n"
        "Generation:\n"
        "  - `run` to call any registered GPU service directly\n"
        "  - `list_models` to discover available models by category\n"
        "  - `list_services` to see all registered services and their capabilities\n"
        "  - `get_service` for detailed info on a specific service\n\n"
        "Audio:\n"
        "  - `tts_speak` to generate speech (kokoro CPU, qwen3_tts GPU, moss GPU)\n"
        "  - `tts_voices` to list TTS engines and voice presets\n"
        "  - `transcribe` for speech-to-text (whisper CPU, vibevoice GPU)\n"
        "  - `generate_sound` for text-to-sound-effects (MOSS-SoundEffect)\n"
        "  - `generate_music` for text-to-music (ACE-Step)\n\n"
        "LLM:\n"
        "  - `chat` to send messages to the on-prem LLM\n"
        "  - `llm_configure` to change model or settings\n\n"
        "Admin:\n"
        "  - `forge_status` to check GPU/VRAM state\n"
        "  - `load_service` to preload a model on GPU\n"
        "  - `unload_services` to free all GPU memory\n\n"
        "Workflows:\n"
        "  - `workflow_list_specs` to discover pipeline definitions\n"
        "  - `workflow_start_run` to create a manual run, then\n"
        "  - `workflow_execute_step` to run steps one at a time\n"
        "  - `workflow_get_run` to check status and artifacts"
    ),
    lifespan=service_lifespan,
)


# ========== TOOL REGISTRATION ==========

from .tools.generate import run
from .tools.status import list_models, list_services, get_service, forge_status
from .tools.tts import tts_speak, tts_voices
from .tools.audio import transcribe, generate_sound, generate_music
from .tools.llm import chat, llm_configure
from .tools.admin import load_service, unload_services
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


# ========== RESOURCE REGISTRATION ==========

@mcp.resource("wan2gp://apps/list")
def apps_list_resource() -> str:
    """List all available MCP app widgets."""
    import json
    return json.dumps(list_apps())


@mcp.resource("wan2gp://apps/{app_name}")
def app_resource(app_name: str) -> str:
    """Serve an MCP app HTML widget."""
    html = get_app_html(f"wan2gp://apps/{app_name}")
    if html is None:
        raise ValueError(f"Unknown app: {app_name}")
    return html

# Forge core
mcp.add_tool(run)
mcp.add_tool(list_models)
mcp.add_tool(list_services)
mcp.add_tool(get_service)
mcp.add_tool(forge_status)

# TTS
mcp.add_tool(tts_speak)
mcp.add_tool(tts_voices)

# Audio (ASR + generation)
mcp.add_tool(transcribe)
mcp.add_tool(generate_sound)
mcp.add_tool(generate_music)

# LLM
mcp.add_tool(chat)
mcp.add_tool(llm_configure)

# Admin
mcp.add_tool(load_service)
mcp.add_tool(unload_services)

# Workflows
mcp.add_tool(workflow_list_specs)
mcp.add_tool(workflow_get_spec)
mcp.add_tool(workflow_start_run)
mcp.add_tool(workflow_get_run)
mcp.add_tool(workflow_cancel_run)
mcp.add_tool(workflow_execute_step)
mcp.add_tool(workflow_approve_step)
mcp.add_tool(workflow_rerun_step)


# ========== ASGI APP (for uvicorn) ==========

from starlette.applications import Starlette
from starlette.routing import Mount, Route

from .app_host import handle_app_host

_base_app = mcp.http_app(stateless_http=True)

app = Starlette(
    routes=[
        Route("/host", handle_app_host, methods=["POST"]),
        Mount("/", app=_base_app),
    ],
)


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Wan2GP Studio MCP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")

    mcp.run(transport="http", host=host, port=port)
