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
- tts_speak: Generate speech (kokoro, moss_tts, espeak, index_tts)
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
        "  - `tts_speak` to generate speech (kokoro CPU, moss GPU, espeak CPU, index_tts GPU)\n"
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
from .tools.image import generate, get_model_preset
from .tools.vnccs import generate_character_sheet, edit, clone_character
from .tools.status import list_models, list_services, get_service, forge_status, list_pipelines
from .tools.tts import tts_speak, tts_voices
from .tools.audio import (
    transcribe,
    generate_sound,
    generate_music,
    voice_creator,
    voice_creator_examples,
    voice_creator_batch,
    generate_batch,
    generate_music_batch,
    generate_sound_batch,
)
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

@mcp.resource("ui://apps/list", mime_type=MCP_APP_MIME,
             meta={"ui": {"prefersBorder": True}})
def apps_list_resource() -> str:
    """List all available MCP app widgets."""
    import json
    return json.dumps(list_apps())


def _register_app_resources():
    """Register individual app HTML resources.

    FastMCP resource templates don't propagate mime_type/meta to content
    items, so we register each app as its own resource instead.
    """
    APP_META = {"ui": {"prefersBorder": True}}

    for uri in APPS:
        html = get_app_html(uri)
        if html is None:
            continue

        # Capture html in closure
        def make_handler(content: str):
            def handler() -> str:
                return content
            return handler

        mcp.resource(uri, mime_type=MCP_APP_MIME, meta=APP_META)(
            make_handler(html)
        )


_register_app_resources()

# ========== TOOL REGISTRATION WITH MCP APP METADATA ==========
#
# Tools with a `meta.ui.resourceUri` pointing to an HTML widget will render
# inline in the assistant-ui chat via McpAppRenderer.  Tools without one
# fall back to plain-text rendering.

# Forge core
mcp.tool(run, meta={
    "ui": {"resourceUri": "ui://apps/generate"},
    "openai/toolInvocation/invoking": "Generating…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate, meta={
    "ui": {"resourceUri": "ui://apps/image"},
    "openai/toolInvocation/invoking": "Generating image…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(get_model_preset, meta={
    "description": "Get model-specific preset defaults for image generation",
})
mcp.tool(generate_character_sheet, meta={
    "ui": {"resourceUri": "ui://apps/image"},
    "openai/toolInvocation/invoking": "Generating character sheet…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(edit, meta={
    "ui": {"resourceUri": "ui://apps/image"},
    "openai/toolInvocation/invoking": "Editing image…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(clone_character, meta={
    "ui": {"resourceUri": "ui://apps/image"},
    "openai/toolInvocation/invoking": "Cloning character…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(list_models)
mcp.tool(list_services)
mcp.tool(get_service)
mcp.tool(list_pipelines, meta={
    "ui": {"resourceUri": "ui://apps/pipelines"},
    "openai/toolInvocation/invoking": "Loading pipelines…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(forge_status, meta={
    "ui": {"resourceUri": "ui://apps/admin"},
    "openai/toolInvocation/invoking": "Checking GPU…",
    "openai/toolInvocation/invoked": "Done",
})

# TTS
mcp.tool(tts_speak, meta={
    "ui": {"resourceUri": "ui://apps/tts"},
    "openai/toolInvocation/invoking": "Synthesizing speech…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(tts_voices)

# Audio (ASR + generation)
mcp.tool(transcribe, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Transcribing…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate_sound, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Generating sound…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate_music, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Generating music…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(voice_creator, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Creating voice…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(voice_creator_examples, meta={
    "description": "Get voice creation examples and presets from vendor demos",
})
mcp.tool(voice_creator_batch, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Creating voices…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate_batch, meta={
    "ui": {"resourceUri": "ui://apps/image"},
    "openai/toolInvocation/invoking": "Generating images…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate_music_batch, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Generating music…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(generate_sound_batch, meta={
    "ui": {"resourceUri": "ui://apps/audio"},
    "openai/toolInvocation/invoking": "Generating sounds…",
    "openai/toolInvocation/invoked": "Done",
})

# LLM
mcp.tool(chat)
mcp.tool(llm_configure)

# Admin
mcp.tool(load_service, meta={
    "ui": {"resourceUri": "ui://apps/admin"},
    "openai/toolInvocation/invoking": "Loading on GPU…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(unload_services, meta={
    "ui": {"resourceUri": "ui://apps/admin"},
    "openai/toolInvocation/invoking": "Releasing GPU…",
    "openai/toolInvocation/invoked": "Done",
})

# Workflows
mcp.tool(workflow_list_specs, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Loading pipelines…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_get_spec, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Loading spec…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_start_run, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Starting run…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_get_run, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Fetching status…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_cancel_run, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Cancelling…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_execute_step, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Executing step…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_approve_step, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Approving…",
    "openai/toolInvocation/invoked": "Done",
})
mcp.tool(workflow_rerun_step, meta={
    "ui": {"resourceUri": "ui://apps/workflow"},
    "openai/toolInvocation/invoking": "Rerunning step…",
    "openai/toolInvocation/invoked": "Done",
})


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
    lifespan=_base_app.lifespan,
)


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Wan2GP Studio MCP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")

    mcp.run(transport="http", host=host, port=port)
