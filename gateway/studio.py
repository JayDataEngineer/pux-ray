"""Studio switcher — unified interface for switching between GPU tools.

Provides the "app launcher" that lets users swap between interactive creative
tools (ComfyUI, StretchyStudio, etc.) with one click. Handles GPU model
swapping via the GPUScheduler and exposes service metadata for the frontend.

Endpoints:
    GET  /studio              — serve the studio HTML page
    GET  /studio/api/apps     — all services with status + UI metadata
    POST /studio/api/switch   — swap GPU to a specific service
    POST /studio/api/release  — release GPU (unload everything)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from gateway.dashboard import query_service_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Studio app registry — extends KNOWN_DEPLOYMENTS with UI metadata
# ---------------------------------------------------------------------------

STUDIO_APPS: dict[str, dict] = {
    # --- Interactive tools (have native Web UIs) ---
    "comfyui": {
        "label": "ComfyUI",
        "url": "/comfyui/",
        "has_ui": True,
        "category": "Image",
        "gpu": True,
        "manage_type": "subprocess",
    },
    # --- API-only tools ---
    "llm": {
        "label": "LLM (llama.cpp)",
        "url": None,
        "has_ui": False,
        "category": "LLM",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "qwen3.6-27b-ud-q4_k_xl",
    },
    "trellis": {
        "label": "TRELLIS.2",
        "url": None,
        "has_ui": False,
        "category": "3D",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "trellis",
    },
    "anigen": {
        "label": "AniGen",
        "url": None,
        "has_ui": False,
        "category": "3D",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "anigen",
    },
    "ace_step": {
        "label": "ACE-Step",
        "url": None,
        "has_ui": False,
        "category": "Music",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "ace_step",
    },
    "see_through": {
        "label": "See-Through",
        "url": None,
        "has_ui": False,
        "category": "Creative",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "see_through",
    },
    "index_tts": {
        "label": "IndexTTS",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "index_tts",
    },
    "qwen_tts": {
        "label": "Qwen3-TTS",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "qwen_tts",
    },
    "vibevoice_community_tts": {
        "label": "VibeVoice Community TTS (7B)",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "vibevoice_community_tts",
    },
    "gpt_sovits": {
        "label": "GPT-SoVITS",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "gpt_sovits",
    },
    "kokoro_tts": {
        "label": "Kokoro TTS",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": False,
        "manage_type": "none",
    },
    "espeak_tts": {
        "label": "eSpeak TTS",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": False,
        "manage_type": "none",
    },
    "faster_whisper": {
        "label": "Faster-Whisper",
        "url": None,
        "has_ui": False,
        "category": "ASR",
        "gpu": False,
        "manage_type": "none",
    },
    "vibevoice_microsoft": {
        "label": "VibeVoice Microsoft ASR",
        "url": None,
        "has_ui": False,
        "category": "ASR",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "vibevoice_microsoft",
    },
    "qwen_asr": {
        "label": "Qwen ASR",
        "url": None,
        "has_ui": False,
        "category": "ASR",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "qwen_asr",
    },
    # --- MCP Servers (always-on, CPU, managed outside Ray) ---
    "local_web_mcp": {
        "label": "Local Web MCP",
        "url": "/mcp/web/",
        "has_ui": False,
        "category": "MCP",
        "gpu": False,
        "manage_type": "persistent",
    },
    "media_analysis_mcp": {
        "label": "Media Analysis MCP",
        "url": "/mcp/media/",
        "has_ui": False,
        "category": "MCP",
        "gpu": False,
        "manage_type": "persistent",
    },
}

# Ordered categories for sidebar grouping
CATEGORY_ORDER = ["Image", "LLM", "3D", "Music", "Creative", "TTS", "ASR", "MCP"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_governor():
    """Get the GPUGovernor named actor, or None."""
    try:
        return ray.get_actor("gpu_governor")
    except ValueError:
        return None


async def _get_governor_state() -> dict:
    """Get current governor state (holder service)."""
    governor = _get_governor()
    if not governor:
        return {"holder": None}
    try:
        return await governor.status.remote()
    except Exception:
        return {"holder": None}


async def _is_comfyui_running() -> bool:
    """Check if ComfyUI subprocess is alive."""
    try:
        handle = serve.get_deployment_handle("comfyui", "comfyui")
        return await handle.options(method_name="is_running").remote()
    except Exception:
        return False


async def _stop_comfyui() -> None:
    """Stop the ComfyUI subprocess."""
    try:
        handle = serve.get_deployment_handle("comfyui", "comfyui")
        await handle.options(method_name="stop_comfyui").remote()
        logger.info("Studio: stopped ComfyUI")
    except Exception as e:
        logger.warning("Studio: failed to stop ComfyUI: %s", e)


async def _start_comfyui() -> bool:
    """Start the ComfyUI subprocess. Returns True on success."""
    try:
        handle = serve.get_deployment_handle("comfyui", "comfyui")
        result = await handle.options(method_name="start_comfyui").remote()
        return bool(result)
    except Exception as e:
        logger.error("Studio: failed to start ComfyUI: %s", e)
        return False


# ---------------------------------------------------------------------------
# Endpoint functions
# ---------------------------------------------------------------------------

_STUDIO_HTML = Path(__file__).parent / "studio.html"


async def studio_page(request: Request) -> HTMLResponse:
    """GET /studio — serve the studio switcher page."""
    return HTMLResponse(_STUDIO_HTML.read_text())


async def studio_apps(request: Request) -> JSONResponse:
    """GET /studio/api/apps — all services with status + UI metadata."""
    # Merge Ray Serve deployment status with studio metadata
    deploy_status = {s["name"]: s for s in query_service_status()}
    gov_state = await _get_governor_state()
    comfyui_running = await _is_comfyui_running()

    active_service = gov_state.get("holder")
    # If ComfyUI is running but governor doesn't track it
    if comfyui_running and not active_service:
        active_service = "comfyui"

    apps = []
    for name, meta in STUDIO_APPS.items():
        dep = deploy_status.get(name, {})
        is_active = (name == active_service)

        apps.append({
            "name": name,
            "label": meta["label"],
            "url": meta.get("url"),
            "has_ui": meta.get("has_ui", False),
            "category": meta["category"],
            "gpu": meta["gpu"],
            "manage_type": meta.get("manage_type", "none"),
            "default_model": meta.get("default_model"),
            "status": dep.get("status", "UNKNOWN"),
            "running_replicas": dep.get("running_replicas", 0),
            "is_active": is_active,
        })

    # Sort by category order
    cat_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    apps.sort(key=lambda a: (cat_index.get(a["category"], 99), a["label"]))

    return JSONResponse({
        "apps": apps,
        "active_service": active_service,
        "active_model": gov_state.get("holder"),
        "comfyui_running": comfyui_running,
    })


async def studio_switch(request: Request) -> JSONResponse:
    """POST /studio/api/switch — swap GPU to a specific service.

    Body: {"service": "comfyui", "model": "optional-model-name"}
    """
    body = await request.json()
    service_name = body.get("service", "")

    if service_name not in STUDIO_APPS:
        return JSONResponse(
            {"error": f"Unknown service: {service_name}"},
            status_code=400,
        )

    app_meta = STUDIO_APPS[service_name]
    governor = _get_governor()

    # --- Unload current ---
    # Stop ComfyUI if running
    if await _is_comfyui_running():
        await _stop_comfyui()

    # Release governor-managed service
    if governor:
        try:
            state = await governor.status.remote()
            holder = state.get("holder")
            if holder:
                await governor.release.remote(holder)
        except Exception as e:
            logger.warning("Studio: governor release failed: %s", e)

    # --- Load target ---
    if app_meta["manage_type"] == "subprocess":
        # Subprocess-managed services (ComfyUI)
        if service_name == "comfyui":
            success = await _start_comfyui()
            if not success:
                return JSONResponse(
                    {"error": "Failed to start ComfyUI"},
                    status_code=500,
                )
            return JSONResponse({
                "status": "loaded",
                "service": service_name,
                "url": app_meta.get("url"),
            })

    elif app_meta["manage_type"] == "scheduler":
        # Governor-managed services (BaseGPUDeployment)
        if not governor:
            return JSONResponse(
                {"error": "GPUGovernor not available"},
                status_code=503,
            )
        try:
            await governor.acquire.remote(service_name)
        except Exception as e:
            logger.error("Studio: acquire failed: %s", e)
            return JSONResponse(
                {"error": f"Failed to load {service_name}: {e}"},
                status_code=500,
            )
        return JSONResponse({
            "status": "loaded",
            "service": service_name,
            "url": app_meta.get("url"),
        })

    # CPU services or unmanaged
    return JSONResponse({
        "status": "loaded",
        "service": service_name,
        "url": app_meta.get("url"),
    })


async def studio_release(request: Request) -> JSONResponse:
    """POST /studio/api/release — unload everything from GPU."""
    # Stop ComfyUI if running
    if await _is_comfyui_running():
        await _stop_comfyui()

    # Release governor-managed service
    governor = _get_governor()
    if governor:
        try:
            state = await governor.status.remote()
            holder = state.get("holder")
            if holder:
                await governor.release.remote(holder)
        except Exception as e:
            logger.warning("Studio: governor release failed: %s", e)

    return JSONResponse({"status": "released"})
