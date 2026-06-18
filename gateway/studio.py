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
    "kimodo_demo": {
        "label": "Kimodo Motion",
        "url": "/kimodo/",
        "has_ui": True,
        "category": "3D",
        "gpu": True,
        "manage_type": "subprocess",
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
    # qwen_tts removed — superseded by MOSS VoiceGenerator (Tier A) + Kokoro (Tier D)
    "vibevoice_community_tts": {
        "label": "VibeVoice Community TTS (7B)",
        "url": None,
        "has_ui": False,
        "category": "TTS",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "vibevoice_community_tts",
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
    """Get the Forge deployment handle."""
    try:
        return serve.get_deployment_handle("forge", "forge")
    except Exception:
        return None


async def _get_governor_state() -> dict:
    """Get current Forge state (loaded services, VRAM)."""
    governor = _get_governor()
    if not governor:
        return {"loaded": {}}
    try:
        return await governor.status.remote()
    except Exception:
        return {"loaded": {}}


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

    active_service = None
    loaded = gov_state.get("loaded", {})
    if loaded:
        # Show the first loaded service as active
        active_service = next(iter(loaded))

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
        "active_model": active_service,
        "comfyui_running": comfyui_running,
        "vram_free_mb": gov_state.get("vram_free_mb"),
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

    # GPU services — route through Forge
    if app_meta.get("gpu"):
        try:
            forge = _get_governor()
            if not forge:
                return JSONResponse({"error": "Forge not available"}, status_code=503)
            result = await forge.preload.remote(service_name, body.get("model"))
            return JSONResponse({
                "status": result.get("status", "loaded"),
                "service": service_name,
                "url": app_meta.get("url"),
                "vram_free_mb": result.get("vram_free_mb"),
            })
        except Exception as e:
            logger.error("Studio: forge preload failed for %s: %s", service_name, e)
            return JSONResponse({"error": str(e)}, status_code=500)

    # CPU services or unmanaged
    return JSONResponse({
        "status": "loaded",
        "service": service_name,
        "url": app_meta.get("url"),
    })


async def studio_release(request: Request) -> JSONResponse:
    """POST /studio/api/release — unload everything from GPU."""
    try:
        forge = _get_governor()
        if forge:
            await forge.release.remote()
    except Exception as e:
        logger.warning("Studio: forge release failed: %s", e)

    return JSONResponse({"status": "released"})
