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
    # ════════════════════════════════════════════════════════════════════════
    # Auto-generated from SERVICE_REGISTRY — one entry per registered service
    # with UI metadata added for the Studio frontend.
    # ════════════════════════════════════════════════════════════════════════
    "comfyui": {
        "label": "ComfyUI",
        "url": "/comfyui/",
        "has_ui": True,
        "category": "Image",
        "gpu": True,
        "manage_type": "subprocess",
    },
    "llm": {
        "label": "LLM (llama.cpp)",
        "url": None,
        "has_ui": False,
        "category": "LLM",
        "gpu": True,
        "manage_type": "scheduler",
        "default_model": "qwen3.6-27b-q5_k_s-32k",
    },
    # ── Image ─────────────────────────────────────────────────────────────
    "native": {
        "label": "Native Models",
        "url": None, "has_ui": False,
        "category": "Image", "gpu": True, "manage_type": "scheduler",
        "default_model": "z_image",
    },
    "z_image": {
        "label": "Z-Image Generation",
        "url": None, "has_ui": False,
        "category": "Image", "gpu": True, "manage_type": "scheduler",
        "default_model": "z_image",
    },
    "anima": {
        "label": "Anima (Anime)",
        "url": None, "has_ui": False,
        "category": "Image", "gpu": True, "manage_type": "scheduler",
        "default_model": "anima",
    },
    "nvidia_upscale": {
        "label": "GPU Upscale",
        "url": None, "has_ui": False,
        "category": "Image", "gpu": True, "manage_type": "scheduler",
        "default_model": "nvidia_upscale",
    },
    "dwpose": {
        "label": "DWPose Detection",
        "url": None, "has_ui": False,
        "category": "Image", "gpu": False, "manage_type": "none",
    },
    # ── 3D ────────────────────────────────────────────────────────────────
    "trellis": {
        "label": "TRELLIS.2 3D",
        "url": None, "has_ui": False,
        "category": "3D", "gpu": True, "manage_type": "scheduler",
        "default_model": "trellis",
    },
    "anigen": {
        "label": "AniGen 3D",
        "url": None, "has_ui": False,
        "category": "3D", "gpu": True, "manage_type": "scheduler",
        "default_model": "anigen",
    },
    "body_mesh": {
        "label": "BodyMesh Renderer",
        "url": None, "has_ui": False,
        "category": "3D", "gpu": False, "manage_type": "none",
    },
    # ── Motion ────────────────────────────────────────────────────────────
    "kimodo_demo": {
        "label": "Kimodo Motion",
        "url": "/kimodo/", "has_ui": True,
        "category": "Motion", "gpu": True, "manage_type": "subprocess",
    },
    "kimodo": {
        "label": "Kimodo (API)",
        "url": None, "has_ui": False,
        "category": "Motion", "gpu": True, "manage_type": "scheduler",
        "default_model": "kimodo",
    },
    "hy_motion": {
        "label": "HY-Motion",
        "url": None, "has_ui": False,
        "category": "Motion", "gpu": True, "manage_type": "scheduler",
        "default_model": "hy_motion",
    },
    "gemx": {
        "label": "GEM-X Pose",
        "url": None, "has_ui": False,
        "category": "Motion", "gpu": True, "manage_type": "scheduler",
        "default_model": "gemx",
    },
    # ── Audio / Music ──────────────────────────────────────────────────────────
    "ace_step": {
        "label": "ACE-Step Music",
        "url": None, "has_ui": False,
        "category": "Audio", "gpu": True, "manage_type": "scheduler",
        "default_model": "ace_step",
    },
    "moss_soundeffect": {
        "label": "MOSS SoundEffect",
        "url": None, "has_ui": False,
        "category": "Audio", "gpu": True, "manage_type": "scheduler",
        "default_model": "moss_soundeffect",
    },
    # ── Creative ───────────────────────────────────────────────────────────────
    "see_through": {
        "label": "See-Through",
        "url": None, "has_ui": False,
        "category": "Creative", "gpu": True, "manage_type": "scheduler",
        "default_model": "see_through",
    },
    "lance": {
        "label": "Lance Multimodal",
        "url": None, "has_ui": False,
        "category": "Creative", "gpu": True, "manage_type": "scheduler",
        "default_model": "lance",
    },
    # ── Avatar ────────────────────────────────────────────────────────────
    "avatar": {
        "label": "Avatar Pipeline",
        "url": None, "has_ui": False,
        "category": "Avatar", "gpu": True, "manage_type": "scheduler",
        "default_model": "avatar",
    },
    # ── TTS ─────────────────────────────────────────────────────────────────
    "moss_tts": {
        "label": "MOSS TTS (GPU)",
        "url": None, "has_ui": False,
        "category": "TTS", "gpu": True, "manage_type": "scheduler",
        "default_model": "moss_tts",
    },
    "moss_voicegenerator": {
        "label": "MOSS Voice Design (GPU)",
        "url": None, "has_ui": False,
        "category": "TTS", "gpu": True, "manage_type": "scheduler",
        "default_model": "moss_voicegenerator",
    },
    "index_tts": {
        "label": "IndexTTS (GPU)",
        "url": None, "has_ui": False,
        "category": "TTS", "gpu": True, "manage_type": "scheduler",
        "default_model": "index_tts",
    },
    "kokoro": {
        "label": "Kokoro TTS (CPU)",
        "url": None, "has_ui": False,
        "category": "TTS", "gpu": False, "manage_type": "none",
    },
    "espeak": {
        "label": "eSpeak TTS (CPU)",
        "url": None, "has_ui": False,
        "category": "TTS", "gpu": False, "manage_type": "none",
    },
    # ── ASR / Transcription ───────────────────────────────────────────────
    "vibevoice_asr": {
        "label": "VibeVoice ASR (GPU)",
        "url": None, "has_ui": False,
        "category": "ASR", "gpu": True, "manage_type": "scheduler",
        "default_model": "vibevoice_asr",
    },
    "faster_whisper": {
        "label": "Faster-Whisper (CPU)",
        "url": None, "has_ui": False,
        "category": "ASR", "gpu": False, "manage_type": "none",
    },
    # ── Training ──────────────────────────────────────────────────────────
    "kohya": {
        "label": "kohya_ss LoRA Trainer",
        "url": None, "has_ui": False,
        "category": "Training", "gpu": True, "manage_type": "scheduler",
        "default_model": "kohya",
    },
    # ── MCP Servers (always-on, CPU) ──────────────────────────────────────
    "local_web_mcp": {
        "label": "Local Web MCP",
        "url": "/mcp/web/", "has_ui": False,
        "category": "MCP", "gpu": False, "manage_type": "persistent",
    },
    "media_analysis_mcp": {
        "label": "Media Analysis MCP",
        "url": "/mcp/media/", "has_ui": False,
        "category": "MCP", "gpu": False, "manage_type": "persistent",
    },
}

# Ordered categories for sidebar grouping
CATEGORY_ORDER = ["Image", "LLM", "3D", "Motion", "Audio", "Creative", "Avatar", "TTS", "ASR", "Training", "MCP"]


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
