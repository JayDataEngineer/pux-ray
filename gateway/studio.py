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
# Studio app registry — built dynamically from SERVICE_REGISTRY
# ---------------------------------------------------------------------------

# Special overrides for services that need extra UI metadata not in the registry.
# All other services get sensible defaults from their ServiceEntry.
_STUDIO_OVERRIDES: dict[str, dict] = {
    # Services with web UIs
    "comfyui": {"url": "/comfyui/", "has_ui": True, "manage_type": "subprocess"},
    "kimodo_demo": {"url": "/kimodo/", "has_ui": True, "manage_type": "subprocess"},
    # MCP servers (not in SERVICE_REGISTRY — deployed separately)
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


def _build_studio_apps() -> dict[str, dict]:
    """Build studio app registry from SERVICE_REGISTRY + overrides.

    Every registered service gets a studio entry with sensible defaults.
    Services with special UI metadata are overridden via _STUDIO_OVERRIDES.
    """
    from services.registry import SERVICE_REGISTRY

    apps: dict[str, dict] = {}

    for name, entry in SERVICE_REGISTRY.items():
        label = entry.label
        category = entry.category.capitalize() if entry.category else "Other"
        gpu = entry.needs_gpu
        mgr = entry.default_model

        # Infer manage_type from service properties
        if gpu:
            manage_type = "scheduler"
        else:
            manage_type = "none"

        app: dict = {
            "label": label,
            "url": None,
            "has_ui": False,
            "category": category,
            "gpu": gpu,
            "manage_type": manage_type,
        }
        if mgr:
            app["default_model"] = mgr

        # Apply overrides
        if name in _STUDIO_OVERRIDES:
            app.update(_STUDIO_OVERRIDES[name])

        apps[name] = app

    # Add extras not in SERVICE_REGISTRY (MCP servers deployed separately)
    for name, override in _STUDIO_OVERRIDES.items():
        if name not in apps:
            apps[name] = dict(override)

    return apps


STUDIO_APPS = _build_studio_apps()

# Ordered categories for sidebar grouping — derived from actual categories
_CATEGORY_PRIORITY = ["Image", "Llm", "3d", "Motion", "Audio", "Creative",
                      "Avatar", "Tts", "Asr", "Training", "Mcp"]
CATEGORY_ORDER = sorted(
    {a["category"] for a in STUDIO_APPS.values()},
    key=lambda c: _CATEGORY_PRIORITY.index(c) if c in _CATEGORY_PRIORITY else 99,
)


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
    from services.registry import SERVICE_REGISTRY

    html = _STUDIO_HTML.read_text()

    # Build dynamic endpoint map from SERVICE_REGISTRY
    endpoints = {}
    for name, entry in SERVICE_REGISTRY.items():
        if entry.category == "llm":
            endpoints[name] = [{"method": "POST", "path": "/v1/chat/completions"}]
        elif name == "comfyui":
            endpoints[name] = [{"method": "GET", "path": "/comfyui/"}]
        elif name == "kimodo_demo":
            endpoints[name] = [{"method": "GET", "path": "/kimodo/"}]
        elif entry.category == "asr":
            endpoints[name] = [{"method": "POST", "path": "/v1/audio/transcriptions"}]
        elif entry.category == "tts":
            endpoints[name] = [{"method": "POST", "path": "/v1/audio/speech"}]
        elif name == "ace_step":
            endpoints[name] = [{"method": "POST", "path": "/music/generate"}]
        elif name == "see_through":
            endpoints[name] = [{"method": "POST", "path": "/creative/decompose"}]
        else:
            endpoints[name] = [{"method": "POST", "path": "/v1/run"}]

    # Inject into HTML before the first <script>
    import json
    script_tag = f"<script>var __API_ENDPOINTS__ = {json.dumps(endpoints)};</script>"
    html = html.replace("</title>", f"</title>{script_tag}", 1)

    return HTMLResponse(html)


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
