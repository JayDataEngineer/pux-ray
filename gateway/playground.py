"""Playground — interactive web UI for all Ray Serve services.

Provides a unified TNAP-driven interface where users can browse services
by category, dynamically generate input forms from service metadata, send
TNAP requests, and see rendered results (audio, image, 3D, text).

Endpoints:
    GET  /playground              — serve the playground HTML page
    GET  /playground/api/services — all services with UI metadata
"""

from __future__ import annotations

import logging
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from gateway.dashboard import query_service_status
from services.registry import get_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playground metadata — built dynamically from SERVICE_REGISTRY
# ---------------------------------------------------------------------------

def _build_playground_meta() -> dict[str, dict]:
    """Build playground service metadata from SERVICE_REGISTRY."""
    from services.registry import SERVICE_REGISTRY, ParamSpec

    meta: dict[str, dict] = {}

    for name, entry in SERVICE_REGISTRY.items():
        inp = _param_to_playground_field(entry.params_schema or [])
        # Infer category override for special cases
        cat = entry.category.capitalize()
        if name == "ace_step":
            cat = "Music"  # not "Audio"

        meta[name] = {
            "category": cat,
            "gpu": entry.needs_gpu,
            "route": _infer_route(name),
            "format": "openai" if name == "llm" else "tnap",
            "input_fields": inp,
        }

    return meta


def _infer_route(name: str) -> str:
    """Infer TNAP/API route from service name."""
    if name == "llm":
        return "/v1/chat/completions"
    if name == "comfyui":
        return "/comfyui"
    return f"/v1/{name}/generate"


def _param_to_playground_field(params: list) -> list[dict]:
    """Convert ParamSpec list to playground input_fields format."""
    fields = []
    for p in params or []:
        ftype = _map_param_type(p.type)
        field = {
            "key": p.label.lower().replace(" ", "_"),
            "label": p.label,
            "type": ftype,
        }
        if p.required:
            field["required"] = True
        if p.default is not None:
            field["default"] = p.default
        if p.placeholder:
            field["placeholder"] = p.placeholder
        if p.description:
            field["help"] = p.description
        if p.options:
            field["options"] = p.options
        fields.append(field)
    return fields


_PARAM_TYPE_MAP = {
    "text": "text",
    "textarea": "textarea",
    "number": "number",
    "select": "select",
    "file": "file",
    "bool": "checkbox",
    "json": "textarea",
    "image": "image",
    "audio": "audio",
    "range": "range",
}


def _map_param_type(ptype: str) -> str:
    return _PARAM_TYPE_MAP.get(ptype, "text")


PLAYGROUND_META = _build_playground_meta()


# Endpoints

_PLAYGROUND_HTML = Path(__file__).parent / "playground.html"


async def playground_page(request: Request) -> HTMLResponse:
    """GET /playground — serve the playground UI."""
    return HTMLResponse(_PLAYGROUND_HTML.read_text())


async def playground_services(request: Request) -> JSONResponse:
    """GET /playground/api/services — all services with UI metadata."""
    deploy_status = {s["name"]: s for s in query_service_status()}

    services = []
    for name, meta in PLAYGROUND_META.items():
        entry = get_service(name)
        dep = deploy_status.get(entry.deployment if entry else name, {})
        services.append({
            "name": name,
            "label": entry.label if entry else name,
            "category": meta["category"],
            "gpu": meta["gpu"],
            "route": meta.get("route"),
            "format": meta.get("format", "tnap"),
            "input_fields": meta["input_fields"],
            "output_type": entry.output_type if entry else "unknown",
            "description": entry.description if entry else "",
            "status": dep.get("status", "UNKNOWN"),
            "running_replicas": dep.get("running_replicas", 0),
        })

    return JSONResponse(services)
