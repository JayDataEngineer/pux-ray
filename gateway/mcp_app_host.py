"""MCP App Host — serves widget HTML and proxies tool calls.

Embedded in the ingress so no separate server is needed.
Handles the MCP Apps protocol used by assistant-ui widgets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

MCP_APP_MIME = "text/html;profile=mcp-app"

_TEMPLATE_DIR = Path(__file__).parent / "mcp_apps"

APPS = {
    "ui://apps/generate": {"name": "Generate", "description": "Run any GPU generation service"},
    "ui://apps/tts": {"name": "TTS Speech", "description": "Text-to-speech with voice design"},
    "ui://apps/audio": {"name": "Audio Studio", "description": "Transcription, sound effects, music"},
    "ui://apps/video": {"name": "Video Director", "description": "VACE video generation via OMNI-vLLM"},
    "ui://apps/admin": {"name": "GPU Admin", "description": "GPU status, load/unload services"},
    "ui://apps/workflow": {"name": "Workflow Runner", "description": "Interactive pipeline runner"},
}

# ── TTS Engine catalog — built dynamically from SERVICE_REGISTRY ────────────

def _build_tts_engines() -> list[dict[str, Any]]:
    """Build TTS engine catalogue from SERVICE_REGISTRY (category='tts').

    Every service with category='tts' automatically appears — no hardcoded
    engine dicts to keep in sync.
    """
    from services.registry import SERVICE_REGISTRY

    engines = []
    for name, entry in SERVICE_REGISTRY.items():
        if entry.category != "tts":
            continue

        gpu_suffix = " (GPU)" if entry.needs_gpu else " (CPU)"
        label = entry.label + gpu_suffix

        params = []
        for p in (entry.params_schema or []):
            param: dict[str, Any] = {
                "name": p.label.lower().replace(" ", "_") if p.label else "unknown",
                "type": p.type,
                "label": p.label,
            }
            if p.required:
                param["required"] = True
            if p.default is not None:
                param["default"] = p.default
            if p.placeholder:
                param["placeholder"] = p.placeholder
            if p.description:
                param["description"] = p.description
            if p.options:
                param["options"] = p.options
            params.append(param)

        engines.append({
            "id": name,
            "label": label,
            "category": entry.category,
            "gpu": entry.needs_gpu,
            "service": name,
            "model": entry.default_model,
            "description": entry.description or f"{entry.label} TTS",
            "params": params,
        })

    return engines


# Only expose engines that tts_speak can dispatch (not voice-design tools).
_TTS_HANDLED = {"kokoro", "moss_tts", "espeak"}
TTS_ENGINES = [e for e in _build_tts_engines() if e["id"] in _TTS_HANDLED]

def _get_html(uri: str) -> str | None:
    """Load HTML template for a widget URI."""
    name = uri.replace("ui://apps/", "")
    path = _TEMPLATE_DIR / f"{name}.html"
    if path.exists():
        return path.read_text()
    return None


async def handle_mcp_host(request: Request) -> JSONResponse:
    """POST /mcp/wan2gp-studio/host — MCP Apps protocol handler."""
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {})

    # resources/list — list available widget apps
    if method == "resources/list":
        return JSONResponse({
            "resources": [
                {
                    "uri": uri,
                    "name": info["name"],
                    "description": info["description"],
                    "mimeType": MCP_APP_MIME,
                    "_meta": {"ui": {"prefersBorder": True}},
                }
                for uri, info in APPS.items()
            ],
        })

    # mcp-apps/read-resource — return widget HTML
    if method == "mcp-apps/read-resource":
        uri = params.get("uri", "")
        html = _get_html(uri)
        if html is None:
            return JSONResponse({"error": f"Resource not found: {uri}"}, status_code=404)
        return JSONResponse({
            "uri": uri,
            "mimeType": MCP_APP_MIME,
            "html": html,
            "meta": {"ui": {"prefersBorder": True}},
        })

    # tools/call — proxy MCP tool calls through the ingress dispatch
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        # ── tts_voices: list available TTS engines and their parameters ──────
        if tool_name == "tts_voices":
            return JSONResponse({"engines": TTS_ENGINES})

        # ── tts_speak: unified dynamic TTS — one endpoint, all engines ──────
        if tool_name == "tts_speak":
            engine_id = tool_args.get("engine", "kokoro")
            engine = next((e for e in TTS_ENGINES if e["id"] == engine_id), None)
            if engine is None:
                return JSONResponse(
                    {"status": "error", "error": f"Unknown TTS engine: {engine_id}. "
                     f"Available: {[e['id'] for e in TTS_ENGINES]}"},
                    status_code=400,
                )

            from gateway.ingress import APIIngress
            ingress = APIIngress()

            try:
                # Forward all params from the tool call to the service dispatch.
                # The engine service name matches engine_id (kokoro, moss_tts, espeak, …).
                payload: dict[str, Any] = {k: v for k, v in tool_args.items() if k != "engine"}
                if "model" not in payload and engine.get("model"):
                    payload["model"] = engine["model"]
                result = await ingress._dispatch_service(engine_id, payload)

                # Normalize result: extract audio_url for the widget
                audio_url = None
                if isinstance(result, dict):
                    audio_url = result.get("audio_url") or result.get("url")
                    # If raw base64 data, return as data URI
                    if not audio_url and result.get("data"):
                        fmt = result.get("format", "wav")
                        mime = f"audio/{fmt}" if fmt != "opus" else "audio/ogg"
                        audio_url = f"data:{mime};base64,{result['data']}"

                return JSONResponse({
                    "engine": engine_id,
                    "audio_url": audio_url,
                    "result": result,
                })

            except ValueError as e:
                return JSONResponse({"status": "error", "error": str(e)}, status_code=404)
            except Exception as e:
                logger.exception("tts_speak failed for engine=%s", engine_id)
                return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

        # The 'run' tool: { service, params, model? }
        if tool_name == "run":
            from gateway.ingress import APIIngress
            ingress = APIIngress()
            service = tool_args.get("service", "kokoro")
            tool_params = tool_args.get("params", {})
            try:
                result = await ingress._dispatch_service(service, tool_params)
                return JSONResponse({"result": result})
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=404)

        # ── list_services: return all registered services ──────────────────
        if tool_name == "list_services":
            from services.registry import SERVICE_REGISTRY
            services = []
            for name, entry in SERVICE_REGISTRY.items():
                services.append({
                    "name": name,
                    "label": entry.label,
                    "category": entry.category,
                    "needs_gpu": entry.needs_gpu,
                    "output_type": entry.output_type,
                    "description": entry.description,
                    "model_aliases": list(entry.model_aliases.keys()),
                })
            return JSONResponse({"services": services})

        # ── list_models: return models grouped by category ──────────────────
        if tool_name == "list_models":
            from services.registry import SERVICE_REGISTRY
            models = []
            for name, entry in SERVICE_REGISTRY.items():
                models.append({
                    "id": name,
                    "label": entry.label,
                    "category": entry.category,
                    "needs_gpu": entry.needs_gpu,
                    "output_type": entry.output_type,
                    "description": entry.description,
                })
            return JSONResponse({"models": models})

        # Generic tool — 404
        return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

    return JSONResponse({"error": f"Unknown method: {method}"}, status_code=400)
