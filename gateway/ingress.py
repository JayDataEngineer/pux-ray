"""API Ingress — single entry point for all AI service requests.

GPU services route through the Forge (VRAM-aware scheduler).
CPU services route directly to their Ray Serve deployments.

Auth: API key via X-API-Key header or api_key query param.
Set secrets.api_key in config/local.yaml or TECH_NOIR_API_KEY env var.
Empty/unset = no auth (dev mode).
"""
from __future__ import annotations

import base64
import hmac
import logging
from typing import Any

import ray
from ray import serve
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from gateway.dashboard import dashboard_page, dashboard_gpu_current, dashboard_gpu_history, dashboard_services
from gateway.pipeline import PipelineSpec, execute_pipeline
from gateway.playground import playground_page, playground_services
from gateway.poser import poser_presets, poser_preset_render
from gateway.studio import studio_page, studio_apps, studio_switch, studio_release
from registry.config import Config
from services.registry import SERVICE_REGISTRY, get_service, resolve_model
from services.forge import SERVICE_MAP as FORGE_SERVICES

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    return Config().get("secrets.api_key", "")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key check. Skips /health and /status."""

    async def dispatch(self, request: Request, call_next):
        api_key = _get_api_key()
        if not api_key:
            return await call_next(request)

        if request.url.path in ("/health", "/status"):
            return await call_next(request)

        key = request.headers.get("x-api-key", "") or request.query_params.get("api_key", "")
        if not hmac.compare_digest(key, api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)


def _get_forge():
    """Get the Forge deployment handle."""
    return serve.get_deployment_handle("forge", "forge")


def _get_wan2gp():
    """Get the Wan2GP deployment handle."""
    return serve.get_deployment_handle("wan2gp", "wan2gp")


def _is_forge_service(service_name: str) -> bool:
    """Check if a service is managed by the Forge (subprocess GPU services)."""
    return service_name in FORGE_SERVICES


def _model_name_for(service_name: str, entry) -> str:
    """Map a service name to the Wan2GP model name in the dynamic registry.

    Wan2GP registry keys use 'family/variant' format (e.g. 'moss/moss-soundeffect').
    Service entries may store just the short form. This function ensures the full
    key is used by prepending the family name derived from the service name.
    """
    default = entry.default_model
    if "/" in default:
        return default
    # Derive family from service_name convention: moss_soundeffect → moss
    family = service_name.split("_")[0]
    variant = default.replace("_", "-")
    return f"{family}/{variant}"


class APIIngress:
    """Main API router. GPU services go through the Forge."""

    # ── Service dispatch (shared by TNAP + pipeline) ───────────────────────────

    async def _dispatch_service(self, service_name: str, body: dict) -> dict:
        """Dispatch a request to the correct backend (Forge/Wan2GP/direct)."""
        entry = get_service(service_name)
        if entry is None:
            raise ValueError(
                f"Unknown service: {service_name}. "
                f"Available: {', '.join(sorted(SERVICE_REGISTRY.keys()))}"
            )

        # Route through Forge if the service or its deployment is forge-managed.
        # Individual model services (moss_soundeffect, ace_step, etc.) have
        # deployment="wan2gp" which is a forge service — route them through
        # the forge using the deployment name as the forge service key.
        forge_key = service_name if _is_forge_service(service_name) else (
            entry.deployment if _is_forge_service(entry.deployment) else None
        )
        if forge_key:
            forge = _get_forge()
            model_name = _model_name_for(service_name, entry)
            body.setdefault("model", model_name)
            model = body.get("model")
            logger.info("DISPATCH service=%s forge_key=%s model=%s body_model=%s",
                        service_name, forge_key, model, model_name)
            return await forge.invoke.remote(forge_key, body, model)

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(body)

    # ── Generic TNAP route ─────────────────────────────────────────────────────

    async def tnap_generate(self, request: Request, service_name: str = None) -> Response:
        """Generic route: POST /v1/{service}/generate"""
        if service_name is None:
            service_name = request.path_params.get("service", "")
        body = await request.json()
        try:
            result = await self._dispatch_service(service_name, body)
            return JSONResponse(result)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    # ── Pipeline execution ─────────────────────────────────────────────────────

    async def execute_pipeline(self, request: Request) -> Response:
        """POST /api/pipelines/execute — run a multi-step inference pipeline."""
        body = await request.json()
        try:
            spec = PipelineSpec.from_dict(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        events = await execute_pipeline(spec, self._dispatch_service)

        import json as _json
        sse_lines = []
        for event in events:
            if event.get("event") == "pipeline_completed":
                sse_lines.append(f"data: {_json.dumps(event)}\n\n")
                sse_lines.append("data: [DONE]\n\n")
            elif event.get("event") == "pipeline_error":
                sse_lines.append(f"data: {_json.dumps(event)}\n\n")
                sse_lines.append("data: [DONE]\n\n")
            else:
                sse_lines.append(f"data: {_json.dumps(event)}\n\n")

        return Response(
            content="".join(sse_lines),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── OpenAI-compatible routes ───────────────────────────────────────────────

    async def list_models(self, request: Request) -> Response:
        """GET /v1/models — OpenAI-compatible model list."""
        from registry.models import ModelRegistry
        registry = ModelRegistry()
        llm_models = registry.list_models("llm").get("llm", {})
        models = []
        for name, meta in llm_models.items():
            if "dflash-draft" in name:
                continue
            models.append({
                "id": name,
                "object": "model",
                "owned_by": "tech-noir",
            })
        return JSONResponse({"object": "list", "data": models})

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        forge = _get_forge()
        result = await forge.invoke.remote("llm", body)
        return JSONResponse(result)

    async def llm_configure(self, request: Request) -> Response:
        """POST /v1/llm/configure — set model, engine, startup flags, session defaults."""
        body = await request.json()
        body["action"] = "configure"
        forge = _get_forge()
        result = await forge.invoke.remote("llm", body)
        return JSONResponse(result)

    async def audio_speech(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "tts-01-kokoro")

        resolved = resolve_model(model)
        if resolved:
            service_key, entry = resolved
        else:
            service_key, entry = "kokoro", get_service("kokoro")

        if _is_forge_service(service_key):
            forge = _get_forge()
            result = await forge.invoke.remote(service_key, body)
            return JSONResponse(result)

        if entry.deployment == "wan2gp":
            body.setdefault("model", _model_name_for(service_key, entry))
            wan2gp = _get_wan2gp()
            result = await wan2gp.invoke.remote(body)
            return JSONResponse(result)

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(request)

    async def audio_transcriptions(self, request: Request) -> Response:
        form = await request.form()
        model_name_for_service = str(form.get("model", "whisper-1"))

        resolved = resolve_model(model_name_for_service)
        if resolved:
            service_key, entry = resolved
        else:
            service_key, entry = "faster_whisper", get_service("faster_whisper")

        # Convert form to dict for JSON passthrough
        body = {k: v for k, v in form.items()}
        body.setdefault("model", _model_name_for(service_key, entry))

        wan2gp = _get_wan2gp()
        result = await wan2gp.invoke.remote(body)
        return JSONResponse(result)

    # ── Service discovery ──────────────────────────────────────────────────────

    async def list_services(self, request: Request) -> Response:
        """GET /v1/services — list all registered services."""
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
        return JSONResponse(services)

    async def service_info(self, request: Request, service_name: str = None) -> Response:
        """GET /v1/services/{service} — info about a specific service."""
        if service_name is None:
            service_name = request.path_params.get("service", "")
        entry = get_service(service_name)
        if not entry:
            return JSONResponse({"error": f"Unknown service: {service_name}"}, status_code=404)
        return JSONResponse({
            "name": service_name,
            "label": entry.label,
            "category": entry.category,
            "needs_gpu": entry.needs_gpu,
            "default_model": entry.default_model,
            "output_type": entry.output_type,
            "model_aliases": entry.model_aliases,
            "description": entry.description,
        })

    # ── LLM proxy ──────────────────────────────────────────────────────────────

    async def llm_proxy(self, request: Request) -> Response:
        """Route to LLM via Forge — auto-loads on first request."""
        forge = _get_forge()
        payload: dict[str, Any] = {
            "method": request.method,
            "path": request.url.path.replace("/llm", "") or "/",
            "params": dict(request.query_params),
            "raw": True,
        }
        if request.method in ("POST", "PUT", "PATCH"):
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                try:
                    payload["body"] = await request.json()
                except Exception:
                    pass
        result = await forge.invoke.remote("llm", payload)
        if isinstance(result, dict) and result.get("raw_response"):
            return Response(
                content=base64.b64decode(result["body"]),
                status_code=result.get("status_code", 200),
                media_type=result.get("content_type", "text/html"),
            )
        return JSONResponse(result)

    # ── ComfyUI proxy ──────────────────────────────────────────────────────────

    async def comfyui_proxy(self, request: Request) -> Response:
        """Route to ComfyUI via Forge — auto-loads on first request."""
        forge = _get_forge()
        # Use raw path to preserve URL-encoded characters (e.g. %2F for
        # ComfyUI userdata API paths). Starlette decodes request.url.path,
        # which breaks ComfyUI's path-based file serving.
        raw_path = request.scope.get("raw_path", b"").decode("ascii", errors="replace")
        proxy_path = (raw_path or request.url.path).replace("/comfyui", "") or "/"
        payload: dict[str, Any] = {
            "method": request.method,
            "path": proxy_path,
            "params": dict(request.query_params),
            "raw": True,
        }
        if request.method == "POST":
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                try:
                    payload["body"] = await request.json()
                except Exception:
                    pass
        result = await forge.invoke.remote("comfyui", payload)
        if isinstance(result, dict) and result.get("raw_response"):
            return Response(
                content=base64.b64decode(result["body"]),
                status_code=result.get("status_code", 200),
                media_type=result.get("content_type", "text/html"),
            )
        return JSONResponse(result)

    # ── Status / Health / Admin ────────────────────────────────────────────────

    async def status(self, request: Request) -> Response:
        status = {}
        try:
            forge = _get_forge()
            status = await forge.status.remote()
        except Exception:
            pass

        from services.base import gpu_memory_info, gpu_resources
        status["vram"] = gpu_memory_info()
        status["resources"] = gpu_resources()

        return JSONResponse(status)

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def load_model(self, request: Request) -> Response:
        body = await request.json()
        service = body.get("service", "")
        if not service:
            return JSONResponse({"error": "service required"}, status_code=400)
        try:
            forge = _get_forge()
            result = await forge.preload.remote(service, body.get("model"))
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    async def unload_all(self, request: Request) -> Response:
        try:
            forge = _get_forge()
            result = await forge.release.remote()
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


def create_app() -> Starlette:
    """Create the Starlette app with all routes and auth middleware."""
    ingress = APIIngress()

    routes = [
        # Health / status
        Route("/health", ingress.health),
        Route("/status", ingress.status),
        # Service discovery
        Route("/v1/services", ingress.list_services),
        Route("/v1/services/{service}", ingress.service_info),
        # Generic TNAP generate (covers all services)
        Route("/v1/{service}/generate", ingress.tnap_generate, methods=["POST"]),
        # OpenAI-compatible endpoints (keep for standard clients)
        Route("/v1/models", ingress.list_models),
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
        Route("/v1/llm/configure", ingress.llm_configure, methods=["POST"]),
        Route("/v1/audio/speech", ingress.audio_speech, methods=["POST"]),
        Route("/v1/audio/transcriptions", ingress.audio_transcriptions, methods=["POST"]),
        # LLM proxy (auto-loads via Forge on first request)
        Route("/llm/{path:path}", ingress.llm_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/llm", ingress.llm_proxy, methods=["GET", "POST"]),
        # ComfyUI (proxy all paths — ComfyUI has its own routing)
        Route("/comfyui/{path:path}", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/comfyui", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        # Pipeline execution
        Route("/api/pipelines/execute", ingress.execute_pipeline, methods=["POST"]),
        # Admin
        Route("/admin/load", ingress.load_model, methods=["POST"]),
        Route("/admin/unload", ingress.unload_all, methods=["POST"]),
        # Dashboard (GPU metrics)
        Route("/dashboard", dashboard_page),
        Route("/dashboard/api/gpu", dashboard_gpu_current),
        Route("/dashboard/api/gpu/history", dashboard_gpu_history),
        Route("/dashboard/api/services", dashboard_services),
        # Studio (GPU switching)
        Route("/studio", studio_page),
        Route("/studio/api/apps", studio_apps),
        Route("/studio/api/switch", studio_switch, methods=["POST"]),
        Route("/studio/api/release", studio_release, methods=["POST"]),
        # Playground (interactive service UI)
        Route("/playground", playground_page),
        Route("/playground/api/services", playground_services),
        # Poser (pose presets + skeleton renderer)
        Route("/poser/presets", poser_presets),
        Route("/poser/presets/{name}/render", poser_preset_render),
    ]

    middleware = []
    if _get_api_key():
        middleware.append(Middleware(APIKeyMiddleware))

    return Starlette(routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=18080)
