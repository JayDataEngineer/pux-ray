"""API Ingress — single entry point for all AI service requests.

All services route through the service registry via /v1/{service}/generate.
GPU scheduling is handled automatically based on each service's needs_gpu flag.

Auth: API key via X-API-Key header or api_key query param.
Set secrets.api_key in config/local.yaml or TECH_NOIR_API_KEY env var.
Empty/unset = no auth (dev mode).
"""
from __future__ import annotations

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
from gateway.playground import playground_page, playground_services
from gateway.studio import studio_page, studio_apps, studio_switch, studio_release
from registry.config import Config
from services.registry import SERVICE_REGISTRY, get_service, resolve_model

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
        if key != api_key:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)


class APIIngress:
    """Main API router. GPU coordination via GPUGovernor."""

    def __init__(self):
        self._governor: Any | None = None

    async def _ensure_governor(self):
        if self._governor is None:
            try:
                self._governor = ray.get_actor("gpu_governor")
            except Exception:
                pass

    async def _use_gpu(self, service: str):
        """Acquire GPU lease for a heavy service."""
        await self._ensure_governor()
        if self._governor:
            await self._governor.acquire.remote(service)

    # ── Generic TNAP route ─────────────────────────────────────────────────────

    async def tnap_generate(self, request: Request) -> Response:
        """Generic route: POST /v1/{service}/generate"""
        service_name = request.path_params["service"]
        entry = get_service(service_name)

        if entry is None:
            return JSONResponse(
                {"error": f"Unknown service: {service_name}. "
                          f"Available: {', '.join(sorted(SERVICE_REGISTRY.keys()))}"},
                status_code=404,
            )

        body = await request.json()
        model = body.get("input", {}).get("model") or body.get("model")

        if entry.needs_gpu:
            await self._use_gpu(service_name)

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(request)

    # ── OpenAI-compatible routes ───────────────────────────────────────────────

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "qwen3.6-27b-ud-q4_k_xl")

        entry = get_service("llm")
        if not model.endswith("-cpu") and entry:
            await self._use_gpu("llm")

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        result = await handle.remote(
            messages=body.get("messages", []),
            model=model,
            stream=body.get("stream", False),
            **{k: v for k, v in body.items()
               if k not in ("model", "messages", "stream")},
        )
        return JSONResponse(result)

    async def audio_speech(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "tts-01-kokoro")

        # Resolve model alias to service
        resolved = resolve_model(model)
        if resolved:
            service_key, entry = resolved
        else:
            service_key, entry = "kokoro", get_service("kokoro")

        if entry.needs_gpu:
            await self._use_gpu(service_key)

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(request)

    async def audio_transcriptions(self, request: Request) -> Response:
        form = await request.form()
        model = str(form.get("model", "whisper-1"))

        resolved = resolve_model(model)
        if resolved:
            service_key, entry = resolved
        else:
            service_key, entry = "faster_whisper", get_service("faster_whisper")

        if entry.needs_gpu:
            await self._use_gpu(service_key)

        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(request)

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

    async def service_info(self, request: Request) -> Response:
        """GET /v1/services/{service} — info about a specific service."""
        service_name = request.path_params["service"]
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

    # ── ComfyUI proxy ──────────────────────────────────────────────────────────

    async def comfyui_proxy(self, request: Request) -> Response:
        """Route to ComfyUI via Ray Serve deployment handle."""
        entry = get_service("comfyui")
        await self._use_gpu("comfyui")
        handle = serve.get_deployment_handle(entry.deployment, entry.app)
        return await handle.remote(request)

    # ── Status / Health / Admin ────────────────────────────────────────────────

    async def status(self, request: Request) -> Response:
        await self._ensure_governor()
        status = {}
        if self._governor:
            gpu_status = await self._governor.status.remote()
            status["gpu"] = gpu_status

        from services.base import gpu_memory_info, gpu_resources
        status["vram"] = gpu_memory_info()
        status["resources"] = gpu_resources()

        return JSONResponse(status)

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def load_model(self, request: Request) -> Response:
        await self._ensure_governor()
        body = await request.json()
        service = body["service"]
        if self._governor:
            await self._governor.acquire.remote(service)
            return JSONResponse({"status": "loaded", "service": service})
        return JSONResponse({"error": "governor not available"}, status_code=503)

    async def unload_all(self, request: Request) -> Response:
        await self._ensure_governor()
        if self._governor:
            state = await self._governor.status.remote()
            holder = state.get("holder")
            if holder:
                await self._governor.release.remote(holder)
            return JSONResponse({"status": "unloaded"})
        return JSONResponse({"error": "governor not available"}, status_code=503)


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
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
        Route("/v1/audio/speech", ingress.audio_speech, methods=["POST"]),
        Route("/v1/audio/transcriptions", ingress.audio_transcriptions, methods=["POST"]),
        # ComfyUI (proxy all paths — ComfyUI has its own routing)
        Route("/comfyui/{path:path}", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/comfyui", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
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
    ]

    middleware = []
    if _get_api_key():
        middleware.append(Middleware(APIKeyMiddleware))

    return Starlette(routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=18080)
