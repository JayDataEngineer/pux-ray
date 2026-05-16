"""API Ingress Ray Serve deployment — catch-all gateway at /.

Delegates to the Starlette app built by gateway.ingress.create_app().
Ray Serve matches most-specific route_prefix first, so /tts/kokoro,
/asr/whisper, /forge, /playground all bypass this and go to their
dedicated deployments. Only unmatched paths reach the ingress.

Routes handled:
  /health, /status, /v1/*, /comfyui/*, /admin/*, /dashboard/*, /studio/*
"""
from __future__ import annotations

from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from gateway.ingress import APIIngress, _get_api_key
from gateway.dashboard import (
    dashboard_page, dashboard_gpu_current, dashboard_gpu_history, dashboard_services,
)
from gateway.studio import studio_page, studio_apps, studio_switch, studio_release


@serve.deployment(
    name="api-ingress",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1},
)
class APIIngressDeployment:

    def __init__(self):
        self._ingress = APIIngress()
        self._api_key = _get_api_key()

    async def __call__(self, request: Request) -> Response:
        # Auth check (mirrors APIKeyMiddleware)
        if self._api_key:
            path = request.url.path
            if path not in ("/health", "/status"):
                key = (request.headers.get("x-api-key", "")
                       or request.query_params.get("api_key", ""))
                if key != self._api_key:
                    from starlette.responses import JSONResponse
                    return JSONResponse({"error": "unauthorized"}, status_code=401)

        path = request.url.path
        method = request.method

        # Health / status
        if path == "/health":
            return await self._ingress.health(request)
        if path == "/status":
            return await self._ingress.status(request)

        # Service discovery
        if path == "/v1/services" and method == "GET":
            return await self._ingress.list_services(request)
        if path.startswith("/v1/services/") and method == "GET":
            service = path.split("/v1/services/")[1].rstrip("/")
            request.path_params = {"service": service}
            return await self._ingress.service_info(request)

        # Generic TNAP generate
        if path.startswith("/v1/") and path.endswith("/generate") and method == "POST":
            service = path.split("/v1/")[1].rstrip("/generate").rstrip("/")
            request.path_params = {"service": service}
            return await self._ingress.tnap_generate(request)

        # OpenAI-compatible
        if path == "/v1/models" and method == "GET":
            return await self._ingress.list_models(request)
        if path == "/v1/chat/completions" and method == "POST":
            return await self._ingress.chat_completions(request)
        if path == "/v1/llm/configure" and method == "POST":
            return await self._ingress.llm_configure(request)
        if path == "/v1/audio/speech" and method == "POST":
            return await self._ingress.audio_speech(request)
        if path == "/v1/audio/transcriptions" and method == "POST":
            return await self._ingress.audio_transcriptions(request)

        # LLM proxy (auto-loads via Forge on first request)
        if path == "/llm" or path.startswith("/llm/"):
            return await self._ingress.llm_proxy(request)

        # ComfyUI proxy
        if path == "/comfyui" or path.startswith("/comfyui/"):
            return await self._ingress.comfyui_proxy(request)

        # Admin
        if path == "/admin/load" and method == "POST":
            return await self._ingress.load_model(request)
        if path == "/admin/unload" and method == "POST":
            return await self._ingress.unload_all(request)

        # Dashboard
        if path == "/dashboard":
            return await dashboard_page(request)
        if path == "/dashboard/api/gpu":
            return await dashboard_gpu_current(request)
        if path == "/dashboard/api/gpu/history":
            return await dashboard_gpu_history(request)
        if path == "/dashboard/api/services":
            return await dashboard_services(request)

        # Studio
        if path == "/studio":
            return await studio_page(request)
        if path == "/studio/api/apps":
            return await studio_apps(request)
        if path == "/studio/api/switch" and method == "POST":
            return await studio_switch(request)
        if path == "/studio/api/release" and method == "POST":
            return await studio_release(request)

        from starlette.responses import JSONResponse
        return JSONResponse({"error": "not found", "path": path}, status_code=404)
