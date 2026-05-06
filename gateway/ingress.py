"""API Ingress — single entry point for all AI service requests.

Routes requests through the GPUScheduler to serialize GPU access.
All GPU services route via Ray Serve deployment handles — no direct
port proxying needed since Ray manages container networking.

Auth: API key via X-API-Key header or api_key query param.
Set secrets.api_key in config/local.yaml or TECH_NOIR_API_KEY env var.
Empty/unset = no auth (dev mode).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
import ray
from ray import serve
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from registry.config import Config

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
    """Main API router. All GPU routes go through GPUScheduler."""

    def __init__(self):
        self.gpu_scheduler: Optional[Any] = None
        self._initialized = False

    def _ensure_scheduler(self):
        if self._initialized:
            return
        try:
            self.gpu_scheduler = ray.get_actor("gpu_scheduler")
            self._initialized = True
            logger.info("Ingress: GPU scheduler ready")
        except Exception as e:
            logger.warning("Ingress: GPU scheduler not available (%s)", e)

    async def _use_gpu(self, service: str, model: str | None = None):
        """Acquire GPU for a service. Blocks until GPU is freed and service loaded."""
        self._ensure_scheduler()
        if self.gpu_scheduler:
            await self.gpu_scheduler.acquire_gpu.remote(service, model)

    async def _proxy_to_port(self, request: Request, port: int, *, target_path: str = "") -> Response:
        """Proxy request to a local port (legacy — only for TRELLIS until migrated)."""
        async with httpx.AsyncClient(timeout=600) as client:
            path = target_path or request.url.path
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=f"http://127.0.0.1:{port}{path}",
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=body,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/octet-stream"),
            )

    # --- LLM Routes ---

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "qwen3.6-27b-ud-q4_k_xl")

        if not model.endswith("-cpu"):
            await self._use_gpu("llm", model)

        handle = serve.get_deployment_handle("llm", "llm")
        result = await handle.remote(
            messages=body.get("messages", []),
            model=model,
            stream=body.get("stream", False),
            **{k: v for k, v in body.items()
               if k not in ("model", "messages", "stream")},
        )
        return JSONResponse(result)

    # --- TTS Routes ---

    async def audio_speech(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "tts-01-kokoro")

        tts_services = {
            "tts-01-kokoro": ("kokoro_tts", "kokoro_tts"),
            "tts-01-espeak": ("espeak_tts", "espeak_tts"),
            "tts-01-index": ("index_tts", "index_tts"),
            "tts-01-qwen": ("qwen_tts", "qwen_tts"),
            "tts-01-vibevoice": ("vibevoice", "vibevoice"),
            "tts-01-gpt-sovits": ("gpt_sovits", "gpt_sovits"),
        }

        dep_name, app_name = tts_services.get(model, ("kokoro_tts", "kokoro_tts"))

        if dep_name in ("index_tts", "qwen_tts", "vibevoice", "gpt_sovits"):
            await self._use_gpu(dep_name)

        handle = serve.get_deployment_handle(dep_name, app_name)
        return await handle.remote(request)

    # --- ASR Routes ---

    async def audio_transcriptions(self, request: Request) -> Response:
        form = await request.form()
        model = str(form.get("model", "whisper-1"))

        asr_services = {
            "whisper-1": ("faster_whisper", "faster_whisper"),
            "qwen-asr": ("qwen_asr", "qwen_asr"),
            "vibevoice-asr": ("vibevoice_asr", "vibevoice_asr"),
        }

        dep_name, app_name = asr_services.get(model, ("faster_whisper", "faster_whisper"))

        if dep_name in ("qwen_asr", "vibevoice_asr"):
            await self._use_gpu(dep_name)

        handle = serve.get_deployment_handle(dep_name, app_name)
        return await handle.remote(request)

    # --- 3D Routes ---

    async def trellis_generate(self, request: Request) -> Response:
        await self._use_gpu("trellis")
        return await self._proxy_to_port(request, 18401, target_path="/generate")

    async def anigen_generate(self, request: Request) -> Response:
        await self._use_gpu("anigen")
        handle = serve.get_deployment_handle("anigen", "anigen")
        return await handle.remote(request)

    async def hymotion_generate(self, request: Request) -> Response:
        await self._use_gpu("hy_motion")
        handle = serve.get_deployment_handle("hy_motion", "hy_motion")
        return await handle.remote(request)

    # --- Music Routes ---

    async def music_generate(self, request: Request) -> Response:
        await self._use_gpu("ace_step")
        handle = serve.get_deployment_handle("ace_step", "ace_step")
        return await handle.remote(request)

    # --- Multimodal Routes ---

    async def multimodal_chat(self, request: Request) -> Response:
        await self._use_gpu("phi4mm")
        handle = serve.get_deployment_handle("phi4mm", "phi4mm")
        return await handle.remote(request)

    # --- Vision Routes ---

    async def vision_analyze(self, request: Request) -> Response:
        await self._use_gpu("florence2")
        handle = serve.get_deployment_handle("florence2", "florence2")
        return await handle.remote(request)

    # --- Audio Generation Routes ---

    async def audio_sfx_generate(self, request: Request) -> Response:
        await self._use_gpu("moss_soundeffect")
        handle = serve.get_deployment_handle("moss_soundeffect", "moss_soundeffect")
        return await handle.remote(request)

    async def audio_tangoflux_generate(self, request: Request) -> Response:
        await self._use_gpu("tangoflux")
        handle = serve.get_deployment_handle("tangoflux", "tangoflux")
        return await handle.remote(request)

    # --- Creative Routes ---

    async def decompose(self, request: Request) -> Response:
        await self._use_gpu("see_through")
        handle = serve.get_deployment_handle("see_through", "see_through")
        return await handle.remote(request)

    # --- ComfyUI proxy ---

    async def comfyui_proxy(self, request: Request) -> Response:
        """Route to ComfyUI via Ray Serve deployment handle."""
        await self._use_gpu("comfyui")
        handle = serve.get_deployment_handle("comfyui", "comfyui")
        return await handle.remote(request)

    # --- Status Routes ---

    async def status(self, request: Request) -> Response:
        self._ensure_scheduler()
        status = {}
        if self.gpu_scheduler:
            gpu_status = await self.gpu_scheduler.status.remote()
            status["gpu"] = gpu_status

        from services.base import gpu_memory_info, gpu_resources
        status["vram"] = gpu_memory_info()
        status["resources"] = gpu_resources()

        return JSONResponse(status)

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    # --- Admin Routes ---

    async def load_model(self, request: Request) -> Response:
        self._ensure_scheduler()
        body = await request.json()
        service = body["service"]
        model = body.get("model")
        if self.gpu_scheduler:
            await self.gpu_scheduler.acquire_gpu.remote(service, model)
            return JSONResponse({"status": "loaded", "service": service, "model": model})
        return JSONResponse({"error": "scheduler not available"}, status_code=503)

    async def unload_all(self, request: Request) -> Response:
        self._ensure_scheduler()
        if self.gpu_scheduler:
            await self.gpu_scheduler.release_gpu.remote()
            return JSONResponse({"status": "unloaded"})
        return JSONResponse({"error": "scheduler not available"}, status_code=503)


def create_app() -> Starlette:
    """Create the Starlette app with all routes and auth middleware."""
    ingress = APIIngress()

    routes = [
        Route("/health", ingress.health),
        Route("/status", ingress.status),
        # LLM (OpenAI-compatible)
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
        # TTS (OpenAI-compatible)
        Route("/v1/audio/speech", ingress.audio_speech, methods=["POST"]),
        # ASR (OpenAI-compatible)
        Route("/v1/audio/transcriptions", ingress.audio_transcriptions, methods=["POST"]),
        # ComfyUI (proxy all paths)
        Route("/comfyui/{path:path}", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/comfyui", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        # 3D
        Route("/3d/trellis", ingress.trellis_generate, methods=["POST"]),
        Route("/3d/anigen", ingress.anigen_generate, methods=["POST"]),
        Route("/3d/hy-motion", ingress.hymotion_generate, methods=["POST"]),
        # Music
        Route("/music/generate", ingress.music_generate, methods=["POST"]),
        # Multimodal
        Route("/multimodal/chat", ingress.multimodal_chat, methods=["POST"]),
        # Vision
        Route("/vision/florence2", ingress.vision_analyze, methods=["POST"]),
        # Audio generation
        Route("/audio/soundeffect", ingress.audio_sfx_generate, methods=["POST"]),
        Route("/audio/tangoflux", ingress.audio_tangoflux_generate, methods=["POST"]),
        # Creative
        Route("/creative/decompose", ingress.decompose, methods=["POST"]),
        # Admin
        Route("/admin/load", ingress.load_model, methods=["POST"]),
        Route("/admin/unload", ingress.unload_all, methods=["POST"]),
    ]

    middleware = []
    if _get_api_key():
        middleware.append(Middleware(APIKeyMiddleware))

    return Starlette(routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=18080)
