"""API Ingress - single entry point for all AI service requests.

Routes requests to the appropriate Ray Serve deployment,
handling GPU model swaps via the GPUScheduler.

Auth: API key via X-API-Key header or api_key query param.
Set secrets.api_key in config/local.yaml or TECH_NOIR_API_KEY env var.
Empty/unset = no auth (dev mode).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

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
    """Read API key from config (supports env var interpolation)."""
    return Config().get("secrets.api_key", "")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key check. Skips /health and /status."""

    async def dispatch(self, request: Request, call_next):
        api_key = _get_api_key()
        if not api_key:
            return await call_next(request)

        # Skip public endpoints
        if request.url.path in ("/health", "/status"):
            return await call_next(request)

        key = request.headers.get("x-api-key", "") or request.query_params.get("api_key", "")
        if key != api_key:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)


class APIIngress:
    """Main API router. Composes all deployment handles."""

    def __init__(self):
        self.gpu_scheduler: Optional[Any] = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            self.gpu_scheduler = ray.get_actor("gpu_scheduler")
            self._initialized = True
            logger.info("Ingress initialized with GPU scheduler")
        except Exception as e:
            logger.warning("Ingress init deferred: %s", e)

    # --- LLM Routes ---

    async def chat_completions(self, request: Request) -> Response:
        """POST /v1/chat/completions - OpenAI-compatible chat."""
        self._ensure_initialized()
        body = await request.json()
        model = body.get("model", "qwen3.5-27b")

        if self.gpu_scheduler and not model.endswith("-cpu"):
            await self.gpu_scheduler.acquire_gpu.remote("llm", model)

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
        """POST /v1/audio/speech - OpenAI-compatible TTS.

        Routes to the appropriate TTS service based on the 'model' field:
          - tts-01-kokoro (default), tts-01-espeak, tts-01-index,
            tts-01-qwen, tts-01-vibevoice, tts-01-gpt-sovits
        """
        body = await request.json()
        model = body.get("model", "tts-01-kokoro")

        # Map model names to Ray Serve deployment names and app names
        tts_services = {
            "tts-01-kokoro": ("kokoro_tts", "kokoro_tts"),
            "tts-01-espeak": ("espeak_tts", "espeak_tts"),
            "tts-01-index": ("index_tts", "index_tts"),
            "tts-01-qwen": ("qwen_tts", "qwen_tts"),
            "tts-01-vibevoice": ("vibevoice", "vibevoice"),
            "tts-01-gpt-sovits": ("gpt_sovits", "gpt_sovits"),
        }

        dep_name, app_name = tts_services.get(model, ("kokoro_tts", "kokoro_tts"))
        handle = serve.get_deployment_handle(dep_name, app_name)
        return await handle.remote(request)

    # --- ASR Routes ---

    async def audio_transcriptions(self, request: Request) -> Response:
        """POST /v1/audio/transcriptions - OpenAI-compatible ASR.

        Routes to whisper by default, or qwen-asr / vibevoice-asr.
        """
        form = await request.form()
        model = str(form.get("model", "whisper-1"))

        asr_services = {
            "whisper-1": ("faster_whisper", "faster_whisper"),
            "qwen-asr": ("qwen_asr", "qwen_asr"),
            "vibevoice-asr": ("vibevoice_asr", "vibevoice_asr"),
        }

        dep_name, app_name = asr_services.get(model, ("faster_whisper", "faster_whisper"))
        handle = serve.get_deployment_handle(dep_name, app_name)
        return await handle.remote(request)

    # --- 3D Routes ---

    async def trellis_generate(self, request: Request) -> Response:
        """POST /3d/trellis - Image to 3D mesh."""
        handle = serve.get_deployment_handle("trellis", "creative")
        return await handle.remote(request)

    async def anigen_generate(self, request: Request) -> Response:
        """POST /3d/anigen - Image to rigged 3D."""
        handle = serve.get_deployment_handle("anigen", "creative")
        return await handle.remote(request)

    # --- Music Routes ---

    async def music_generate(self, request: Request) -> Response:
        """POST /music/generate - Text to music."""
        handle = serve.get_deployment_handle("ace_step", "creative")
        return await handle.remote(request)

    # --- Creative Routes ---

    async def decompose(self, request: Request) -> Response:
        """POST /creative/decompose - Image to layers."""
        handle = serve.get_deployment_handle("see_through", "creative")
        return await handle.remote(request)

    # --- Status Routes ---

    async def status(self, request: Request) -> Response:
        """GET /status - infrastructure overview."""
        self._ensure_initialized()
        status = {}
        if self.gpu_scheduler:
            gpu_status = await self.gpu_scheduler.status.remote()
            status["gpu"] = gpu_status

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = result.stdout.strip().split(",")
            status["vram"] = {
                "free_mb": int(parts[0].strip()),
                "used_mb": int(parts[1].strip()),
                "total_mb": int(parts[2].strip()),
            }
        except Exception:
            status["vram"] = {"error": "nvidia-smi unavailable"}

        return JSONResponse(status)

    async def health(self, request: Request) -> Response:
        """GET /health"""
        return JSONResponse({"status": "ok"})

    # --- Admin Routes ---

    async def load_model(self, request: Request) -> Response:
        """POST /admin/load - explicitly load a model."""
        self._ensure_initialized()
        body = await request.json()
        service = body["service"]
        model = body["model"]

        if self.gpu_scheduler:
            await self.gpu_scheduler.acquire_gpu.remote(service, model)
            return JSONResponse({"status": "loaded", "service": service, "model": model})
        return JSONResponse({"error": "scheduler not available"}, status_code=503)

    async def unload_all(self, request: Request) -> Response:
        """POST /admin/unload - release GPU."""
        self._ensure_initialized()
        if self.gpu_scheduler:
            await self.gpu_scheduler.release_gpu.remote()
            return JSONResponse({"status": "unloaded"})
        return JSONResponse({"error": "scheduler not available"}, status_code=503)


def create_app() -> Starlette:
    """Create the Starlette app with all routes and auth middleware."""
    ingress = APIIngress()

    routes = [
        # Health & Status (public, no auth)
        Route("/health", ingress.health),
        Route("/status", ingress.status),
        # LLM (OpenAI-compatible)
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
        # TTS (OpenAI-compatible)
        Route("/v1/audio/speech", ingress.audio_speech, methods=["POST"]),
        # ASR (OpenAI-compatible)
        Route("/v1/audio/transcriptions", ingress.audio_transcriptions, methods=["POST"]),
        # 3D
        Route("/3d/trellis", ingress.trellis_generate, methods=["POST"]),
        Route("/3d/anigen", ingress.anigen_generate, methods=["POST"]),
        # Music
        Route("/music/generate", ingress.music_generate, methods=["POST"]),
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
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
