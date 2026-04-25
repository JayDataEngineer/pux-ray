"""API Ingress - single entry point for all AI service requests.

Routes requests to the appropriate Ray Serve deployment,
handling GPU model swaps via the GPUScheduler.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

import ray
from ray import serve
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class APIIngress:
    """Main API router. Composes all deployment handles."""

    def __init__(self):
        self.gpu_scheduler: Optional[Any] = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy init - get handles after deployments are live."""
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

        # Ensure GPU is loaded for this model
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

    # --- 3D Routes ---

    async def trellis_generate(self, request: Request) -> Response:
        """POST /3d/trellis - Image to 3D mesh via TRELLIS.2."""
        handle = serve.get_deployment_handle("trellis", "creative")
        return await handle.remote(request)

    async def anigen_generate(self, request: Request) -> Response:
        """POST /3d/anigen - Image to rigged 3D via AniGen."""
        handle = serve.get_deployment_handle("anigen", "creative")
        return await handle.remote(request)

    # --- Music Routes ---

    async def music_generate(self, request: Request) -> Response:
        """POST /music/generate - Text to music via ACE-STEP."""
        handle = serve.get_deployment_handle("ace_step", "creative")
        return await handle.remote(request)

    # --- Creative Routes ---

    async def decompose(self, request: Request) -> Response:
        """POST /creative/decompose - Image to layers via See-Through."""
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
    """Create the Starlette app with all routes."""
    ingress = APIIngress()

    routes = [
        # Health & Status
        Route("/health", ingress.health),
        Route("/status", ingress.status),
        # LLM
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
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

    return Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
