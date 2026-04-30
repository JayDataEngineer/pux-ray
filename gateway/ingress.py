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
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from registry.config import Config
from gateway import dashboard
from gateway import studio

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
        if request.url.path in ("/health", "/status") or request.url.path.startswith("/dashboard") or request.url.path.startswith("/studio"):
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
        """POST /v1/chat/completions - OpenAI-compatible chat.

        Proxies directly to llama-server, bypassing Ray Serve for zero
        overhead. The GPUScheduler handles model loading; the ingress
        handles routing.
        """
        self._ensure_initialized()
        body = await request.json()
        model = body.get("model", "qwen3.5-27b")
        stream = body.get("stream", False)

        if self.gpu_scheduler and not model.endswith("-cpu"):
            await self.gpu_scheduler.acquire_gpu.remote("llm", model)

        if stream:
            return await self._proxy_llm(body, stream=True)

        # Non-streaming: proxy to llama-server directly, return JSON
        return await self._proxy_llm(body, stream=False)

    async def _proxy_llm(self, body: dict, stream: bool = True) -> Response:
        """Proxy request to llama-server — single path for both modes."""
        import httpx as _httpx

        payload = {**body, "stream": stream}
        url = "http://127.0.0.1:8399/v1/chat/completions"

        if stream:
            async def _sse_generator():
                async with _httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                _sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming: single request, return JSON
        async with _httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            return JSONResponse(resp.json(), status_code=resp.status_code)

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

    # --- MCP Routes (direct proxy to persistent processes) ---

    async def _proxy_mcp(self, port: int, prefix: str, request: Request) -> Response:
        """Proxy to an MCP server running as a persistent process."""
        import httpx as _httpx

        path = request.url.path.replace(prefix, "") or "/"
        url = f"http://127.0.0.1:{port}{path}"
        if request.query_params:
            url += f"?{request.query_params}"

        async with _httpx.AsyncClient(timeout=120) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=await request.body(),
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

    async def mcp_web_proxy(self, request: Request) -> Response:
        """Proxy to local-web-mcp (port 8327)."""
        return await self._proxy_mcp(8327, "/mcp/web", request)

    async def mcp_media_proxy(self, request: Request) -> Response:
        """Proxy to media-analysis-mcp (port 8101)."""
        return await self._proxy_mcp(8101, "/mcp/media", request)

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

    # --- Job Routes (queued, async generation) ---

    def _job_manager(self):
        try:
            return ray.get_actor("job_manager")
        except ValueError:
            return None

    async def job_submit(self, request: Request) -> Response:
        """POST /jobs/{type} — submit a generation job. Returns job_id immediately."""
        self._ensure_initialized()
        job_type = request.path_params.get("type", "")
        jm = self._job_manager()
        if jm is None:
            return JSONResponse({"error": "JobManager not available"}, status_code=503)

        body = await request.json()
        kwargs = dict(body)
        kwargs.pop("type", None)

        if job_type == "trellis":
            form = await request.form() if "image" not in kwargs else None
            if form:
                image = await form["image"].read()
                kwargs["image"] = image
            if "image" not in kwargs:
                return JSONResponse({"error": "image required"}, status_code=400)
            job_id = await jm.submit.remote("trellis", **kwargs)
        elif job_type == "anigen":
            form = await request.form() if "image" not in kwargs else None
            if form:
                image = await form["image"].read()
                kwargs["image"] = image
            if "image" not in kwargs:
                return JSONResponse({"error": "image required"}, status_code=400)
            job_id = await jm.submit.remote("anigen", **kwargs)
        elif job_type == "ace_step":
            task_type = kwargs.get("task_type", "text2music")
            audio_modes = ("cover", "repaint", "lego", "extract", "complete")
            # Audio-to-audio modes: accept multipart form with audio file
            if task_type in audio_modes:
                form = await request.form() if "audio" not in kwargs else None
                if form and "audio" in form:
                    kwargs["audio"] = await form["audio"].read()
                if "audio" not in kwargs:
                    return JSONResponse(
                        {"error": f"audio file required for task_type={task_type}"},
                        status_code=400,
                    )
            elif not kwargs.get("prompt"):
                return JSONResponse({"error": "prompt required"}, status_code=400)
            job_id = await jm.submit.remote("ace_step", **kwargs)
        elif job_type == "comfyui":
            if "workflow" not in kwargs:
                return JSONResponse({"error": "workflow required"}, status_code=400)
            job_id = await jm.submit.remote("comfyui", **kwargs)
        else:
            return JSONResponse({"error": f"unknown job type: {job_type}"}, status_code=400)

        return JSONResponse({"job_id": job_id, "type": job_type, "status": "queued"})

    async def job_status(self, request: Request) -> Response:
        """GET /jobs/{job_id} — get job status."""
        self._ensure_initialized()
        job_id = request.path_params["job_id"]
        jm = self._job_manager()
        if jm is None:
            return JSONResponse({"error": "JobManager not available"}, status_code=503)
        status = await jm.status.remote(job_id)
        return JSONResponse(status)

    async def job_result(self, request: Request) -> Response:
        """GET /jobs/{job_id}/result — get job result bytes."""
        self._ensure_initialized()
        job_id = request.path_params["job_id"]
        jm = self._job_manager()
        if jm is None:
            return JSONResponse({"error": "JobManager not available"}, status_code=503)

        try:
            result = await jm.result.remote(job_id)
        except Exception as e:
            return JSONResponse({"error": str(e), "job_id": job_id}, status_code=500)

        if result is None:
            return JSONResponse({"error": "job not found"}, status_code=404)

        # Binary results (bytes) returned directly
        if isinstance(result, bytes):
            return Response(content=result, media_type="application/octet-stream")

        # Dict results (AniGen: mesh + skeleton)
        if isinstance(result, dict):
            return JSONResponse({"status": "completed", "keys": list(result.keys())})

        return JSONResponse({"result": str(result)})

    async def job_list(self, request: Request) -> Response:
        """GET /jobs — list all jobs."""
        self._ensure_initialized()
        jm = self._job_manager()
        if jm is None:
            return JSONResponse({"error": "JobManager not available"}, status_code=503)
        jobs = await jm.list_jobs.remote()
        return JSONResponse(jobs)

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
    dashboard.start_collector()
    ingress = APIIngress()

    routes = [
        # Health & Status (public, no auth)
        Route("/health", ingress.health),
        Route("/status", ingress.status),
        # Dashboard (public, no auth)
        Route("/dashboard", dashboard.dashboard_page),
        Route("/dashboard/api/gpu", dashboard.dashboard_gpu_current),
        Route("/dashboard/api/gpu/history", dashboard.dashboard_gpu_history),
        Route("/dashboard/api/services", dashboard.dashboard_services),
        # Studio (public, no auth)
        Route("/studio", studio.studio_page),
        Route("/studio/api/apps", studio.studio_apps),
        Route("/studio/api/switch", studio.studio_switch, methods=["POST"]),
        Route("/studio/api/release", studio.studio_release, methods=["POST"]),
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
        # MCP (proxied to subprocess servers)
        Route("/mcp/web/{path:path}", ingress.mcp_web_proxy, methods=["GET", "POST", "DELETE", "PUT", "PATCH"]),
        Route("/mcp/media/{path:path}", ingress.mcp_media_proxy, methods=["GET", "POST", "DELETE", "PUT", "PATCH"]),
        # Jobs (queued generation)
        Route("/jobs", ingress.job_list, methods=["GET"]),
        Route("/jobs/{type:str}", ingress.job_submit, methods=["POST"]),
        Route("/jobs/{job_id:str}", ingress.job_status, methods=["GET"]),
        Route("/jobs/{job_id:str}/result", ingress.job_result, methods=["GET"]),
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
