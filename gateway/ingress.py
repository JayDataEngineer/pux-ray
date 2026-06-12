"""API Ingress — single entry point for all AI service requests.

GPU services route through the Forge (VRAM-aware scheduler).
CPU services route directly to their Ray Serve deployments.

Auth: API key via X-API-Key header or api_key query param.
Set secrets.api_key in config/local.yaml or TECH_NOIR_API_KEY env var.
Empty/unset = no auth (dev mode).
"""
from __future__ import annotations

import asyncio
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
from starlette.routing import Route, WebSocketRoute

from gateway.dashboard import dashboard_page, dashboard_gpu_current, dashboard_gpu_history, dashboard_services
from gateway.pipeline import PipelineSpec, execute_pipeline
from gateway.playground import playground_page, playground_services
from gateway.poser import poser_presets, poser_preset_render
from gateway.routes.editor import editor_page, editor_static, lora_list
from gateway.routes.workflows import (
    list_workflows, get_workflow, execute_workflow,
)
from gateway.routes.wf_engine import (
    wf_list_specs, wf_get_spec, wf_start_run, wf_get_run, wf_cancel_run,
    wf_approve_step, wf_rerun_step, wf_execute_step, wf_get_artifact, wf_list_artifacts, wf_events,
)
from gateway.studio import studio_page, studio_apps, studio_switch, studio_release
from gateway.mcp_app_host import handle_mcp_host
from registry.config import Config
from services.registry import SERVICE_REGISTRY, get_service, resolve_model, list_all_models
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


def _is_forge_service(service_name: str) -> bool:
    """Check if a service is managed by the Forge (subprocess GPU services)."""
    return service_name in FORGE_SERVICES


def _model_name_for(service_name: str, entry) -> str:
    """Map a service name to the Wan2GP model name."""
    return entry.default_model


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
        """GET /v1/models — unified model list across all service types."""
        category = request.query_params.get("category")
        models_raw = list_all_models(category)
        models = []
        for m in models_raw:
            models.append({
                "id": m["id"],
                "object": "model",
                "owned_by": "tech-noir",
                "category": m["category"],
                "label": m["label"],
                "output_type": m["output_type"],
                "needs_gpu": m["needs_gpu"],
                "description": m["description"],
            })
        return JSONResponse({"object": "list", "data": models})

    async def model_info(self, request: Request) -> Response:
        """GET /v1/models/{model} — info about a specific model."""
        model_id = request.path_params.get("model", "")
        entry = get_service(model_id)
        if not entry:
            # Try resolving by alias
            resolved = resolve_model(model_id)
            if resolved:
                _, entry = resolved
        if not entry:
            return JSONResponse({"error": f"Model '{model_id}' not found"}, status_code=404)
        return JSONResponse({
            "id": model_id,
            "object": "model",
            "owned_by": "tech-noir",
            "category": entry.category,
            "label": entry.label,
            "output_type": entry.output_type,
            "needs_gpu": entry.needs_gpu,
            "default_model": entry.default_model,
            "description": entry.description,
            "model_aliases": list(entry.model_aliases.keys()),
        })

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

    async def images_generations(self, request: Request) -> Response:
        """POST /v1/images/generations — OpenAI-compatible image generation.

        Supports model names from the service registry (e.g. "z_image", "flux_schnell")
        or aliases. Falls back to z_image (Flux/Z-Image via Wan2GP) if model is unknown.
        """
        body = await request.json()
        prompt = body.get("prompt", "")
        model_alias = body.get("model", "z_image")
        size = body.get("size", "1024x1024")
        n = body.get("n", 1)
        response_format = body.get("response_format", "b64_json")

        # Parse size into width/height
        width, height = 1024, 1024
        if isinstance(size, str) and "x" in size:
            try:
                width, height = [int(x) for x in size.split("x")]
            except ValueError:
                pass

        # Resolve model name → service
        resolved = resolve_model(model_alias)
        if resolved:
            service_key, entry = resolved
        else:
            # Try matching by category="image" services
            image_services = [
                (k, e) for k, e in SERVICE_REGISTRY.items()
                if e.category == "image" and e.output_type == "image"
            ]
            if image_services:
                service_key, entry = image_services[0]
            else:
                service_key, entry = "z_image", get_service("z_image")

        # Build dispatch payload
        dispatch_body = {
            "model": entry.default_model,
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        if body.get("negative_prompt"):
            dispatch_body["negative_prompt"] = body["negative_prompt"]
        if body.get("steps"):
            dispatch_body["steps"] = body["steps"]
        if body.get("seed"):
            dispatch_body["seed"] = body["seed"]

        result = await self._dispatch_service(service_key, dispatch_body)

        # Format as OpenAI images/generations response
        images = []
        image_data = result.get("data") or result.get("image") or result.get("url")
        if image_data:
            if isinstance(image_data, list):
                for item in image_data[:n]:
                    images.append({
                        "b64_json": item if isinstance(item, str) else None,
                        "url": item if isinstance(item, str) and item.startswith("http") else None,
                    })
            elif isinstance(image_data, str):
                # Single image — could be base64 or URL
                is_url = image_data.startswith("http")
                for _ in range(n):
                    images.append({
                        "b64_json": image_data if not is_url else None,
                        "url": image_data if is_url else None,
                    })

        if not images:
            # Fallback: return the raw result wrapped
            images = [{"b64_json": None, "url": None}]

        return JSONResponse({
            "created": int(__import__("time").time()),
            "data": images,
        })

    async def audio_speech(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "tts-01-kokoro")
        response_format = body.get("response_format", "wav")

        resolved = resolve_model(model)
        if resolved:
            service_key, entry = resolved
        else:
            service_key, entry = "kokoro", get_service("kokoro")

        if _is_forge_service(service_key):
            forge = _get_forge()
            result = await forge.invoke.remote(service_key, body)
        elif entry.deployment == "wan2gp":
            body.setdefault("model", _model_name_for(service_key, entry))
            forge = _get_forge()
            result = await forge.invoke.remote("wan2gp", body)
        else:
            handle = serve.get_deployment_handle(entry.deployment, entry.app)
            return await handle.remote(request)

        # OpenAI-compatible: return raw binary audio, not JSON+base64
        if isinstance(result, dict) and result.get("data"):
            audio_bytes = base64.b64decode(result["data"])
            content_types = {
                "wav": "audio/wav", "mp3": "audio/mpeg",
                "opus": "audio/opus", "flac": "audio/flac",
                "aac": "audio/aac", "pcm": "audio/pcm",
            }
            ct = content_types.get(response_format, "audio/wav")
            return Response(content=audio_bytes, media_type=ct)

        return JSONResponse(result)

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

        forge = _get_forge()
        result = await forge.invoke.remote("wan2gp", body)
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
                "params_schema": [
                    {
                        "name": p.label.lower().replace(" ", "_"),
                        "type": p.type,
                        "label": p.label,
                        "required": p.required,
                        "default": p.default,
                        "placeholder": p.placeholder,
                        "description": p.description,
                        "options": p.options,
                    }
                    for p in entry.params_schema
                ],
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
            "params_schema": [
                {
                    "name": p.label.lower().replace(" ", "_"),
                    "type": p.type,
                    "label": p.label,
                    "required": p.required,
                    "default": p.default,
                    "placeholder": p.placeholder,
                    "description": p.description,
                    "options": p.options,
                }
                for p in entry.params_schema
            ],
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

    # ── Kimodo proxy ───────────────────────────────────────────────────────────

    async def kimodo_proxy(self, request: Request) -> Response:
        """Route to Kimodo Viser demo via Forge — auto-loads on first request."""
        path = request.url.path
        if path == "/kimodo" and request.method == "GET":
            from starlette.responses import RedirectResponse
            return RedirectResponse(url="/kimodo/", status_code=307)

        forge = _get_forge()
        raw_path = request.scope.get("raw_path", b"").decode("ascii", errors="replace")
        proxy_path = (raw_path or request.url.path).replace("/kimodo", "") or "/"
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
        result = await forge.invoke.remote("kimodo_demo", payload)
        if isinstance(result, dict) and result.get("raw_response"):
            return Response(
                content=base64.b64decode(result["body"]),
                status_code=result.get("status_code", 200),
                media_type=result.get("content_type", "text/html"),
            )
        return JSONResponse(result)

    async def kimodo_ws_proxy(self, ws):
        """Proxy WebSocket to Kimodo Viser subprocess at 127.0.0.1:18470.

        Viser's client computes the WS URL from the page URL:
        page /kimodo/ → ws://host/kimodo
        We strip the prefix and proxy to the Viser server root.
        """
        import websockets
        from starlette.websockets import WebSocketDisconnect

        await ws.accept()
        path = ws.url.path.replace("/kimodo", "") or "/"
        query = ws.url.query
        target = f"ws://127.0.0.1:18470{path}"
        if query:
            target += f"?{query}"

        logger.info("Kimodo WS proxy: %s -> %s", ws.url.path, target)

        try:
            async with websockets.connect(target) as backend:
                async def to_backend():
                    try:
                        while True:
                            msg = await ws.receive()
                            t = msg.get("type", "")
                            if t == "websocket.disconnect":
                                break
                            if t == "websocket.receive":
                                txt = msg.get("text")
                                if txt is not None:
                                    await backend.send(txt)
                                bts = msg.get("bytes")
                                if bts is not None:
                                    await backend.send(bts)
                    except WebSocketDisconnect:
                        pass

                async def to_client():
                    try:
                        async for msg in backend:
                            if isinstance(msg, str):
                                await ws.send_text(msg)
                            elif isinstance(msg, bytes):
                                await ws.send_bytes(msg)
                    except websockets.exceptions.ConnectionClosed:
                        pass

                await asyncio.gather(to_backend(), to_client())
        except (
            websockets.WebSocketException,
            ConnectionRefusedError,
            OSError,
        ) as e:
            logger.warning("Kimodo WS proxy failed: %s", e)
            try:
                await ws.close(1011)
            except Exception:
                pass

    # ── ComfyUI proxy ──────────────────────────────────────────────────────────

    async def comfyui_proxy(self, request: Request) -> Response:
        """Route to ComfyUI via Forge — auto-loads on first request."""
        # Redirect /comfyui → /comfyui/ so relative asset paths resolve correctly
        path = request.url.path
        if path == "/comfyui" and request.method == "GET":
            from starlette.responses import RedirectResponse
            return RedirectResponse(url="/comfyui/", status_code=307)

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

    async def comfyui_ws_proxy(self, ws):
        """Proxy WebSocket to ComfyUI subprocess at 127.0.0.1:18465.

        Connects directly to the ComfyUI subprocess (bypasses Forge proxy)
        so the frontend gets real-time progress via /ws.
        """
        import websockets
        from starlette.websockets import WebSocketDisconnect

        await ws.accept()
        path = ws.url.path.replace("/comfyui", "") or "/"
        query = ws.url.query
        target = f"ws://127.0.0.1:18465{path}"
        if query:
            target += f"?{query}"

        logger.info("ComfyUI WS proxy: %s -> %s", ws.url.path, target)

        try:
            async with websockets.connect(target) as backend:
                async def to_backend():
                    try:
                        while True:
                            msg = await ws.receive()
                            t = msg.get("type", "")
                            if t == "websocket.disconnect":
                                break
                            if t == "websocket.receive":
                                txt = msg.get("text")
                                if txt is not None:
                                    await backend.send(txt)
                                bts = msg.get("bytes")
                                if bts is not None:
                                    await backend.send(bts)
                    except WebSocketDisconnect:
                        pass

                async def to_client():
                    try:
                        async for msg in backend:
                            if isinstance(msg, str):
                                await ws.send_text(msg)
                            elif isinstance(msg, bytes):
                                await ws.send_bytes(msg)
                    except websockets.exceptions.ConnectionClosed:
                        pass

                await asyncio.gather(to_backend(), to_client())
        except (
            websockets.WebSocketException,
            ConnectionRefusedError,
            OSError,
        ) as e:
            logger.warning("ComfyUI WS proxy failed: %s", e)
            try:
                await ws.close(1011)
            except Exception:
                pass

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

    # ── Unified /v1/run endpoint ──────────────────────────────────────────────

    async def run_unified(self, request: Request) -> Response:
        """POST /v1/run — unified interface for single-service and pipeline calls.

        Three payload shapes:
          Single service:  {"service": "wan2gp", "model": "z_image", "params": {...}}
          Named pipeline:  {"pipeline": "tech-noir/generate", "params": {"prompt": "..."}}
          Inline steps:    {"steps": [{"name": "gen", "service": "wan2gp", "params": {...}}]}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        # Named pipeline — route through Forge's VRAM ledger
        if "pipeline" in body:
            pipeline_id = body["pipeline"]
            # Support both "params" (legacy Python) and "inputs" (YAML spec)
            params = body.get("params", body.get("inputs", {}))
            try:
                forge = _get_forge()
                result = await forge.run_pipeline.remote(pipeline_id, params)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=503)
            if isinstance(result, dict) and result.get("status") == "error":
                return JSONResponse(result, status_code=500)
            return JSONResponse(result)

        # Inline pipeline steps — reuse gateway/pipeline.py with Forge dispatch
        if "steps" in body and isinstance(body["steps"], list):
            try:
                spec = PipelineSpec.from_dict(body)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            events = await execute_pipeline(spec, self._dispatch_service)
            import json as _json
            sse_lines = []
            for event in events:
                if event.get("event") in ("pipeline_completed", "pipeline_error"):
                    sse_lines.append(f"data: {_json.dumps(event)}\n\n")
                    sse_lines.append("data: [DONE]\n\n")
                else:
                    sse_lines.append(f"data: {_json.dumps(event)}\n\n")
            return Response(
                content="".join(sse_lines),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Single service invocation — same as /forge
        service = body.get("service")
        if not service:
            return JSONResponse(
                {"error": "Must specify 'service', 'pipeline', or 'steps'"},
                status_code=400,
            )
        try:
            result = await self._dispatch_service(service, body)
            return JSONResponse(result)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def run_catalog(self, request: Request) -> Response:
        """GET /v1/run/catalog — discover available pipelines and services."""
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        pipelines = [
            {"id": wf_id, "description": fn.__doc__ or ""}
            for wf_id, fn in _WORKFLOW_REGISTRY.items()
        ]
        services = [
            {"name": name, "label": entry.label, "category": entry.category}
            for name, entry in SERVICE_REGISTRY.items()
        ]
        return JSONResponse({"pipelines": pipelines, "services": services})


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
        # OpenAI-compatible endpoints (exact paths before catch-all)
        Route("/v1/models", ingress.list_models),
        Route("/v1/models/{model}", ingress.model_info),
        Route("/v1/chat/completions", ingress.chat_completions, methods=["POST"]),
        Route("/v1/images/generations", ingress.images_generations, methods=["POST"]),
        Route("/v1/llm/configure", ingress.llm_configure, methods=["POST"]),
        Route("/v1/audio/speech", ingress.audio_speech, methods=["POST"]),
        Route("/v1/audio/transcriptions", ingress.audio_transcriptions, methods=["POST"]),
        # Generic TNAP generate (covers all services — must be after exact paths)
        Route("/v1/{service}/generate", ingress.tnap_generate, methods=["POST"]),
        # LLM proxy (auto-loads via Forge on first request)
        Route("/llm/{path:path}", ingress.llm_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/llm", ingress.llm_proxy, methods=["GET", "POST"]),
        # Kimodo (proxy Viser 3D motion UI)
        Route("/kimodo/{path:path}", ingress.kimodo_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/kimodo", ingress.kimodo_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        WebSocketRoute("/kimodo", ingress.kimodo_ws_proxy),
        # ComfyUI (proxy all paths — ComfyUI has its own routing)
        Route("/comfyui/{path:path}", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        Route("/comfyui", ingress.comfyui_proxy, methods=["GET", "POST", "PUT", "DELETE"]),
        WebSocketRoute("/comfyui/ws", ingress.comfyui_ws_proxy),
        # Pipeline execution
        Route("/api/pipelines/execute", ingress.execute_pipeline, methods=["POST"]),
        # Admin
        Route("/admin/load", ingress.load_model, methods=["POST"]),
        Route("/admin/unload", ingress.unload_all, methods=["POST"]),
        # Unified endpoint — single service, named pipeline, or inline steps
        Route("/v1/run", ingress.run_unified, methods=["POST"]),
        Route("/v1/run/catalog", ingress.run_catalog),
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
        # MCP App Host (widget HTML + tool proxy)
        Route("/mcp/wan2gp-studio/host", handle_mcp_host, methods=["POST"]),
        # LoRA listing (for editor LoRA picker)
        Route("/v1/loras", lora_list),
        # Video Editor (React SPA)
        Route("/editor", editor_page),
        Route("/editor/{path:path}", editor_static),
        # Playground (interactive service UI)
        Route("/playground", playground_page),
        Route("/playground/api/services", playground_services),
        # Poser (pose presets + skeleton renderer)
        Route("/poser/presets", poser_presets),
        Route("/poser/presets/{name}/render", poser_preset_render),
        # Workflows — legacy (hardcoded Python functions via Forge)
        Route("/v1/workflows", list_workflows),
        Route("/v1/workflows/{workflow}", get_workflow),
        Route("/v1/workflows/{workflow}", execute_workflow, methods=["POST"]),
        # Workflows — new engine (YAML-based, declarative)
        Route("/v1/wf", wf_list_specs),
        Route("/v1/wf/{spec_name}", wf_get_spec),
        Route("/v1/wf/{spec_name}/runs", wf_start_run, methods=["POST"]),
        Route("/v1/wf/{spec_name}/runs/{run_id}", wf_get_run),
        Route("/v1/wf/{spec_name}/runs/{run_id}", wf_cancel_run, methods=["DELETE"]),
        Route("/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/approve", wf_approve_step, methods=["POST"]),
        Route("/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/rerun", wf_rerun_step, methods=["POST"]),
        Route("/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/execute", wf_execute_step, methods=["POST"]),
        Route("/v1/wf/{spec_name}/runs/{run_id}/artifacts", wf_list_artifacts),
        Route("/v1/wf/{spec_name}/runs/{run_id}/artifacts/{step_id}/{filename}", wf_get_artifact),
        Route("/v1/wf/{spec_name}/runs/{run_id}/events", wf_events),
    ]

    middleware = []
    if _get_api_key():
        middleware.append(Middleware(APIKeyMiddleware))

    return Starlette(routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=30080)
