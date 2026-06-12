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
from starlette.websockets import WebSocket as WS

from gateway.ingress import APIIngress, _get_api_key
from gateway.dashboard import (
    dashboard_page, dashboard_gpu_current, dashboard_gpu_history, dashboard_services,
)
from gateway.studio import studio_page, studio_apps, studio_switch, studio_release
from gateway.routes.workflows import list_workflows, get_workflow, execute_workflow, _get_workflow_json, _execute_workflow
from gateway.routes.wf_engine import (
    wf_list_specs, wf_get_spec, wf_start_run, wf_get_run, wf_cancel_run,
    wf_approve_step, wf_continue_step, wf_rerun_step, wf_execute_step, wf_list_artifacts,
    wf_get_artifact, wf_events,
)
from gateway.routes.editor import (
    editor_page, editor_static,
    llm_key_store, llm_key_list, llm_key_delete, llm_enhance, llm_chat
)


class _ParamsRequest:
    """Wraps a Starlette Request to inject path_params (read-only property workaround)."""
    def __init__(self, request: Request, params: dict):
        self._request = request
        self._params = params

    @property
    def path_params(self):
        return self._params

    def __getattr__(self, name):
        return getattr(self._request, name)

    async def body(self):
        return await self._request.body()

    async def json(self):
        return await self._request.json()

    async def stream(self):
        return self._request.stream()


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
        # WebSocket detection — check ASGI scope type
        scope_type = getattr(request, 'scope', {}).get('type', '')
        if scope_type == 'websocket':
            path = request.url.path
            if path == "/comfyui" or path.startswith("/comfyui/"):
                await self._ingress.comfyui_ws_proxy(request)
            elif path == "/kimodo" or path.startswith("/kimodo/"):
                await self._ingress.kimodo_ws_proxy(request)
            else:
                await request.close()
            return

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
            return await self._ingress.service_info(request, service_name=service)

        # Generic TNAP generate
        if path.startswith("/v1/") and path.endswith("/generate") and method == "POST":
            service = path[len("/v1/"):-len("/generate")]
            return await self._ingress.tnap_generate(request, service_name=service)

        # OpenAI-compatible
        if path == "/v1/models" and method == "GET":
            return await self._ingress.list_models(request)
        if path == "/v1/chat/completions" and method == "POST":
            return await self._ingress.chat_completions(request)
        if path == "/v1/llm/configure" and method == "POST":
            return await self._ingress.llm_configure(request)

        # LLM key management (secure storage)
        if path == "/v1/llm/keys" and method == "POST":
            return await llm_key_store(request)
        if path == "/v1/llm/keys" and method == "GET":
            return await llm_key_list(request)
        if path.startswith("/v1/llm/keys/") and method == "DELETE":
            key_id = path.split("/v1/llm/keys/")[1].rstrip("/")
            return await llm_key_delete(_ParamsRequest(request, {"key_id": key_id}))
        if path == "/v1/llm/enhance" and method == "POST":
            return await llm_enhance(request)
        if path == "/v1/llm/chat" and method == "POST":
            return await llm_chat(request)
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

        # Kimodo proxy (Viser 3D motion UI)
        if path == "/kimodo" or path.startswith("/kimodo/"):
            return await self._ingress.kimodo_proxy(request)

        # Pipeline execution
        if path == "/api/pipelines/execute" and method == "POST":
            return await self._ingress.execute_pipeline(request)

        # Admin
        if path == "/admin/load" and method == "POST":
            return await self._ingress.load_model(request)
        if path == "/admin/unload" and method == "POST":
            return await self._ingress.unload_all(request)

        # Unified endpoint
        if path == "/v1/run" and method == "POST":
            return await self._ingress.run_unified(request)
        if path == "/v1/run/catalog" and method == "GET":
            return await self._ingress.run_catalog(request)

        # Workflows (multi-model orchestration)
        if path == "/v1/workflows" and method == "GET":
            return await list_workflows(request)
        if path.startswith("/v1/workflows/") and method == "GET":
            wf_id = path[len("/v1/workflows/"):].rstrip("/")
            return _get_workflow_json(wf_id)
        if path.startswith("/v1/workflows/") and method == "POST":
            wf_id = path[len("/v1/workflows/"):].rstrip("/")
            body = await request.json()
            return await _execute_workflow(wf_id, body)

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

        # Editor SPA
        if path == "/editor":
            return await editor_page(request)
        if path.startswith("/editor/"):
            sub_path = path[len("/editor/"):]
            return await editor_static(_ParamsRequest(request, {"path": sub_path}))

        # Workflow Engine (YAML-based declarative workflows)
        if path == "/v1/wf" and method == "GET":
            return await wf_list_specs(request)
        if path.startswith("/v1/wf/"):
            parts = path[len("/v1/wf/"):].split("/")
            spec_name = parts[0]
            # /v1/wf/{spec_name}
            if len(parts) == 1 and method == "GET":
                return await wf_get_spec(_ParamsRequest(request, {"spec_name": spec_name}))
            # /v1/wf/{spec_name}/runs
            if len(parts) == 2 and parts[1] == "runs":
                if method == "POST":
                    return await wf_start_run(_ParamsRequest(request, {"spec_name": spec_name}))
                if method == "GET":
                    return await wf_get_run(_ParamsRequest(request, {"spec_name": spec_name}))
            # /v1/wf/{spec_name}/runs/{run_id}
            if len(parts) == 3 and parts[1] == "runs":
                run_id = parts[2]
                if method == "GET":
                    return await wf_get_run(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id}))
                if method == "DELETE":
                    return await wf_cancel_run(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id}))
            # /v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/{action}
            if len(parts) == 6 and parts[1] == "runs" and parts[3] == "steps":
                run_id, step_id, action = parts[2], parts[4], parts[5]
                if action == "approve" and method == "POST":
                    return await wf_approve_step(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id, "step_id": step_id}))
                if action == "continue" and method == "POST":
                    return await wf_continue_step(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id, "step_id": step_id}))
                if action == "rerun" and method == "POST":
                    return await wf_rerun_step(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id, "step_id": step_id}))
                if action == "execute" and method == "POST":
                    return await wf_execute_step(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id, "step_id": step_id}))
            # /v1/wf/{spec_name}/runs/{run_id}/artifacts
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "artifacts":
                run_id = parts[2]
                return await wf_list_artifacts(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id}))
            # /v1/wf/{spec_name}/runs/{run_id}/artifacts/{step_id}/{filename}
            if len(parts) == 6 and parts[1] == "runs" and parts[3] == "artifacts":
                run_id, step_id, filename = parts[2], parts[4], parts[5]
                return await wf_get_artifact(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id, "step_id": step_id, "filename": filename}))
            # /v1/wf/{spec_name}/runs/{run_id}/events
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "events":
                run_id = parts[2]
                return await wf_events(_ParamsRequest(request, {"spec_name": spec_name, "run_id": run_id}))

        from starlette.responses import JSONResponse
        return JSONResponse({"error": "not found", "path": path}, status_code=404)
