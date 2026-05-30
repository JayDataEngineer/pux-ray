"""Workflow engine API routes — new declarative workflow system.

These routes power the new YAML-based workflow engine (services/workflows/engine.py).
They coexist with the legacy workflow routes in workflows.py during migration.

Routes:
  GET  /v1/wf                              — list available workflow specs
  GET  /v1/wf/{spec_name}                  — get spec details + input schema
  POST /v1/wf/{spec_name}/runs             — start a new run
  GET  /v1/wf/{spec_name}/runs             — list runs
  GET  /v1/wf/{spec_name}/runs/{run_id}    — get run status + step states
  DELETE /v1/wf/{spec_name}/runs/{run_id}  — cancel run
  POST /v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/approve  — approve interaction
  POST /v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/rerun    — rerun from step
  GET  /v1/wf/{spec_name}/runs/{run_id}/artifacts/{step_id}/{filename} — serve artifact
  GET  /v1/wf/{spec_name}/runs/{run_id}/events  — SSE stream
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from services.workflows.spec import load_spec, list_specs

logger = logging.getLogger(__name__)


def _get_engine():
    return serve.get_deployment_handle("workflow_engine", "workflow_engine")


# ---------------------------------------------------------------------------
# Spec discovery
# ---------------------------------------------------------------------------

async def wf_list_specs(request: Request) -> JSONResponse:
    specs = list_specs()
    items = []
    for name in specs:
        try:
            spec = load_spec(name)
            items.append({
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "steps": len(spec.steps),
            })
        except Exception:
            items.append({"name": name, "error": "failed to load"})
    return JSONResponse({"object": "list", "data": items})


async def wf_get_spec(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    try:
        spec = load_spec(spec_name)
    except FileNotFoundError:
        return JSONResponse({"error": f"Unknown workflow: {spec_name}"}, status_code=404)
    return JSONResponse({
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "inputs": {k: v.model_dump() for k, v in spec.inputs.items()},
        "steps": [
            {
                "id": s.id,
                "type": s.type,
                "service": s.service,
                "model": s.model,
                "depends_on": s.depends_on,
                "outputs": list(s.outputs.keys()),
                "interaction": s.interaction,
                "params": s.params,
            }
            for s in spec.steps
        ],
    })


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

async def wf_start_run(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    manual = body.pop("_manual", False)
    engine = _get_engine()
    try:
        result = await engine.start_run.remote(spec_name, body, manual=manual)
        return JSONResponse(result, status_code=201)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("Failed to start workflow run")
        return JSONResponse({"error": str(e)}, status_code=500)


async def wf_get_run(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)
    return JSONResponse(run)


async def wf_cancel_run(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if run and run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)
    result = await engine.cancel_run.remote(run_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Step interactions
# ---------------------------------------------------------------------------

async def wf_approve_step(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    step_id = request.path_params.get("step_id", "")

    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if not upload:
            return JSONResponse({"error": "No file in upload"}, status_code=400)
        file_data = await upload.read()
        data = {
            "file_data": file_data,
            "name": form.get("name", upload.filename or "upload"),
            "media_type": upload.content_type or "application/octet-stream",
        }
    else:
        try:
            data = await request.json()
        except Exception:
            data = {}
        except Exception:
            return JSONResponse({"error": "invalid JSON or multipart body"}, status_code=400)

    result = await engine.approve_step.remote(run_id, step_id, data)
    return JSONResponse(result)


async def wf_continue_step(request: Request) -> JSONResponse:
    """Continue past a review pause — signals the engine to advance."""
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    step_id = request.path_params.get("step_id", "")

    engine = _get_engine()
    result = await engine.approve_step.remote(run_id, step_id, {})
    return JSONResponse(result)


async def wf_rerun_step(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    step_id = request.path_params.get("step_id", "")

    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)

    try:
        new_params = await request.json()
    except Exception:
        new_params = None

    result = await engine.rerun_from.remote(run_id, step_id, new_params)
    return JSONResponse(result)


async def wf_execute_step(request: Request) -> JSONResponse:
    """Execute a single step in isolation (no downstream cascade)."""
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    step_id = request.path_params.get("step_id", "")

    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)

    try:
        params = await request.json()
    except Exception:
        params = None

    result = await engine.execute_single_step.remote(run_id, step_id, params)
    status_code = 200 if result.get("status") != "error" else 400
    return JSONResponse(result, status_code=status_code)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

async def wf_get_artifact(request: Request) -> Response:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")
    step_id = request.path_params.get("step_id", "")
    filename = request.path_params.get("filename", "")

    base = Path("/models/workflows")
    artifact_path = base / run_id / step_id / filename
    if not artifact_path.exists():
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    ext = artifact_path.suffix.lstrip(".").lower()
    media_types = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "mp4": "video/mp4", "webm": "video/webm",
        "wav": "audio/wav", "mp3": "audio/mp3", "ogg": "audio/ogg",
        "json": "application/json", "glb": "model/gltf-binary",
        "gltf": "model/gltf+json", "obj": "model/obj", "zip": "application/zip",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return Response(
        content=artifact_path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


async def wf_list_artifacts(request: Request) -> JSONResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")

    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)

    base = Path("/models/workflows") / run_id
    if not base.exists():
        return JSONResponse({"run_id": run_id, "artifacts": []})

    artifacts = []
    for step_dir in sorted(base.iterdir()):
        if step_dir.is_dir() and step_dir.name != "__pycache__":
            for f in sorted(step_dir.iterdir()):
                if f.is_file() and f.name != "state.json":
                    artifacts.append({
                        "step_id": step_dir.name,
                        "filename": f.name,
                        "size_bytes": f.stat().st_size,
                        "url": f"/v1/wf/runs/{run_id}/artifacts/{step_dir.name}/{f.name}",
                    })
    return JSONResponse({"run_id": run_id, "artifacts": artifacts})


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

async def wf_events(request: Request) -> StreamingResponse:
    spec_name = request.path_params.get("spec_name", "")
    run_id = request.path_params.get("run_id", "")

    engine = _get_engine()
    run = await engine.get_run.remote(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    if run.get("spec_name") != spec_name:
        return JSONResponse({"error": "Run does not belong to this spec"}, status_code=400)

    async def event_stream():
        try:
            async for event in engine.stream_events.remote(run_id):
                import json
                event_type = event.pop("event", "update")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
        except Exception as e:
            import json
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
