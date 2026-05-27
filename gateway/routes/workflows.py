"""Workflow API routes — legacy Python + new engine bridge.

GET  /v1/workflows                  — list available workflows
GET  /v1/workflows/{workflow}       — workflow metadata
POST /v1/workflows/{workflow}       — execute workflow (via Forge VRAM ledger)

Legacy workflows are backed by Python functions. New workflows backed by YAML
specs are dispatched to the workflow engine. The listing merges both sources.

Deprecation: legacy /v1/workflows routes are superseded by /v1/wf routes.
Responses include a Deprecation header directing clients to the new API.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

import services.workflows.vnccs as vnccs
import services.workflows.wdc as wdc
import services.workflows.tech_noir as tech_noir

from services.workflows.spec import load_spec, list_specs as list_yaml_specs

logger = logging.getLogger(__name__)

_WORKFLOW_REGISTRY: dict[str, Callable] = {}

def _register(fn: Callable, workflow_id: str, description: str) -> None:
    _WORKFLOW_REGISTRY[workflow_id] = fn

# VNCCS workflows
_register(vnccs.char_sheet, "vnccs/char-sheet", "Text → character base sheet (SD + QWEN refine)")
_register(vnccs.emotions,   "vnccs/emotions",   "Character sheet → emotion variation set")
_register(vnccs.sprite,     "vnccs/sprite",     "Character sheet + poses → animation frames")
_register(vnccs.pose_edit,  "vnccs/pose-edit",  "Character image + pose → posed character")
_register(vnccs.clone,      "vnccs/clone",      "Reference character → cloned variant")
_register(vnccs.detailer,   "vnccs/detailer",   "Face/hand region refinement")
# WDC workflows
_register(wdc.ltx_fflf_2stage, "wdc/ltx-fflf-2stage", "Image-to-video with 2-stage FFLF")
_register(wdc.ltx_fflf_3stage, "wdc/ltx-fflf-3stage", "Image-to-video with 3-stage FFLF + upscale")
_register(wdc.ltx_audio,       "wdc/ltx-audio",       "Image-to-video with audio conditioning")
_register(wdc.timeline,        "wdc/timeline",        "Multi-shot timeline video")
# Tech Noir Studio workflows
_register(tech_noir.generate,     "tech-noir/generate",     "Z-Image character generation")
_register(tech_noir.sheet,        "tech-noir/sheet",        "Clone/re-edit character sheet")
_register(tech_noir.face_detailer, "tech-noir/face-detailer", "Face refinement via QWEN")
_register(tech_noir.emotions,     "tech-noir/emotions",     "Emotion variation set")
_register(tech_noir.sprites_static, "tech-noir/sprites-static", "Sprite extraction from sheet")
_register(tech_noir.sprites_animated, "tech-noir/sprites-animated", "Animated sprite frames")
_register(tech_noir.motion_npz,   "tech-noir/motion-npz",  "HY-Motion motion generation")
_register(tech_noir.outfit,       "tech-noir/outfit",      "Outfit variant via QWEN")
_register(tech_noir.state,        "tech-noir/state",       "Condition state variant")
_register(tech_noir.trellis,      "tech-noir/trellis",     "TRELLIS 3D model generation")
_register(tech_noir.video,        "tech-noir/video",       "LTX Video assembly")
_register(tech_noir.lora_dataset, "tech-noir/lora-dataset", "LoRA dataset preparation")

# Map legacy workflow IDs to YAML spec names where a migration exists
_YAML_MIGRATION_MAP = {
    "vnccs/char-sheet": "character_sheet",
    "vnccs/pose-edit": "vnccs_pose_edit",
    "tech-noir/generate": "tech_noir_generate",
    "tech-noir/sheet": "tech_noir_sheet",
    "tech-noir/video": "tech_noir_video",
    "tech-noir/trellis": "tech_noir_trellis",
    "tech-noir/face-detailer": "tech_noir_face_detailer",
    "tech-noir/motion-npz": "tech_noir_motion_npz",
    "tech-noir/outfit": "tech_noir_outfit",
    "tech-noir/state": "tech_noir_state",
    "wdc/ltx-fflf-2stage": "wdc_ltx_fflf_2stage",
    "wdc/ltx-audio": "wdc_ltx_audio",
}

_DEPRECATION_HEADER = {
    "Deprecation": "true",
    "Link": '</v1/wf>; rel="successor-version"',
}


async def list_workflows(request: Request) -> JSONResponse:
    """List all workflows — legacy Python + YAML spec migrations."""
    items = []
    # Legacy Python workflows
    for wf_id, fn in _WORKFLOW_REGISTRY.items():
        yaml_spec = _YAML_MIGRATION_MAP.get(wf_id)
        items.append({
            "id": wf_id,
            "object": "workflow",
            "description": fn.__doc__ or "",
            "engine": "legacy",
            "yaml_spec": yaml_spec,
        })
    # YAML-only specs not in legacy registry
    legacy_yaml_names = set(_YAML_MIGRATION_MAP.values())
    for spec_name in list_yaml_specs():
        if spec_name.startswith("_"):
            continue
        if spec_name not in legacy_yaml_names:
            try:
                spec = load_spec(spec_name)
                items.append({
                    "id": spec_name,
                    "object": "workflow",
                    "description": spec.description,
                    "engine": "yaml",
                    "yaml_spec": spec_name,
                })
            except Exception:
                items.append({"id": spec_name, "engine": "yaml", "error": "failed to load"})
    return JSONResponse(
        {"object": "list", "data": items},
        headers=_DEPRECATION_HEADER,
    )


async def get_workflow(request: Request) -> JSONResponse:
    wf_id = request.path_params.get("workflow", "")
    return _get_workflow_json(wf_id)


def _get_workflow_json(wf_id: str) -> JSONResponse:
    # Check YAML migration first
    spec_name = _YAML_MIGRATION_MAP.get(wf_id)
    if spec_name:
        try:
            spec = load_spec(spec_name)
            return JSONResponse({
                "id": wf_id,
                "object": "workflow",
                "description": spec.description,
                "engine": "yaml",
                "yaml_spec": spec_name,
                "inputs": {k: v.model_dump() for k, v in spec.inputs.items()},
                "steps": [
                    {"id": s.id, "type": s.type, "depends_on": s.depends_on}
                    for s in spec.steps
                ],
            }, headers=_DEPRECATION_HEADER)
        except FileNotFoundError:
            pass

    fn = _WORKFLOW_REGISTRY.get(wf_id)
    if not fn:
        return JSONResponse({"error": f"Unknown workflow: {wf_id}"}, status_code=404)
    return JSONResponse({
        "id": wf_id,
        "object": "workflow",
        "description": fn.__doc__ or "",
        "engine": "legacy",
    }, headers=_DEPRECATION_HEADER)


async def execute_workflow(request: Request) -> JSONResponse:
    """Execute a workflow — routes YAML migrations to engine, legacy to Forge."""
    wf_id = request.path_params.get("workflow", "")
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    # Route migrated workflows to the new engine
    spec_name = _YAML_MIGRATION_MAP.get(wf_id)
    if spec_name:
        try:
            engine = serve.get_deployment_handle("workflow_engine", "workflow_engine")
            result = await engine.start_run.remote(spec_name, body)
            return JSONResponse(result, status_code=201, headers=_DEPRECATION_HEADER)
        except Exception as e:
            logger.exception("Engine failed for migrated workflow %s", wf_id)
            return JSONResponse({"error": str(e)}, status_code=500)

    # Legacy: route through Forge for VRAM-aware execution
    fn = _WORKFLOW_REGISTRY.get(wf_id)
    if not fn:
        return JSONResponse({"error": f"Unknown workflow: {wf_id}"}, status_code=404)

    try:
        forge = serve.get_deployment_handle("forge", "forge")
        result = await forge.run_pipeline.remote(wf_id, body)
        if isinstance(result, dict) and result.get("status") == "error":
            return JSONResponse(result, status_code=500, headers=_DEPRECATION_HEADER)
        return JSONResponse(result, headers=_DEPRECATION_HEADER)
    except Exception as e:
        logger.exception("Workflow %s failed to route through Forge", wf_id)
        return JSONResponse({"error": str(e)}, status_code=503)
