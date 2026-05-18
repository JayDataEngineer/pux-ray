"""Workflow API routes — pseudo-OpenAI spec for multi-model orchestration.

GET  /v1/workflows                  — list available workflows
GET  /v1/workflows/{workflow}       — workflow metadata
POST /v1/workflows/{workflow}       — execute workflow
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

import services.workflows.vnccs as vnccs
import services.workflows.wdc as wdc
import services.workflows.tech_noir as tech_noir

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


async def list_workflows(request: Request) -> JSONResponse:
    items = [
        {"id": wf_id, "object": "workflow", "description": fn.__doc__ or ""}
        for wf_id, fn in _WORKFLOW_REGISTRY.items()
    ]
    return JSONResponse({"object": "list", "data": items})


async def get_workflow(request: Request) -> JSONResponse:
    wf_id = request.path_params.get("workflow", "")
    fn = _WORKFLOW_REGISTRY.get(wf_id)
    if not fn:
        return JSONResponse({"error": f"Unknown workflow: {wf_id}"}, status_code=404)
    return JSONResponse({
        "id": wf_id,
        "object": "workflow",
        "description": fn.__doc__ or "",
    })


async def execute_workflow(request: Request) -> JSONResponse:
    wf_id = request.path_params.get("workflow", "")
    fn = _WORKFLOW_REGISTRY.get(wf_id)
    if not fn:
        return JSONResponse({"error": f"Unknown workflow: {wf_id}"}, status_code=404)

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    try:
        result = fn(**body)
    except TypeError as e:
        return JSONResponse({"error": f"Invalid parameters: {e}"}, status_code=400)
    except Exception as e:
        logger.exception("Workflow %s failed", wf_id)
        return JSONResponse({"error": str(e)}, status_code=500)

    if isinstance(result, dict) and result.get("status") == "error":
        return JSONResponse(result, status_code=500)

    return JSONResponse(result)
