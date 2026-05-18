"""VNCCS workflow functions — multi-model orchestration on top of Wan2GPService.

Each function maps to one VNCCS ComfyUI workflow JSON. Steps call
svc.load()/svc.infer() in sequence, piping intermediates between model calls.

Workflows available:
  char_sheet    — Text → character base sheet (SD base + QWEN refine)
  emotions      — Sheet → emotion variations (loop over emotion tags)
  sprite        — Sheet + poses → sprite animation frames (BodyMesh + QWEN)
  pose_edit     — Character + BodyMesh pose → posed character image
  clone         — Reference character → new variant
  detailer      — Region face/hand refinement
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

from services.workflows.base import get_service, error_response
from services.workflows.utils.body_mesh import render_pose_b64

logger = logging.getLogger(__name__)


WORKFLOWS = [
    {"id": "vnccs/char-sheet",
     "description": "Text description → character base sheet (SD + QWEN refine)"},
    {"id": "vnccs/emotions",
     "description": "Character sheet → emotion variation set"},
    {"id": "vnccs/sprite",
     "description": "Character sheet + poses → animation sprite frames"},
    {"id": "vnccs/pose-edit",
     "description": "Character image + BodyMesh pose → posed character"},
    {"id": "vnccs/clone",
     "description": "Reference character → cloned variant"},
    {"id": "vnccs/detailer",
     "description": "Face/hand region refinement via inpainting"},
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


def _compose_images_side_by_side(*images_b64: str) -> str:
    """Place base64-encoded images side by side, return composite as base64."""
    from PIL import Image
    imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in images_b64]
    h = max(img.height for img in imgs)
    total_w = sum(img.width for img in imgs)
    composite = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for img in imgs:
        composite.paste(img, (x, 0))
        x += img.width
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def char_sheet(
    prompt: str,
    seed: int = 42,
    quality: str = "turbo",
    negative_prompt: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 1: text prompt → character base sheet.

    Two-stage pipeline:
      1. SD base (Z-Image) generates initial character image
      2. QWEN-Image-Edit refines face/body details
    """
    svc = get_service()
    steps = 8 if quality == "turbo" else 50

    svc.load("z_image")
    base = svc.infer({
        "input_prompt": prompt,
        "n_prompt": negative_prompt or "bad quality,worst quality",
        "seed": seed,
        "sampling_steps": steps,
        "guide_scale": 1.0 if quality == "turbo" else 4.0,
        "width": 1024,
        "height": 1024,
    })

    if base.get("status") != "ok":
        return error_response(f"Base generation failed: {base.get('error', 'unknown')}")

    svc.load("qwen-image-edit")
    refined = svc.infer({
        "input_prompt": "Draw character from image2",
        "image_b64": base["data"],
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
        "loras_selected": [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        ] if quality == "turbo" else [],
    })

    if refined.get("status") != "ok":
        return error_response(f"Refinement failed: {refined.get('error', 'unknown')}")

    return refined


def emotions(
    sheet_image_b64: str,
    emotions_list: list[str],
    costumes: list[str] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 3: character sheet → emotion variations.

    Emulates VNCCS EmotionGenerator by prompting QWEN with emotion tags.
    Each emotion+costume is a separate infer() call (looped, not batched).
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    if costumes is None:
        costumes = ["default"]

    results = []
    for costume in costumes:
        for emotion in emotions_list:
            result = svc.infer({
                "input_prompt": f"Draw character from image2, {emotion} expression",
                "image_b64": sheet_image_b64,
                "seed": seed,
                "sampling_steps": 4,
                "guide_scale": 1.0,
                "loras_selected": [
                    "VNCCS/EmotionCoreV1_000003000.safetensors",
                ],
            })
            if result.get("status") == "ok":
                results.append({
                    "emotion": emotion,
                    "costume": costume,
                    "data": result["data"],
                    "media_type": result.get("media_type", "image/png"),
                })
            else:
                logger.warning("Emotion %s/%s failed: %s", costume, emotion, result.get("error"))

    return {
        "status": "ok",
        "results": results,
        "total": len(results),
        "model": svc._loaded_model,
    }


def sprite(
    sheet_image_b64: str,
    poses: list[dict[str, Any]],
    directions: list[float] | None = None,
    seed: int = 42,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 4: character sheet + pose definitions → sprite frames.

    For each pose:
      1. BodyMeshRenderer renders the pose as a 3D mesh image (CPU)
      2. Mesh + character composited side-by-side → QWEN edit
      3. QWEN renders the character in the target pose

    Poses are rotations_json dicts (joint name → [rx, ry, rz]).
    Directions are model_rotation_y values (0=front, 90=right, 180=back, 270=left).
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    if directions is None:
        directions = [0.0]

    from services.workflows.utils.dwpose import skeleton_from_image_b64
    import numpy as np
    results = []
    for pose_idx, pose in enumerate(poses):
        for direction in directions:
            mesh_b64 = render_pose_b64(
                pose, model_rotation_y=direction, backend=backend)
            skeleton_b64 = skeleton_from_image_b64(mesh_b64, 1024, 1024)
            composite = _compose_images_side_by_side(mesh_b64, character_image_b64, skeleton_b64)

            result = svc.infer({
                "input_prompt": VNCCS_INSTRUCTION,
                "image_b64": composite,
                "seed": seed,
                "sampling_steps": 4,
                "guide_scale": 1.0,
                "loras_selected": [
                    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
                ],
            })
            if result.get("status") == "ok":
                results.append({
                    "pose_index": pose_idx,
                    "direction": direction,
                    "data": result["data"],
                    "media_type": result.get("media_type", "image/png"),
                })

    return {
        "status": "ok",
        "results": results,
        "total": len(results),
        "model": svc._loaded_model,
    }


def pose_edit(
    character_image_b64: str,
    rotations: dict[str, list[float]],
    model_rotation_y: float = 0.0,
    seed: int = 42,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS pose edit: character + BodyMesh pose → posed character image.

    Pipeline:
      1. BodyMeshRenderer renders the target pose as 3D mesh (CPU)
      2. DWPose extracts skeleton from mesh
      3. Mesh + character + skeleton composited side-by-side
      4. QWEN edits the character to match the pose
    """
    from services.workflows.utils.dwpose import skeleton_from_image_b64
    svc = get_service()
    svc.load("qwen-image-edit")

    mesh_b64 = render_pose_b64(rotations, model_rotation_y=model_rotation_y, backend=backend)
    skeleton_b64 = skeleton_from_image_b64(mesh_b64, 1024, 1024)
    composite = _compose_images_side_by_side(mesh_b64, character_image_b64, skeleton_b64)

    result = svc.infer({
        "input_prompt": VNCCS_INSTRUCTION,
        "image_b64": composite,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
        "loras_selected": [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
        ],
    })

    return result


def clone(
    reference_image_b64: str,
    character_def: dict[str, Any],
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 1.1: reference character → new variant.

    Uses CharacterCreator-like attributes + QWEN edit to re-render
    an existing character with modified appearance.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": "Draw character from image2",
        "image_b64": reference_image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
        **character_def,
    })

    return result


def detailer(
    image_b64: str,
    region_prompt: str = "improve face details",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS detailer: face/hand region refinement via inpainting."""
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": region_prompt,
        "image_b64": image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    return result
