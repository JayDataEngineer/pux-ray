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

VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

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


def _build_char_prompt(
    aesthetics: str = "",
    background_color: str = "green",
    nsfw: bool = False,
    sex: str = "female",
    age: int = 18,
    race: str = "",
    eyes: str = "",
    hair: str = "",
    face: str = "",
    body: str = "",
    skin_color: str = "",
    additional_details: str = "",
    lora_prompt: str = "",
) -> tuple[str, str]:
    """Build positive and negative prompts matching VNCCS CharacterCreator exactly."""
    pos = f"{aesthetics or 'masterpiece,best quality,amazing quality'}, simple background, expressionless"
    if background_color:
        pos += f", {background_color} background"

    # Sex tokens
    sex = sex.lower().strip() if sex else "female"
    is_male = sex in ("male", "man", "boy", "m")
    if is_male:
        pos += ", (1boy)"
        neg_sex = "1girl, girl, woman, femine, breasts, vagina"
    else:
        pos += ", (1girl)"
        neg_sex = "1boy, man, penis, dick"

    # NSFW / nude phrase
    if nsfw:
        nude = "(naked, nude, penis)" if is_male else "(naked, nude, vagina, nipples)"
    else:
        nude = "(bare chest, wear white boxers)" if is_male else "(wear white bra and panties)"
    pos += f", {nude}"

    # Age
    pos += f", {age}yo"
    if is_male:
        if age <= 3: pos += ", (toddler boy:1.0)"
        elif age <= 11: pos += ", (shota:1.0)"
        elif age <= 16: pos += ", (teenager boy:1.0)"
        elif age <= 18: pos += ", (young_adult man:1.0)"
        elif age <= 24: pos += ", (young_adult man:1.5)"
        elif age <= 50: pos += ", (adult man:1.0)"
        elif age <= 60: pos += ", (old man:1.0)"
    else:
        if age <= 3: pos += ", (toddler girl:1.0)"
        elif age <= 11: pos += ", (loli:1.0)"
        elif age <= 18: pos += ", (teenager girl:1.0)"
        elif age <= 24: pos += ", (young_adult woman:1.0)"
        elif age <= 50: pos += ", (adult woman:1.0)"
        elif age <= 60: pos += ", (old woman:1.0)"

    # Attributes
    if race: pos += f", ({race} race:1.0)"
    if hair: pos += f", ({hair} hair:1.0)"
    if eyes: pos += f", ({eyes} eyes:1.0)"
    if face: pos += f", ({face} face:1.0)"
    if body: pos += f", ({body} body:1.0)"
    if skin_color: pos += f", ({skin_color} skin:1.0)"
    if additional_details: pos += f", ({additional_details})"
    if lora_prompt: pos += f", {lora_prompt}"

    neg = f"bad quality,worst quality,worst detail,sketch,censor, missing arm, missing leg, distorted body, footwear, {neg_sex}"

    return pos, neg


def _build_face_details(
    sex: str = "female",
    race: str = "",
    eyes: str = "",
    hair: str = "",
    face: str = "",
    skin_color: str = "",
    additional_details: str = "",
) -> str:
    """Build face detail string matching VNCCS build_face_details."""
    parts = ["1girl" if sex in ("female",) else "1boy"]
    if race: parts.append(f"{race} race")
    if eyes: parts.append(f"{eyes} eyes")
    if hair: parts.append(f"{hair} hair")
    if face: parts.append(f"{face} face")
    if skin_color: parts.append(f"{skin_color} skin")
    if additional_details: parts.append(additional_details)
    return ", ".join(p for p in parts if p) + ", (expressionless:1.0)"


def char_sheet(
    prompt: str = "",
    image_b64: str | None = None,
    reference_image_b64: str | None = None,
    model: str = "z_image",
    seed: int = 42,
    quality: str = "turbo",
    negative_prompt: str = "",
    sex: str = "female",
    age: int = 18,
    nsfw: bool = False,
    background_color: str = "green",
    aesthetics: str = "",
    race: str = "",
    eyes: str = "",
    hair: str = "",
    face: str = "",
    body: str = "",
    skin_color: str = "",
    additional_details: str = "",
    lora_prompt: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 1: full character sheet generation matching ComfyUI workflow.

    Builds the prompt exactly like VNCCS CharacterCreator, generates the base
    character via SD, refines with QWEN, and outputs body sheet + face details.

    If image_b64 is provided, the SD base generation is skipped and the
    image goes directly to QWEN refinement.

    Returns base64-encoded image data.
    """
    svc = get_service()

    # Build prompts matching CharacterCreator
    char_prompt, built_negative = _build_char_prompt(
        aesthetics=aesthetics or "masterpiece,best quality,amazing quality",
        background_color=background_color,
        nsfw=nsfw, sex=sex, age=age, race=race,
        eyes=eyes, hair=hair, face=face, body=body,
        skin_color=skin_color, additional_details=additional_details,
        lora_prompt=lora_prompt,
    )
    final_negative = negative_prompt or built_negative

    # Use prompt param as override if provided
    gen_prompt = prompt if prompt else char_prompt

    if image_b64:
        base_image = image_b64
    else:
        steps = 8 if quality == "turbo" else 50
        svc.load(model)
        infer_kw: dict[str, Any] = {
            "model": model,
            "input_prompt": gen_prompt,
            "seed": seed,
            "sampling_steps": steps,
            "guide_scale": 1.0 if quality == "turbo" else 4.0,
            "width": 1024,
            "height": 1024,
        }
        if final_negative:
            infer_kw["n_prompt"] = final_negative
        base = svc.infer(infer_kw)
        if base.get("status") != "ok":
            return error_response(f"Base generation failed: {base.get('error', 'unknown')}")
        base_image = base["data"]

    # Return result - QWEN refinement disabled temporarily
    if image_b64:
        return {"status": "ok", "data": image_b64, "media_type": "image/png",
                "message": "Input image passed through (QWEN refinement disabled)"}
    return base


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

            result = svc.infer({
                "input_prompt": VNCCS_INSTRUCTION,
                "reference_images": [mesh_b64, sheet_image_b64, skeleton_b64],
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


# Pre-defined VNCCS pose presets matching the original PoseGenerator
_POSE_PRESETS: dict[str, dict[str, list[float]]] = {
    "front": {},
    "side": {"model_rotation_y": [0, 90, 0]},
    "walk": {"r_elbow": [0, 0, -30], "l_elbow": [0, 0, 30], "r_knee": [0, 0, 20], "l_knee": [0, 0, -10]},
    "sit": {"r_knee": [90, 0, 0], "l_knee": [90, 0, 0], "r_hip": [-45, 0, 0], "l_hip": [-45, 0, 0]},
    "arms_crossed": {"r_shoulder": [0, 0, -90], "l_shoulder": [0, 0, 90], "r_elbow": [0, 0, -90], "l_elbow": [0, 0, 90]},
    "hand_on_hip": {"r_shoulder": [0, 0, -45], "r_elbow": [0, 0, -90], "r_wrist": [0, 0, -45]},
    "pointing": {"r_shoulder": [0, 0, -90], "r_elbow": [0, 0, -180], "r_wrist": [0, 0, 0]},
    "salute": {"r_shoulder": [0, 0, -180], "r_elbow": [0, 0, 0], "r_wrist": [0, 0, 0]},
    "kneel": {"r_knee": [90, 0, 0], "l_knee": [45, 0, 0], "r_hip": [-90, 0, 0], "l_hip": [-45, 0, 0]},
    "lean": {"r_hip": [15, 0, 0], "l_hip": [15, 0, 0], "spine": [-15, 0, 0]},
}


def pose_edit(
    character_image_b64: str,
    pose_image_b64: str | None = None,
    rotations: dict[str, list[float]] | None = None,
    model_rotation_y: float = 0.0,
    seed: int = 42,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS pose edit: character + pose reference → posed character image.

    If pose_image_b64 is provided, DWPose extracts skeleton from the
    reference pose image directly (matching the original Pose Studio
    workflow). Otherwise, BodyMesh renders a default T-pose.

    Pipeline:
      1. DWPose extracts skeleton from reference pose (or BodyMesh renders T-pose)
      2. Mesh/skeleton + character composited side-by-side
      3. QWEN edits the character to match the target pose
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    if pose_image_b64:
        # Extract pose from reference image via DWPose
        from services.workflows.utils.dwpose import skeleton_from_image_b64
        skeleton_b64 = skeleton_from_image_b64(pose_image_b64, 1024, 1024)
        reference_images = [pose_image_b64, character_image_b64, skeleton_b64]
        input_prompt = (
            "Match the body pose shown in Picture 1. "
            "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
            "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
            "while maintaining the character's identity, clothing, and appearance."
        )
    else:
        # Fall back to BodyMesh default T-pose
        from services.workflows.utils.dwpose import skeleton_from_image_b64
        resolved = rotations if rotations else {}
        mesh_b64 = render_pose_b64(resolved, model_rotation_y=model_rotation_y, backend=backend)
        skeleton_b64 = skeleton_from_image_b64(mesh_b64, 1024, 1024)
        reference_images = [mesh_b64, character_image_b64, skeleton_b64]
        input_prompt = VNCCS_INSTRUCTION

    result = svc.infer({
        "input_prompt": input_prompt,
        "reference_images": reference_images,
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
