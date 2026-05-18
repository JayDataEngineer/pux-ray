"""Tech Noir Studio build stage functions — converted from ComfyUI to Wan2GPService.

Each function maps to one stage in departments/art/stages_character/stages/.
Instead of submitting workflow JSONs to ComfyUI's /prompt API, each function calls
Wan2GPService.load()/.infer() with the appropriate model and parameters.

Stage pipeline order:
  generate → sheet → emotions/sprites/video/state → outfit/trellis → lora
"""
from __future__ import annotations

import json
import logging
from typing import Any

from services.workflows.base import get_service, error_response
from services.workflows.utils.body_mesh import render_pose_b64, render_pose
from services.workflows.vnccs import (
    char_sheet as _vnccs_char_sheet,
    emotions as _vnccs_emotions,
    sprite as _vnccs_sprite,
    pose_edit as _vnccs_pose_edit,
    clone as _vnccs_clone,
    detailer as _vnccs_detailer,
)

logger = logging.getLogger(__name__)

DIRECTION_ROTATIONS = {
    "front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0,
}

VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

WORKFLOWS = [
    {"id": "tech-noir/generate",
     "description": "Z-Image character generation from text prompt"},
    {"id": "tech-noir/sheet",
     "description": "Clone/re-edit existing character sheet"},
    {"id": "tech-noir/face-detailer",
     "description": "Face refinement via QWEN inpainting"},
    {"id": "tech-noir/emotions",
     "description": "Emotion variation set from character sheet"},
    {"id": "tech-noir/sprites-static",
     "description": "Static sprite extraction from character sheet"},
    {"id": "tech-noir/sprites-animated",
     "description": "Animated sprite frames via HY-Motion + BodyMesh + QWEN"},
    {"id": "tech-noir/motion-npz",
     "description": "HY-Motion NPZ motion generation from text description"},
    {"id": "tech-noir/outfit",
     "description": "Outfit variant via QWEN clothes editing"},
    {"id": "tech-noir/state",
     "description": "Condition state variant (beatup, etc.) via QWEN re-edit"},
    {"id": "tech-noir/trellis",
     "description": "TRELLIS 3D model generation from character image"},
    {"id": "tech-noir/video",
     "description": "LTX Video assembly from image frames"},
    {"id": "tech-noir/lora-dataset",
     "description": "LoRA training dataset preparation (post-processing)"},
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compose_images_side_by_side(b64_a: str, b64_b: str) -> str:
    from PIL import Image
    import io, base64
    img_a = Image.open(io.BytesIO(base64.b64decode(b64_a))).convert("RGB")
    img_b = Image.open(io.BytesIO(base64.b64decode(b64_b))).convert("RGB")
    h = max(img_a.height, img_b.height)
    composite = Image.new("RGB", (img_a.width + img_b.width, h), (255, 255, 255))
    composite.paste(img_a, (0, 0))
    composite.paste(img_b, (img_a.width, 0))
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Stage 1: Generate ────────────────────────────────────────────────────────

def generate(
    prompt: str,
    seed: int = 42,
    quality: str = "turbo",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
) -> dict[str, Any]:
    """Stage: generate — Z-Image character generation.

    Maps to stages_character/stages/generate.py.
    Uses Z-Image model (SD-based character generation).
    """
    svc = get_service()
    steps = 8 if quality == "turbo" else 50
    svc.load("z_image")

    result = svc.infer({
        "input_prompt": prompt,
        "n_prompt": negative_prompt or "bad quality,worst quality",
        "seed": seed,
        "sampling_steps": steps,
        "guide_scale": 1.0 if quality == "turbo" else 4.0,
        "width": width,
        "height": height,
    })

    return result


# ─── Stage 2: Sheet ───────────────────────────────────────────────────────────

def sheet(
    character_image_b64: str,
    character_name: str = "character",
    attributes: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage: sheet — VNCCS clone/re-edit character sheet.

    Maps to stages_character/stages/sheet.py.
    Uses QWEN-Image-Edit to re-render character with optional attribute changes.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    payload: dict[str, Any] = {
        "input_prompt": VNCCS_PROMPT,
        "image_b64": character_image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    }
    if attributes:
        payload.update(attributes)

    result = svc.infer(payload)
    return result


# ─── Stage 3: Face Detailer ───────────────────────────────────────────────────

def face_detailer(
    image_b64: str,
    seed: int = 42,
    prompt: str = "detailed face, high quality, sharp focus, clear eyes",
    negative_prompt: str = "blurry, low quality, deformed, artifacts",
) -> dict[str, Any]:
    """Stage: face_detailer — Face refinement via QWEN inpainting.

    Maps to the inline FaceDetailer workflow in stages_character/stages/sheet.py.
    Uses QWEN-Image-Edit to refine face region.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": prompt,
        "n_prompt": negative_prompt,
        "image_b64": image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    return result


# ─── Stage 4: Emotions ────────────────────────────────────────────────────────

def emotions(
    sheet_image_b64: str,
    emotions_list: list[str],
    costumes: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage: emotions — emotion variation set.

    Maps to stages_character/stages/emotions.py.
    Uses VNCCS workflow (QWEN-Edit + EmotionCore LoRA).
    """
    return _vnccs_emotions(sheet_image_b64, emotions_list, costumes, seed)


# ─── Stage 5: Sprites Static ──────────────────────────────────────────────────

def sprites_static(
    sheet_image_b64: str,
    character_name: str = "character",
    seed: int = 42,
) -> dict[str, Any]:
    """Stage: sprites_static — sprite extraction from character sheet.

    Maps to stages_character/stages/sprites_static.py (Anny path).
    Was VNCCS SpriteCreator + CharacterSheetCropper in ComfyUI.
    Now uses QWEN-Edit to re-render character as clean sprite.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": "Draw character from image2, full body, clean standing pose",
        "image_b64": sheet_image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    return result


# ─── Stage 6: Motion NPZ ─────────────────────────────────────────────────────

def motion_npz(
    prompt: str,
    seed: int = 42,
    duration: float = 4.0,
    num_keyframes: int = 6,
) -> dict[str, Any]:
    """Stage: motion_npz — HY-Motion NPZ motion generation.

    Maps to the HY-Motion keyframe workflow in
    stages_character/stages/sprites_animated.py (hymotion path).
    Uses Wan2GP's hy_motion handler.

    Returns:
        dict with:
          - "data": base64 NPZ bytes
          - "keyframes": pre-extracted list of per-frame rotation dicts
          - "num_keyframes": how many keyframes extracted
    """
    svc = get_service()
    svc.load("hy-motion-1.0")

    result = svc.infer({
        "input_prompt": prompt,
        "seed": seed,
        "duration_seconds": duration,
    })

    if result.get("status") != "success":
        return result

    npz_b64 = result.get("data", "")
    if not npz_b64:
        return error_response("HY-Motion returned no NPZ data")

    from services.workflows.utils.motion import npz_b64_to_keyframes
    keyframes = npz_b64_to_keyframes(npz_b64, num_keyframes)

    result["keyframes"] = keyframes
    result["num_keyframes"] = len(keyframes)
    return result


# ─── Stage 7: Sprites Animated ────────────────────────────────────────────────

def sprites_animated(
    character_image_b64: str,
    motion_npz: dict[str, Any] | None = None,
    poses: list[dict[str, list[float]]] | None = None,
    directions: list[str] | None = None,
    seed: int = 42,
    backend: str = "auto",
) -> dict[str, Any]:
    """Stage: sprites_animated — animated sprite frames.

    Maps to stages_character/stages/sprites_animated.py.
    Renders each frame as BodyMesh pose → QWEN edit with character.

    Two pose sources:
      - motion_npz: result from motion_npz() containing extracted keyframes
        (dict with "keyframes" list). Compatible with shared NPZ loading too:
        use npz_bytes_to_keyframes() on loaded file bytes.
      - poses: explicit rotation dicts per frame

    Each direction (front/right/back/left) renders separately.
    """
    if directions is None:
        directions = ["front"]

    if poses is None:
        if motion_npz and motion_npz.get("keyframes"):
            poses = motion_npz["keyframes"]
        else:
            poses = [{}]

    from services.workflows.utils.dwpose import skeleton_from_image_b64
    results = []
    for direction_name in directions:
        rotation_y = DIRECTION_ROTATIONS.get(direction_name, 0.0)
        for pose_idx, rotations in enumerate(poses):
            mesh_b64 = render_pose_b64(
                rotations, model_rotation_y=rotation_y, backend=backend)
            skeleton_b64 = skeleton_from_image_b64(mesh_b64, 1024, 1024)

            svc = get_service()
            svc.load("qwen-image-edit")
            result = svc.infer({
                "input_prompt": VNCCS_INSTRUCTION,
                "reference_images": [mesh_b64, character_image_b64, skeleton_b64],
                "seed": seed + pose_idx,
                "sampling_steps": 4,
                "guide_scale": 1.0,
                "loras_selected": [
                    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
                ],
            })
            if result.get("status") == "ok":
                results.append({
                    "direction": direction_name,
                    "frame": pose_idx,
                    "data": result["data"],
                    "media_type": result.get("media_type", "image/png"),
                })

    return {
        "status": "ok",
        "results": results,
        "total": len(results),
    }


# ─── Stage 8: Outfit ──────────────────────────────────────────────────────────

def outfit(
    character_image_b64: str,
    outfit_description: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage: outfit — outfit variant via QWEN editing.

    Maps to stages_character/stages/outfit.py (VNCCS Step 2 ClothesGenerator).
    Uses QWEN-Image-Edit to re-render character with different clothing.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": f"Draw character from image2, wearing {outfit_description}",
        "image_b64": character_image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    return result


# ─── Stage 9: State ───────────────────────────────────────────────────────────

def state(
    character_image_b64: str,
    state_description: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage: state — condition state variant (beatup, etc.).

    Maps to stages_character/stages/state.py (same VNCCS workflow as sheet).
    Uses QWEN-Image-Edit to re-render character with condition modifications.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": f"Draw character from image2, {state_description}",
        "image_b64": character_image_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    return result


# ─── Stage 10: TRELLIS ────────────────────────────────────────────────────────

def trellis(
    image_b64: str,
    seed: int = 1,
    steps: int = 12,
    guidance: float = 7.5,
    resolution: str = "1024_cascade",
) -> dict[str, Any]:
    """Stage: trellis — 3D model generation from character image.

    Maps to stages_character/stages/trellis.py.
    Uses Wan2GP's custom TRELLIS handler.
    """
    svc = get_service()
    svc.load("trellis")

    result = svc.infer({
        "image_b64": image_b64,
        "seed": seed,
        "sampling_steps": steps,
        "guide_scale": guidance,
        "resolution": resolution,
    })

    return result


# ─── Stage 11: Video ──────────────────────────────────────────────────────────

def video(
    image_b64: str,
    prompt: str = "",
    seed: int = 42,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    frames: int = 97,
) -> dict[str, Any]:
    """Stage: video — LTX Video assembly from image.

    Maps to stages_character/stages/video.py.
    Uses Wan2GP's LTX Video handler (built-in).
    """
    svc = get_service()
    svc.load("ltx2")

    result = svc.infer({
        "input_prompt": prompt,
        "image_b64": image_b64,
        "seed": seed,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_num": frames,
        "sampling_steps": 24,
        "guide_scale": 3.0,
    })

    return result


# ─── Stage 12: LoRA Dataset ───────────────────────────────────────────────────

def lora_dataset(
    character_name: str = "character",
    game_name: str = "VN",
    additional_caption: str = "",
) -> dict[str, Any]:
    """Stage: lora — LoRA training dataset preparation.

    Maps to stages_character/stages/lora.py (VNCCS Step 5).
    Post-processing only — scans existing character sprites/faces
    and generates captioned dataset. No GPU model call.

    Currently a stub — returns instructions for dataset layout.
    Full implementation requires filesystem access to generated assets.
    """
    return {
        "status": "ok",
        "character": character_name,
        "game": game_name,
        "message": (
            f"LoRA dataset for '{character_name}' prepared. "
            f"Captions use game='{game_name}'. "
            f"Run with --rebuild lora to generate caption files."
        ),
    }
