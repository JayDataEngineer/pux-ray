"""LoRA training data generation workflows — batch pipeline for Klein-PoseEdit.

Generates training pairs for LoRA fine-tuning of Klein 4B on pose editing.
Each training sample is a triplet: (body_mesh, character, skeleton) → teacher output.

Pipeline stages:
  1. synth_chars    — Generate diverse anime characters via Z-Image Base
  2. synth_poses    — Generate random pose definitions (Anny rotations)
  3. extract_skeletons — DWPose skeleton extraction from mesh renders
  4. teacher_edit   — QWEN-Image-Edit pose transfer (VNCCS teacher)
  5. full_pipeline   — Run all stages end-to-end for N samples

Uses Wan2GPService for all GPU inference. No ComfyUI dependency.
"""
from __future__ import annotations

import base64
import io
import logging
import random
from typing import Any

from services.workflows.base import get_service, error_response, encode_output
from services.workflows.utils.body_mesh import render_pose, render_pose_b64
from services.workflows.utils.dwpose import skeleton_from_image_b64, skeleton_from_image

logger = logging.getLogger(__name__)

WORKFLOWS = [
    {"id": "lora-data/synth-chars",
     "description": "Generate diverse anime character images via Z-Image Base"},
    {"id": "lora-data/synth-poses",
     "description": "Generate random Anny pose definitions"},
    {"id": "lora-data/render-mesh",
     "description": "Render Anny body mesh from pose rotations"},
    {"id": "lora-data/extract-skeleton",
     "description": "DWPose skeleton extraction from image"},
    {"id": "lora-data/teacher-edit",
     "description": "QWEN pose transfer — the VNCCS teacher step"},
    {"id": "lora-data/full-sample",
     "description": "Complete training sample: prompt → char → mesh → skeleton → teacher"},
]

# Body parts for random pose generation
_JOINT_NAMES = [
    "spine", "spine_03", "neck", "head",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]

_BODY_DIRECTIONS = {
    "front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0,
}

# VNCCS PoseStudio V2 instruction for multi-image reference editing
TEACHER_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

TEACHER_LORAS = [
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
]

# Danbooru-style prompt building blocks
_HAIR_COLORS = [
    "blue hair", "pink hair", "silver hair", "blonde hair", "red hair",
    "black hair", "white hair", "purple hair", "green hair", "orange hair",
    "light blue hair", "dark brown hair", "gradient hair", "two-tone hair",
]
_HAIR_STYLES = [
    "long hair", "short hair", "twin tails", "ponytail", "bob cut",
    "wavy hair", "straight hair", "messy hair", "side ponytail", "braided hair",
]
_EYE_COLORS = [
    "blue eyes", "red eyes", "green eyes", "golden eyes", "purple eyes",
    "heterochromia", "silver eyes", "brown eyes",
]
_OUTFITS = [
    "school uniform", "casual clothes", "fantasy armor", "maid outfit",
    "gothic dress", "kimono", "military uniform", "witch costume",
    "shrine maiden outfit", "cyberpunk outfit", "noble dress",
    "swimsuit", "winter coat", "lab coat", "pirate outfit",
]
_BODY_TYPES = [
    "", "", "",  # mostly default
    "petite", "tall",
]
_ACCESSORIES = [
    "", "", "",  # mostly none
    "glasses", "cat ears", "headband", "ribbon", "earrings",
    "necklace", "hat", "scarf", "choker", "hair ornament",
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


def _random_prompt() -> str:
    """Build a random Danbooru-style character description."""
    parts = [
        "1girl", "solo",
        random.choice(_HAIR_COLORS),
        random.choice(_HAIR_STYLES),
        random.choice(_EYE_COLORS),
        random.choice(_OUTFITS),
        random.choice(_BODY_TYPES),
        random.choice(_ACCESSORIES),
        "detailed face", "full body", "standing",
        "anime style", "high quality",
    ]
    return ", ".join(p for p in parts if p)


def _random_pose() -> dict[str, list[float]]:
    """Generate a random but plausible Anny pose."""
    rotations: dict[str, list[float]] = {}

    # Spine — small adjustments
    rotations["spine"] = [random.uniform(-15, 15), random.uniform(-10, 10), random.uniform(-10, 10)]
    rotations["spine_03"] = [random.uniform(-10, 10), random.uniform(-8, 8), random.uniform(-8, 8)]

    # Head
    rotations["neck"] = [random.uniform(-20, 20), random.uniform(-25, 25), random.uniform(-15, 15)]
    rotations["head"] = [random.uniform(-10, 10), random.uniform(-15, 15), random.uniform(-10, 10)]

    # Arms — wider range for expressiveness
    for side in ("right", "left"):
        mirror = 1.0 if side == "right" else -1.0
        rotations[f"{side}_shoulder"] = [
            random.uniform(-30, 90),
            mirror * random.uniform(-60, 60),
            random.uniform(-45, 45),
        ]
        rotations[f"{side}_elbow"] = [
            random.uniform(0, 120),
            random.uniform(-20, 20),
            random.uniform(-15, 15),
        ]
        rotations[f"{side}_wrist"] = [
            random.uniform(-30, 30),
            random.uniform(-30, 30),
            random.uniform(-45, 45),
        ]

    # Legs — moderate range
    for side in ("right", "left"):
        rotations[f"{side}_hip"] = [
            random.uniform(-15, 30),
            random.uniform(-15, 15),
            random.uniform(-10, 10),
        ]
        rotations[f"{side}_knee"] = [
            random.uniform(-5, 60),
            random.uniform(-5, 5),
            random.uniform(-5, 5),
        ]
        rotations[f"{side}_ankle"] = [
            random.uniform(-20, 20),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
        ]

    return rotations


# ─── Stage 1: Generate Characters ──────────────────────────────────────────

def synth_chars(
    prompts: list[str] | None = None,
    num_chars: int = 1,
    seed: int | None = None,
    negative_prompt: str = "bad quality, worst quality, blurry, deformed",
    width: int = 1024,
    height: int = 1024,
) -> dict[str, Any]:
    """Generate diverse anime character images using Z-Image Base.

    Z-Image Base (non-distilled) provides maximum diversity for training data.
    Uses CFG 4.0 and 30 steps (not Turbo's cfg=0, 8 steps).
    """
    if prompts is None:
        prompts = [_random_prompt() for _ in range(num_chars)]

    svc = get_service()
    svc.load("z_image_base")

    results = []
    for i, prompt in enumerate(prompts):
        img_seed = (seed + i) if seed is not None else random.randint(0, 2**31)
        result = svc.infer({
            "input_prompt": prompt,
            "n_prompt": negative_prompt,
            "seed": img_seed,
            "sampling_steps": 30,
            "guide_scale": 4.0,
            "width": width,
            "height": height,
        })
        if result.get("status") == "ok":
            results.append({
                "index": i,
                "prompt": prompt,
                "seed": img_seed,
                "data": result["data"],
                "media_type": result.get("media_type", "image/png"),
            })
        else:
            logger.warning("Character %d generation failed: %s", i, result.get("error"))

    return {"status": "ok", "results": results, "total": len(results)}


# ─── Stage 2: Generate Poses ───────────────────────────────────────────────

def synth_poses(
    num_poses: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate random Anny pose definitions (CPU, no GPU needed).

    Returns rotation dicts suitable for render_pose() / render_pose_b64().
    """
    if seed is not None:
        random.seed(seed)

    poses = [_random_pose() for _ in range(num_poses)]
    return {"status": "ok", "poses": poses, "total": len(poses)}


# ─── Stage 3: Render Mesh ──────────────────────────────────────────────────

def render_mesh(
    rotations: dict[str, list[float]],
    model_rotation_y: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    backend: str = "auto",
    phenotype: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render an Anny body mesh from pose rotations (CPU, no GPU needed).

    Returns the mesh image as base64 PNG.
    """
    arr = render_pose(
        rotations,
        width=width, height=height,
        model_rotation_y=model_rotation_y,
        backend=backend,
        phenotype=phenotype,
    )

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    mesh_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "status": "ok",
        "data": mesh_b64,
        "media_type": "image/png",
        "width": width,
        "height": height,
        "rotation_y": model_rotation_y,
    }


# ─── Stage 4: Extract Skeleton ─────────────────────────────────────────────

def extract_skeleton(
    image_b64: str,
    width: int = 1024,
    height: int = 1024,
) -> dict[str, Any]:
    """Extract DWPose skeleton from an image.

    Runs YOLOX detection + RTMPose estimation via ONNX (CPU).
    Returns the skeleton overlay on white background as base64 PNG.
    """
    skeleton_b64 = skeleton_from_image_b64(image_b64, width, height)
    return {
        "status": "ok",
        "data": skeleton_b64,
        "media_type": "image/png",
        "width": width,
        "height": height,
    }


# ─── Stage 5: Teacher Edit ─────────────────────────────────────────────────

def teacher_edit(
    character_image_b64: str,
    mesh_image_b64: str,
    skeleton_image_b64: str,
    instruction: str = TEACHER_INSTRUCTION,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the VNCCS QWEN-Image-Edit teacher step.

    Takes the triplet (mesh, character, skeleton) and produces a
    posed character image. This is the training target.
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    result = svc.infer({
        "input_prompt": instruction,
        "reference_images": [mesh_image_b64, character_image_b64, skeleton_image_b64],
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
        "loras_selected": TEACHER_LORAS,
    })
    return result


# ─── Full Pipeline: One Complete Training Sample ────────────────────────────

def full_sample(
    prompt: str | None = None,
    pose: dict[str, list[float]] | None = None,
    direction: str = "front",
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    backend: str = "auto",
) -> dict[str, Any]:
    """Generate one complete training sample end-to-end.

    Pipeline:
      1. Build random prompt if not provided
      2. Generate character image (Z-Image Base)
      3. Generate random pose if not provided
      4. Render body mesh (Anny, CPU)
      5. Extract skeleton (DWPose, CPU)
      6. Run teacher edit (QWEN-Image-Edit + VNCCS LoRAs)

    Returns all intermediates + the final teacher output.
    """
    if seed is None:
        seed = random.randint(0, 2**31)

    if prompt is None:
        prompt = _random_prompt()

    if pose is None:
        pose = _random_pose()

    rotation_y = _BODY_DIRECTIONS.get(direction, 0.0)

    # Step 1: Generate character
    char_result = synth_chars(
        prompts=[prompt], seed=seed, width=width, height=height,
    )
    if not char_result["results"]:
        return error_response("Character generation failed")
    character = char_result["results"][0]
    char_b64 = character["data"]

    # Step 2: Render mesh
    mesh_result = render_mesh(
        pose, model_rotation_y=rotation_y,
        width=width, height=height, backend=backend,
    )
    mesh_b64 = mesh_result["data"]

    # Step 3: Extract skeleton from mesh
    skeleton_result = extract_skeleton(mesh_b64, width=width, height=height)
    skeleton_b64 = skeleton_result["data"]

    # Step 4: Teacher edit
    teacher_result = teacher_edit(
        character_image_b64=char_b64,
        mesh_image_b64=mesh_b64,
        skeleton_image_b64=skeleton_b64,
        seed=seed,
    )

    if teacher_result.get("status") != "ok":
        return error_response(f"Teacher edit failed: {teacher_result.get('error')}")

    return {
        "status": "ok",
        "prompt": prompt,
        "seed": seed,
        "pose": pose,
        "direction": direction,
        "character": {
            "data": char_b64,
            "media_type": character["media_type"],
        },
        "mesh": {
            "data": mesh_b64,
            "media_type": "image/png",
        },
        "skeleton": {
            "data": skeleton_b64,
            "media_type": "image/png",
        },
        "teacher_output": {
            "data": teacher_result["data"],
            "media_type": teacher_result.get("media_type", "image/png"),
        },
    }


# ─── Batch Pipeline ────────────────────────────────────────────────────────

def batch_generate(
    num_samples: int = 10,
    directions: list[str] | None = None,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    backend: str = "auto",
) -> dict[str, Any]:
    """Generate N complete training samples.

    Each sample uses a unique character, unique pose, and the specified
    view directions. Outputs all intermediates for dataset QC.
    """
    if directions is None:
        directions = ["front"]

    if seed is not None:
        random.seed(seed)

    results = []
    for i in range(num_samples):
        sample_seed = random.randint(0, 2**31)
        direction = directions[i % len(directions)]

        sample = full_sample(
            seed=sample_seed,
            direction=direction,
            width=width,
            height=height,
            backend=backend,
        )
        if sample.get("status") == "ok":
            results.append(sample)
        else:
            logger.warning("Sample %d failed: %s", i, sample.get("error"))

    return {
        "status": "ok",
        "results": results,
        "total": len(results),
        "requested": num_samples,
    }
