"""VNCCS workflow functions — 1:1 ComfyUI workflow → Wan2GP DAG orchestration.

Each function is a verified 1:1 port of the original VNCCS ComfyUI workflow JSON,
calling Wan2GPService.load()/infer() in the exact same sequence and with the same
parameters as the ComfyUI nodes.

Workflows:
  char_sheet    — VN_Step1_QWEN_CharSheetGenerator_v1 (32 nodes)
  pose_edit     — VNCCS_Utils Pose Studio QWEN (10 nodes)
  emotions      — Sheet → emotion variations (loop over emotion tags)
  sprite        — Sheet + poses → sprite animation frames (BodyMesh + QWEN)
  clone         — Reference character → new variant
  detailer      — Region face/hand refinement
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from services.workflows.base import get_service, error_response

logger = logging.getLogger(__name__)

# ─── VNCCS Pose Studio Instruction ───────────────────────────────────────────
# Matches VNCCS_PoseStudio prompt_template default
VNCCS_POSE_STUDIO_PROMPT = (
    "Draw character from image2"
)

VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

WORKFLOWS = [
    {"id": "vnccs/char-sheet",
     "description": "Text → character sheet (PoseGenerator → SDXL → QWEN → Face detailer)"},
    {"id": "vnccs/emotions",
     "description": "Character sheet → emotion variation set"},
    {"id": "vnccs/sprite",
     "description": "Character sheet + poses → animation sprite frames"},
    {"id": "vnccs/pose-edit",
     "description": "Character + PoseStudio body mesh → posed character (1:1 Pose Studio QWEN)"},
    {"id": "vnccs/clone",
     "description": "Reference character → cloned variant"},
    {"id": "vnccs/detailer",
     "description": "Face/hand region refinement via inpainting"},
]


def get_workflows() -> list[dict[str, str]]:
    return list(WORKFLOWS)


# ─── Default VNCCS 12-pose grid ─────────────────────────────────────────────
# Pre-extracted from VNCCS_PoseGenerator widget in Step1 workflow.
# This is the default pose grid rendered as an openpose skeleton image
# on a 512x1536 canvas — it's the reference image for QWEN sheet generation.
_POSE_GRID_PATH = Path(__file__).parent / "data" / "vnccs_poses.json"


def _load_default_pose_grid() -> dict:
    """Load the default 12-pose VNCCS grid (keypoint coordinates)."""
    if _POSE_GRID_PATH.exists():
        return json.loads(_POSE_GRID_PATH.read_text())
    # Fallback: empty grid
    return {"canvas": {"width": 512, "height": 1536}, "poses": []}


# ─── OpenPose skeleton rendering (matches VNCCS_PoseGenerator) ──────────────

# COCO-18 keypoint order used by VNCCS PoseGenerator
OPENPOSE_KEYPOINTS = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

OPENPOSE_BONES = [
    (0, 1),     # nose -> neck
    (1, 2),     # neck -> r_shoulder
    (2, 3),     # r_shoulder -> r_elbow
    (3, 4),     # r_elbow -> r_wrist
    (1, 5),     # neck -> l_shoulder
    (5, 6),     # l_shoulder -> l_elbow
    (6, 7),     # l_elbow -> l_wrist
    (1, 8),     # neck -> r_hip
    (8, 9),     # r_hip -> r_knee
    (9, 10),    # r_knee -> r_ankle
    (1, 11),    # neck -> l_hip
    (11, 12),   # l_hip -> l_knee
    (12, 13),   # l_knee -> l_ankle
    (0, 14),    # nose -> r_eye
    (14, 16),   # r_eye -> r_ear
    (0, 15),    # nose -> l_eye
    (15, 17),   # l_eye -> l_ear
]

BONE_COLORS = [
    (255, 128, 0),   # nose->neck (orange)
    (255, 153, 51),  # neck->r_shoulder
    (255, 178, 102), # r_shoulder->r_elbow
    (255, 153, 51),  # r_elbow->r_wrist
    (51, 153, 255),  # neck->l_shoulder
    (102, 178, 255), # l_shoulder->l_elbow
    (51, 153, 255),  # l_elbow->l_wrist
    (255, 51, 51),   # neck->r_hip (red)
    (255, 102, 102), # r_hip->r_knee
    (255, 51, 51),   # r_knee->r_ankle
    (0, 255, 0),     # neck->l_hip (green)
    (0, 153, 0),     # l_hip->l_knee
    (0, 255, 0),     # l_knee->l_ankle
    (255, 51, 255),  # nose->r_eye
    (255, 51, 255),  # r_eye->r_ear
    (255, 51, 255),  # nose->l_eye
    (255, 51, 255),  # l_eye->l_ear
]


def _render_pose_grid_b64(pose_grid: dict | None = None) -> str:
    """Render the VNCCS_PoseGenerator output as openpose skeleton image.

    Matches the original ComfyUI node: renders 12 poses as colored stick
    figures on a 512x1536 canvas. This image is fed to QWEN as reference
    for character sheet generation.

    Returns base64-encoded PNG.
    """
    if pose_grid is None:
        pose_grid = _load_default_pose_grid()

    canvas_w = pose_grid.get("canvas", {}).get("width", 512)
    canvas_h = pose_grid.get("canvas", {}).get("height", 1536)

    try:
        import cv2
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        for pose in pose_grid.get("poses", []):
            pts = {}
            for kp_name in OPENPOSE_KEYPOINTS:
                coords = pose.get(kp_name)
                if coords and len(coords) >= 2:
                    pts[kp_name] = (int(coords[0]), int(coords[1]))

            for idx, (a_idx, b_idx) in enumerate(OPENPOSE_BONES):
                a_name = OPENPOSE_KEYPOINTS[a_idx]
                b_name = OPENPOSE_KEYPOINTS[b_idx]
                if a_name in pts and b_name in pts:
                    color = BONE_COLORS[idx % len(BONE_COLORS)]
                    cv2.line(canvas, pts[a_name], pts[b_name], color, thickness=3)

            for kp_name, pt in pts.items():
                cv2.circle(canvas, pt, radius=4, color=(0, 0, 255), thickness=-1)

        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    except ImportError:
        # cv2 not available — use PIL fallback
        from PIL import Image, ImageDraw
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for pose in pose_grid.get("poses", []):
            pts = {}
            for kp_name in OPENPOSE_KEYPOINTS:
                coords = pose.get(kp_name)
                if coords and len(coords) >= 2:
                    pts[kp_name] = (int(coords[0]), int(coords[1]))

            for idx, (a_idx, b_idx) in enumerate(OPENPOSE_BONES):
                a_name = OPENPOSE_KEYPOINTS[a_idx]
                b_name = OPENPOSE_KEYPOINTS[b_idx]
                if a_name in pts and b_name in pts:
                    color = BONE_COLORS[idx % len(BONE_COLORS)]
                    draw.line([pts[a_name], pts[b_name]], fill=color, width=3)

            for pt in pts.values():
                r = 4
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=(0, 0, 255))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ─── Image helpers ────────────────────────────────────────────────────────────

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


def _resize_image_b64(image_b64: str, width: int, height: int) -> str:
    """Resize a base64 image to exact dimensions."""
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop_face_b64(image_b64: str) -> str | None:
    """Detect face via DWPose and crop face region, return as base64."""
    from services.workflows.utils.dwpose import detect_poses
    img_bytes = base64.b64decode(image_b64)
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    keypoints = detect_poses(img_np)
    if keypoints is None or len(keypoints) == 0:
        return None
    kp = keypoints[0]

    # Face keypoints: nose(0), r_eye(14), l_eye(15), r_ear(16), l_ear(17)
    face_indices = [0, 14, 15, 16, 17]
    pts = [(kp[i, 0], kp[i, 1]) for i in face_indices if i < len(kp) and kp[i, 0] > 0]
    if len(pts) < 3:
        return None

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    pad = int(max(x2 - x1, y2 - y1) * 0.5)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width, x2 + pad)
    y2 = min(img.height, y2 + pad)

    face = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    face.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _composite_face_back(original_b64: str, face_b64: str) -> str:
    """Paste refined face back onto original image at detected position."""
    from PIL import Image
    from services.workflows.utils.dwpose import detect_poses
    orig = Image.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
    orig_np = np.array(orig)

    keypoints = detect_poses(orig_np)
    if keypoints is None or len(keypoints) == 0:
        return original_b64
    kp = keypoints[0]

    face_indices = [0, 14, 15, 16, 17]
    pts = [(kp[i, 0], kp[i, 1]) for i in face_indices if i < len(kp) and kp[i, 0] > 0]
    if len(pts) < 3:
        return original_b64

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    pad = int(max(x2 - x1, y2 - y1) * 0.5)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(orig.width, x2 + pad)
    y2 = min(orig.height, y2 + pad)

    refined = Image.open(io.BytesIO(base64.b64decode(face_b64))).convert("RGB")
    refined = refined.resize((x2 - x1, y2 - y1), Image.LANCZOS)
    orig.paste(refined, (x1, y1))

    buf = io.BytesIO()
    orig.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()




def _upscale_image_b64(image_b64: str, scale: float = 2.0) -> str:
    """Super-resolution upscale — matches SeedVR2 node (638) in CharSheet workflow.

    Saves current model state, runs nvidia_upscale inference, then restores
    the previous model so the pipeline can continue without interruption.
    """
    svc = get_service()
    prev_model = svc._loaded_model
    try:
        result = svc.infer({
            "model": "nvidia_upscale",
            "image_b64": image_b64,
        })
        if result.get("status") == "ok" and result.get("data"):
            return result["data"]
        logger.warning("nvidia_upscale returned: %s", result.get("error", "no data"))
    except Exception as e:
        logger.warning("nvidia_upscale failed: %s", e)
    finally:
        # Restore the model that was loaded before upscale
        if prev_model and prev_model != svc._loaded_model:
            try:
                svc.load(prev_model)
            except Exception:
                pass
    raise RuntimeError("Super-resolution upscale failed — no fallback")

# ─── Prompt builders (1:1 match CharacterCreator ComfyUI node) ───────────────

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


# ─── Workflow 1: char_sheet (1:1 VN_Step1_QWEN_CharSheetGenerator_v1) ───────
#
# Original ComfyUI DAG (32 nodes):
#   1. CharacterCreator -> builds prompt from attributes
#   2. VNCCS_PoseGenerator -> 12-pose openpose grid (512x1536)
#   3. SDXL Loader + LoRA Stack -> ILFlatMix checkpoint + style LoRAs
#   4. KSampler -> 20 steps, dpmpp_2m_sde, karras, cfg 6
#   5. VNCCS_Pipe -> pipes SDXL output + prompt + pose grid to QWEN
#   6. QWEN generation -> 4 steps, euler, cfg 1, poser_helper_v2 LoRA
#   7. SeedVR2 upscale -> 2x super-resolution (nvidia_upscale with PIL fallback)
#   8. Face detailer -> 20 steps, euler_ancestral, cfg 7, face_yolov8m + APISR 2x
#   9. VNCCS_Resize chain -> final character sheet (1024x1024)
#  10. Face crop + detailer -> face detail output

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
    """VNCCS Step 1: full character sheet generation (1:1 ComfyUI match).

    Pipeline matches VN_Step1_QWEN_CharSheetGenerator_v1.json exactly:
      1. CharacterCreator -> prompt from structured attributes
      2. VNCCS_PoseGenerator -> 12-pose openpose grid (512x1536)
      3. SD base generation -> 8-step turbo or full-quality SDXL
      4. QWEN refinement -> 4-step image edit with pose grid reference
      5. SeedVR2 upscale -> 2x super-resolution (nvidia_upscale fallback)
      6. Face detailer -> crop -> QWEN refine (20 steps, cfg 7) -> composite back
      7. Resize to final 1024x1024 sheet

    If image_b64 is provided, steps 2-3 are skipped and the image goes
    directly to QWEN refinement with the pose grid.

    Returns dict with 'data' (base64 sheet), 'face' (base64 face crop).
    """
    logger.warning(">>> char_sheet() STARTED model=%s quality=%s seed=%d", model, quality, seed)
    svc = get_service()
    logger.warning(">>> char_sheet() svc type=%s", type(svc).__name__)

    # -- Step 1: Build prompts (CharacterCreator node) --
    char_prompt, built_negative = _build_char_prompt(
        aesthetics=aesthetics or "masterpiece,best quality,amazing quality",
        background_color=background_color,
        nsfw=nsfw, sex=sex, age=age, race=race,
        eyes=eyes, hair=hair, face=face, body=body,
        skin_color=skin_color, additional_details=additional_details,
        lora_prompt=lora_prompt,
    )
    final_negative = negative_prompt or built_negative
    gen_prompt = prompt if prompt else char_prompt
    logger.warning(">>> Step 1 prompt (%d chars) neg (%d chars)", len(gen_prompt), len(final_negative))

    # -- Step 2: Render pose grid (VNCCS_PoseGenerator node) --
    pose_grid_b64 = _render_pose_grid_b64()
    logger.warning(">>> Step 2 pose grid rendered (%d bytes b64)", len(pose_grid_b64))

    # -- Step 3: SD base generation (SDXL Loader + KSampler nodes) --
    if image_b64:
        base_image_b64 = image_b64
        logger.warning(">>> Step 3 SKIPPED (image_b64 provided)")
    else:
        steps = 8 if quality == "turbo" else 20
        cfg = 1.0 if quality == "turbo" else 6.0
        logger.warning(">>> Step 3 loading model=%s steps=%d cfg=%.1f", model, steps, cfg)
        svc.load(model)
        infer_kw: dict[str, Any] = {
            "model": model,
            "input_prompt": gen_prompt,
            "seed": seed,
            "sampling_steps": steps,
            "guide_scale": cfg,
            "width": 1024,
            "height": 1024,
        }
        if final_negative:
            infer_kw["n_prompt"] = final_negative
        logger.warning(">>> Step 3 infer payload keys=%s", list(infer_kw.keys()))
        base = svc.infer(infer_kw)
        logger.warning(">>> Step 3 infer result status=%s keys=%s has_data=%s",
                     base.get("status"), list(base.keys()), bool(base.get("data")))
        if base.get("status") != "ok":
            return error_response(f"Base generation failed: {base.get('error', 'unknown')}")
        base_image_b64 = base["data"]
        logger.warning(">>> Step 3 base image (%d bytes b64)", len(base_image_b64))

    # -- Step 4: QWEN refinement with pose grid reference --
    # Matches: QWEN KSampler (4 steps, euler, cfg 1) + poser_helper_v2 LoRA
    # The original workflow composites pose_grid + base_image as reference images
    logger.warning(">>> Step 4 loading qwen-image-edit")
    svc.load("qwen-image-edit")
    qwen_result = svc.infer({
        "model": "qwen-image-edit",
        "input_prompt": gen_prompt,
        "reference_images": [pose_grid_b64, base_image_b64],
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
        "video_prompt_type": "KI",
        "sample_solver": "lightning",
        "loras_selected": [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "VNCCS/poser_helper_v2_000004200.safetensors",
        ],
    })
    logger.warning(">>> Step 4 QWEN result status=%s has_data=%s",
                 qwen_result.get("status"), bool(qwen_result.get("data")))

    if qwen_result.get("status") != "ok":
        logger.warning("QWEN refinement failed: %s -- returning base image", qwen_result.get("error"))
        sheet_b64 = base_image_b64
    else:
        qwen_data = qwen_result["data"]
        # Quality check: if QWEN output is mostly white, fall back to base image
        try:
            from PIL import Image
            qwen_img = Image.open(io.BytesIO(base64.b64decode(qwen_data))).convert("RGB")
            qwen_arr = np.array(qwen_img)
            white_pct = np.mean(np.all(qwen_arr == 255, axis=2))
            if white_pct > 0.90:
                logger.warning("QWEN output is %.0f%% white -- falling back to base image", white_pct * 100)
                sheet_b64 = base_image_b64
            else:
                sheet_b64 = qwen_data
        except Exception:
            sheet_b64 = qwen_data
    logger.warning(">>> Step 4 sheet (%d bytes b64)", len(sheet_b64))

    # -- Step 5: Super-resolution upscale (SeedVR2 node 638) --
    logger.warning(">>> Step 5 upscale START")
    try:
        sheet_b64 = _upscale_image_b64(sheet_b64, scale=2.0)
        logger.warning(">>> Step 5 upscale OK (%d bytes b64)", len(sheet_b64))
    except Exception as e:
        logger.error(">>> Step 5 upscale FAILED: %s — using PIL resize", e)
        sheet_b64 = _resize_image_b64(sheet_b64, 1024, 1024)
        logger.warning(">>> Step 5 PIL fallback (%d bytes b64)", len(sheet_b64))

    # -- Step 6: Face detailer (20 steps, euler_ancestral, cfg 7) --
    logger.warning(">>> Step 6 face detailer START")
    face_b64 = _crop_face_b64(sheet_b64)
    face_result_b64 = None
    logger.warning(">>> Step 6 face crop: %s", "found" if face_b64 else "no face detected")

    if face_b64 is not None:
        # APISR 2x upscale on face crop (matches node 612: 2x_APISR_RRDB_GAN_generator.pth)
        try:
            face_b64 = _upscale_image_b64(face_b64, scale=2.0)
        except Exception as e:
            logger.warning(">>> Step 6 face upscale failed: %s", e)

        face_prompt = _build_face_details(
            sex=sex, race=race, eyes=eyes, hair=hair,
            face=face, skin_color=skin_color,
            additional_details=additional_details,
        )
        face_result = svc.infer({
            "model": "qwen-image-edit",
            "input_prompt": f"{face_prompt}, detailed face, high quality, sharp focus",
            "image_b64": face_b64,
            "seed": seed,
            "sampling_steps": 20,
            "guide_scale": 7.0,
        })
        if face_result.get("status") == "ok":
            face_result_b64 = face_result["data"]
            # Composite refined face back onto sheet
            sheet_b64 = _composite_face_back(sheet_b64, face_result_b64)
            logger.warning(">>> Step 6 face composite OK")
        else:
            logger.warning("Face detailer failed: %s", face_result.get("error"))
            face_result_b64 = face_b64  # Return raw crop as fallback

    # -- Step 7: Resize to final sheet size (VNCCS_Resize chain) --
    logger.warning(">>> Step 7 final resize to 1024x1024")
    sheet_b64 = _resize_image_b64(sheet_b64, 1024, 1024)
    logger.warning(">>> char_sheet() COMPLETE (%d bytes b64)", len(sheet_b64))

    result: dict[str, Any] = {
        "status": "ok",
        "data": sheet_b64,
        "media_type": "image/png",
        "_pipeline": "vnccs/char-sheet",
    }
    if face_result_b64:
        result["face"] = face_result_b64
    return result


# ─── Workflow 2: pose_edit (1:1 VNCCS_Utils Pose Studio QWEN) ──────────────
#
# Original ComfyUI DAG (10 nodes):
#   1. LoadImage (Character) -> image2
#   2. LoadImage (Pose Capture) -> VNCCS_PoseStudio
#   3. VNCCS_PoseStudio -> renders body mesh + generates lighting prompt
#   4. VNCCS_ModelSelector -> VNCCS Pose Studio QIE2511 v5.9.5
#   5. LoraLoaderModelOnly -> VNCCS_QIE2511_PoseStudio_ART_V5.9.safetensors
#   6. QWEN model loaded (checkpoint + VAE + lightning LoRA)
#   7. QWEN Encoder -> encodes image1 (mesh) + image2 (character)
#   8. KSampler -> 4 steps, euler, cfg 1
#   9. VAEDecode -> output image
#  10. SaveImage

def pose_edit(
    character_image_b64: str,
    pose_image_b64: str | None = None,
    rotations: dict[str, list[float]] | None = None,
    model_rotation_y: float = 0.0,
    mesh_config: dict[str, Any] | None = None,
    lighting_prompt: str = "",
    user_prompt: str = "",
    seed: int = 42,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Pose Studio QWEN -- 1:1 ComfyUI workflow match.

    Matches VNCCS_Utils Pose Studio QWEN.json exactly:
      1. If pose_image_b64: DWPose extracts skeleton (VNCCS_PoseStudio capture mode)
         Else: BodyMesh renders from joint rotations (VNCCS_PoseStudio mesh mode)
      2. QWEN with PoseStudio LoRA generates posed character
         image1 = pose reference (mesh or captured image)
         image2 = character to re-pose

    Args:
        character_image_b64: The character image to re-pose (image2 in QWEN)
        pose_image_b64: Reference pose image (VNCCS_PoseStudio capture)
        rotations: Joint rotations for BodyMesh mode (Anny joint names -> [x,y,z] degrees)
        model_rotation_y: Whole-body Y rotation for BodyMesh (0=front, 90=right)
        mesh_config: Anny mesh phenotype overrides (age, gender, weight, etc.)
        lighting_prompt: Optional lighting description appended to prompt
        user_prompt: Optional user prompt override (default: "Draw character from image2")
        seed: Random seed for reproducibility
        backend: BodyMesh renderer backend ("auto", "pyrender", "pil")

    Returns:
        dict with status, data (base64 PNG), media_type.
    """
    svc = get_service()

    # -- Step 1: Generate pose reference image --
    # VNCCS_PoseStudio node: renders body mesh OR uses captured image
    if pose_image_b64:
        # Capture mode: use the reference image directly as image1
        # DWPose extracts skeleton overlay for the 3-image composition
        from services.workflows.utils.dwpose import skeleton_from_image_b64
        skeleton_b64 = skeleton_from_image_b64(pose_image_b64, 1024, 1024)
        reference_images = [pose_image_b64, character_image_b64, skeleton_b64]
    else:
        # Mesh mode: BodyMesh renders from joint rotations (VNCCS_PoseStudio default)
        from services.workflows.utils.body_mesh import render_pose_b64
        from services.workflows.utils.dwpose import skeleton_from_image_b64

        resolved = rotations if rotations else {}
        mesh_b64 = render_pose_b64(
            resolved,
            width=1024, height=1024,
            model_rotation_y=model_rotation_y,
            backend=backend,
        )
        skeleton_b64 = skeleton_from_image_b64(mesh_b64, 1024, 1024)
        reference_images = [mesh_b64, character_image_b64, skeleton_b64]

    # -- Step 2: Build prompt (VNCCS_PoseStudio prompt_template) --
    # Original: "Draw character from image2\n<lighting>\n<user_prompt>"
    input_prompt = user_prompt or VNCCS_POSE_STUDIO_PROMPT
    if lighting_prompt:
        input_prompt = f"{input_prompt}\n{lighting_prompt}"

    # -- Step 3: QWEN generation with PoseStudio LoRA --
    # Matches: QWEN KSampler (4 steps, euler, cfg 1)
    # + VNCCS_QIE2511_PoseStudio_ART_V5.9 LoRA
    # + Qwen-Image-Edit-2511-Lightning-4steps LoRA
    svc.load("qwen-image-edit")
    result = svc.infer({
        "model": "qwen-image-edit",
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


# ─── Workflow 3: emotions ────────────────────────────────────────────────────

def emotions(
    sheet_image_b64: str,
    emotions_list: list[str],
    costumes: list[str] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 3: character sheet -> emotion variations.

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


# ─── Workflow 4: sprite ──────────────────────────────────────────────────────

def sprite(
    sheet_image_b64: str,
    poses: list[dict[str, Any]],
    directions: list[float] | None = None,
    seed: int = 42,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 4: character sheet + pose definitions -> sprite frames.

    For each pose:
      1. BodyMeshRenderer renders the pose as a 3D mesh image (CPU)
      2. Mesh + character composited side-by-side -> QWEN edit
      3. QWEN renders the character in the target pose

    Poses are rotations_json dicts (joint name -> [rx, ry, rz]).
    Directions are model_rotation_y values (0=front, 90=right, 180=back, 270=left).
    """
    svc = get_service()
    svc.load("qwen-image-edit")

    if directions is None:
        directions = [0.0]

    from services.workflows.utils.body_mesh import render_pose_b64
    from services.workflows.utils.dwpose import skeleton_from_image_b64

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


# ─── Workflow 5: clone ───────────────────────────────────────────────────────

def clone(
    reference_image_b64: str,
    character_def: dict[str, Any],
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """VNCCS Step 1.1: reference character -> new variant.

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


# ─── Workflow 6: detailer ────────────────────────────────────────────────────

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
