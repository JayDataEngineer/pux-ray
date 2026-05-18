"""Face detailer — DWPose keypoints + QWEN-Edit crop-refine-composite.

Detects face bbox from DWPose keypoints, crops the face region, runs
QWEN-Edit to refine it, then pastes the refined face back into the
original image. No Wan2GPService changes needed.

Usage:
    from services.workflows.utils.detailer import refine_faces
    result = refine_faces(image_b64)
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np

from services.workflows.base import get_service

logger = logging.getLogger(__name__)

# COCO-18 face keypoints
NOSE, R_EYE, L_EYE, R_EAR, L_EAR = 0, 14, 15, 16, 17
FACE_INDICES = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR]
PAD_FACTOR = 0.5


def _detect_face_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    from services.workflows.utils.dwpose import detect_poses
    keypoints = detect_poses(image)
    if keypoints is None or len(keypoints) == 0:
        return None
    kp = keypoints[0]
    pts = [(kp[i, 0], kp[i, 1]) for i in FACE_INDICES
           if i < len(kp) and kp[i, 0] > 0 and kp[i, 1] > 0]
    if len(pts) < 3:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2, y1, y2 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20:
        return None
    pad = int(max(w, h) * PAD_FACTOR)
    return (
        max(0, x1 - pad), max(0, y1 - pad),
        min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad),
    )


def refine_faces(
    image_b64: str,
    prompt: str = "detailed face, high quality, sharp focus, clear eyes",
    seed: int = 42,
) -> dict[str, Any]:
    """Refine face regions via DWPose detection → crop → QWEN edit → composite.

    Pipeline:
      1. DWPose detects face from DWPose keypoints (no extra model needed)
      2. Face region cropped with 50% padding
      3. QWEN-Edit refines the cropped face region
      4. Refined face pasted back into the original image

    Falls back to full-image QWEN edit if no face detected.
    """
    from PIL import Image as PILImage

    svc = get_service()
    svc.load("qwen-image-edit")

    img_bytes = base64.b64decode(image_b64)
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    bbox = _detect_face_bbox(img_np)
    if bbox is None:
        logger.info("No face detected — full-image QWEN edit")
        return svc.infer({
            "input_prompt": prompt,
            "image_b64": image_b64,
            "seed": seed,
            "sampling_steps": 4,
            "guide_scale": 1.0,
        })

    x1, y1, x2, y2 = bbox
    logger.info("Face bbox: (%d, %d, %d, %d)", x1, y1, x2, y2)

    # Crop face region
    face = img.crop((x1, y1, x2, y2))
    face_buf = io.BytesIO()
    face.save(face_buf, format="PNG")
    face_b64 = base64.b64encode(face_buf.getvalue()).decode()

    # Refine face with QWEN
    result = svc.infer({
        "input_prompt": prompt,
        "image_b64": face_b64,
        "seed": seed,
        "sampling_steps": 4,
        "guide_scale": 1.0,
    })

    if result.get("status") != "ok":
        logger.warning("Face refinement failed: %s", result.get("error"))
        # Fallback: return original image
        return {"status": "ok", "data": image_b64, "media_type": "image/png"}

    # Decode refined face, composite back
    refined_bytes = base64.b64decode(result["data"])
    refined_face = PILImage.open(io.BytesIO(refined_bytes)).convert("RGB")
    refined_face = refined_face.resize((x2 - x1, y2 - y1), PILImage.LANCZOS)

    img.paste(refined_face, (x1, y1))
    out_buf = io.BytesIO()
    img.save(out_buf, format="PNG")
    out_b64 = base64.b64encode(out_buf.getvalue()).decode()

    return {"status": "ok", "data": out_b64, "media_type": "image/png"}
