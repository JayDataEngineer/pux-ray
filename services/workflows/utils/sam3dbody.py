"""SAM3DBody 3D pose extraction from 2D images.

Uses SAM3DBody (DINOv3 backbone + MHR decoder) to extract joint rotations,
body shape, and 3D keypoints from a single image. Outputs rotation dicts
compatible with Anny body mesh renderer.

Usage:
    from services.workflows.utils.sam3dbody import extract_pose_3d

    rotations = extract_pose_3d(image_b64)
    # rotations is a dict like {"right_shoulder": [x, y, z], ...}
"""
import os
import sys
from pathlib import Path

import numpy as np

# ── SAM3DBody model path resolution ──────────────────────────────────

_COMFYUI_ROOT = Path(os.environ.get(
    "COMFYUI_ROOT",
    str(Path(__file__).resolve().parent.parent.parent.parent.parent / "infra" / "repos" / "ComfyUI"),
))
_SAM3D_NODE = _COMFYUI_ROOT / "custom_nodes" / "ComfyUI-SAM3DBody_utills"

# ── Anny joint mapping (from body_mesh.py) ────────────────────────────
# Maps human-readable names to Anny bone indices (subset used by renderer)

_ANNY_JOINTS = {
    "spine": 44, "spine01": 44, "spine02": 44, "spine03": 47,
    "pelvis": 0, "neck": 102, "head": 103,
    "right_shoulder": 51, "right_upper_arm": 51,
    "right_forearm": 52, "right_hand": 53,
    "left_shoulder": 57, "left_upper_arm": 57,
    "left_forearm": 58, "left_hand": 59,
    "right_thigh": 10, "right_leg": 12, "right_foot": 14,
    "left_thigh": 6, "left_leg": 8, "left_foot": 9,
}

# MHR outputs ~70 joints. Map MHR joint indices to Anny joint names.
# Based on mhr70.py metadata in SAM3DBody.
_MHR_TO_ANNY = {
    0: "pelvis",
    1: "spine",
    2: "spine01",
    3: "spine02",
    4: "spine03",
    5: "neck",
    6: "head",
    7: "left_shoulder",
    8: "left_upper_arm",
    9: "left_forearm",
    10: "left_hand",
    11: "right_shoulder",
    12: "right_upper_arm",
    13: "right_forearm",
    14: "right_hand",
    15: "left_thigh",
    16: "left_leg",
    17: "left_foot",
    18: "right_thigh",
    19: "right_leg",
    20: "right_foot",
}

_estimator = None


def _get_estimator():
    """Lazy-load SAM3DBody model (GPU, one-time)."""
    global _estimator
    if _estimator is not None:
        return _estimator

    if not _SAM3D_NODE.exists():
        raise FileNotFoundError(
            f"SAM3DBody not found at {_SAM3D_NODE}. "
            "Run 'task setup:nodes' to clone it."
        )

    sys.path.insert(0, str(_SAM3D_NODE / "nodes"))

    from sam_3d_body.build_models import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    checkpoint = os.environ.get(
        "SAM3DBODY_CHECKPOINT",
        str(_SAM3D_NODE / "models" / "sam_3d_body.safetensors"),
    )
    mhr_path = os.environ.get(
        "MHR_MODEL_PATH",
        str(_SAM3D_NODE / "models" / "mhr"),
    )

    model, cfg, _ = load_sam_3d_body(
        checkpoint_path=checkpoint,
        device="cuda",
        mhr_path=mhr_path,
    )
    _estimator = SAM3DBodyEstimator(model, cfg)
    return _estimator


def _global_rots_to_anny(output: dict) -> dict[str, list[float]]:
    """Convert MHR global rotations to Anny-compatible rotation dict.

    MHR outputs rotations as axis-angle vectors per joint.
    Convert to degree tuples for the Anny renderer.
    """
    global_rots = output.get("pred_global_rots")
    if global_rots is None:
        raise ValueError("SAM3DBody output missing pred_global_rots")

    rotations = {}
    for mhr_idx, anny_name in _MHR_TO_ANNY.items():
        if mhr_idx < len(global_rots):
            rot = global_rots[mhr_idx]  # axis-angle [3]
            # Convert axis-angle to degrees
            angle = np.linalg.norm(rot)
            if angle > 1e-6:
                axis = rot / angle
                # Simplified: just use the raw rotation components as euler-ish degrees
                degrees = np.degrees(rot).tolist()
            else:
                degrees = [0.0, 0.0, 0.0]
            rotations[anny_name] = [round(d, 2) for d in degrees]

    return rotations


def extract_pose_3d(image_b64: str) -> dict[str, list[float]]:
    """Extract 3D pose from a base64-encoded image via SAM3DBody.

    Args:
        image_b64: Base64-encoded source image.

    Returns:
        Rotation dict compatible with Anny body mesh renderer,
        e.g. {"right_shoulder": [0, 0, 45], "left_elbow": [0, -30, 0], ...}
    """
    import base64
    import io
    import cv2

    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image from base64")

    estimator = _get_estimator()
    results = estimator.process_one_image(img, inference_type="body")

    if not results:
        raise ValueError("SAM3DBody detected no human in the image")

    # Use first detected person
    return _global_rots_to_anny(results[0])
