"""SAM3DBody 3D pose extraction from 2D images.

Uses SAM3DBody (DINOv3 backbone + MHR decoder) to extract 3D pose,
body shape, and mesh vertices from a single image. Returns raw MHR
output for consumption by SOMA's pose inversion pipeline.

Usage:
    from services.workflows.utils.sam3dbody import extract_pose_3d

    result = extract_pose_3d(image_b64)
    # result["pred_vertices"] — MHR mesh (18439 verts)
    # result["pred_global_rots"] — axis-angle rotations per joint
    # result["pred_body_pose"] — shape parameters
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


def extract_pose_3d(image_b64: str) -> dict:
    """Extract 3D pose from a base64-encoded image via SAM3DBody.

    Args:
        image_b64: Base64-encoded source image.

    Returns:
        Raw MHR output dict from SAM3DBody with keys:
          - pred_vertices: (V, 3) numpy array — MHR mesh vertices
          - pred_global_rots: (J, 3) numpy array — axis-angle rotations
          - pred_body_pose: shape parameters
          - pred keypoints: 3D joint positions
          - faces: face indices (if available from estimator config)
    """
    import base64
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

    raw = results[0]
    out = {}
    for key in ("pred_vertices", "pred_global_rots", "pred_body_pose",
                "pred_keypoints_3d", "pred_cam"):
        if key in raw:
            val = raw[key]
            out[key] = val.cpu().numpy() if hasattr(val, "cpu") else np.asarray(val)
    return out
