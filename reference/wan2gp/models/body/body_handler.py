"""BodyMesh — Skeleton pose rendering from joint rotation parameters.

Renders a 23-joint humanoid skeleton from rotation angles using matplotlib.
No SMPL/SOMA model file needed — pure skeleton wireframe visualization.
Supports the VNCCS pose editing workflow where Kimodo outputs joint rotations.

Available as MCP tool via the Wan2GP `run` endpoint.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from models.base_handler import BaseFamilyHandler, _make_handler_cls

# Humanoid skeleton bone connections (parent → child joint indices)
# Based on SMPL-23 joint structure (hips → spine → head, shoulders → arms, hips → legs)
_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3),  # Spine
    (3, 4), (4, 5),           # Neck → Head
    (3, 6), (6, 7), (7, 8),   # Left arm (shoulder → elbow → wrist)
    (3, 9), (9, 10), (10, 11), # Right arm
    (0, 12), (12, 13), (13, 14),  # Left leg (hip → knee → ankle)
    (0, 15), (15, 16), (16, 17),  # Right leg
    # Face/head detail
    (5, 18), (5, 19),  # Eyes
    (5, 20),            # Nose
    (5, 21), (5, 22),  # Ears
]

# Default rest-pose joint positions (front-facing, standing)
# Shape: (23, 3) for x, y, z
_DEFAULT_POSE = np.array([
    [0.0, 0.0, 0.0],   # 0: pelvis
    [0.0, 0.12, 0.0],  # 1: spine1
    [0.0, 0.24, 0.0],  # 2: spine2
    [0.0, 0.36, 0.0],  # 3: spine3
    [0.0, 0.42, 0.0],  # 4: neck
    [0.0, 0.50, 0.0],  # 5: head
    [0.08, 0.36, 0.0],  # 6: left shoulder
    [0.22, 0.30, 0.0],  # 7: left elbow
    [0.36, 0.24, 0.0],  # 8: left wrist
    [-0.08, 0.36, 0.0],  # 9: right shoulder
    [-0.22, 0.30, 0.0],  # 10: right elbow
    [-0.36, 0.24, 0.0],  # 11: right wrist
    [0.04, -0.10, 0.0],  # 12: left hip
    [0.04, -0.36, 0.0],  # 13: left knee
    [0.04, -0.62, 0.0],  # 14: left ankle
    [-0.04, -0.10, 0.0],  # 15: right hip
    [-0.04, -0.36, 0.0],  # 16: right knee
    [-0.04, -0.62, 0.0],  # 17: right ankle
    [0.02, 0.52, 0.0],  # 18: left eye
    [-0.02, 0.52, 0.0],  # 19: right eye
    [0.0, 0.48, 0.02],   # 20: nose
    [0.04, 0.50, 0.0],   # 21: left ear
    [-0.04, 0.50, 0.0],  # 22: right ear
])


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["body_mesh"]
    FAMILY = "body"
    FAMILY_INFOS = {"body_mesh": (402, "BodyMesh Skeleton Renderer")}
    MODEL_DEF = {"audio_only": False, "image_outputs": True}
    DEFAULTS = {"model_rotation_y": 0.0}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        pipeline = _BodyMeshPipeline()
        pipe_dict = {}
        return pipeline, pipe_dict


class _BodyMeshPipeline:
    """Matplotlib-based skeleton renderer from joint rotations."""

    def generate(self, *, rotations: str = "[]", model_rotation_y: float = 0.0,
                 **kw) -> dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Parse rotations JSON
        try:
            rots = json.loads(rotations) if isinstance(rotations, str) else rotations
        except (json.JSONDecodeError, TypeError):
            rots = []

        # Apply rotations to rest pose
        joints = _DEFAULT_POSE.copy()

        if model_rotation_y:
            # Rotate entire skeleton around Y axis
            angle = math.radians(model_rotation_y)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in range(len(joints)):
                x, z = joints[i, 0], joints[i, 2]
                joints[i, 0] = x * cos_a - z * sin_a
                joints[i, 2] = x * sin_a + z * cos_a

        # Apply per-joint rotations if provided
        if rots:
            self._apply_rotations(joints, rots)

        # Render to image
        fig, ax = plt.subplots(figsize=(5, 8), dpi=96)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Draw bones
        for parent, child in _BONE_PAIRS:
            p1 = joints[parent]
            p2 = joints[child]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color="#e94560", linewidth=3, alpha=0.8, solid_capstyle="round",
            )

        # Draw joints
        for i, (x, y, z) in enumerate(joints):
            size = 40 if i in (5, 8, 11, 14, 17) else 25  # larger at extremities
            alpha_val = 0.9 if i == 5 else 0.7
            ax.scatter(x, y, s=size, c="#0f3460", edgecolors="#e94560",
                       linewidths=1.5, zorder=5, alpha=alpha_val)

        # Set bounds with padding
        margin = 0.15
        xs = joints[:, 0]
        ys = joints[:, 1]
        ax.set_xlim(xs.min() - margin, xs.max() + margin)
        ax.set_ylim(ys.min() - margin, ys.max() + margin)

        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=96, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        return {"status": "success",
                "data": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "image/png"}

    @staticmethod
    def _apply_rotations(joints: np.ndarray, rots: list) -> None:
        """Apply per-joint rotation angles (simplified 1-DoF Z-rotation).

        Full SMPL rotation matrices are 3x3 per joint; this simplified
        version rotates each joint's child bones around the Z axis.
        """
        for i, angle_deg in enumerate(rots):
            if i >= len(joints):
                break
            angle = math.radians(float(angle_deg))
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Rotate this joint's local position relative to its parent
            px, py = joints[i, 0], joints[i, 1]
            # Find parent joint and apply relative rotation
            parent_idx = None
            for p, c in _BONE_PAIRS:
                if c == i:
                    parent_idx = p
                    break

            if parent_idx is not None:
                px, py = joints[parent_idx, 0], joints[parent_idx, 1]
                rx = joints[i, 0] - px
                ry = joints[i, 1] - py
                new_rx = rx * cos_a - ry * sin_a
                new_ry = rx * sin_a + ry * cos_a
                joints[i, 0] = px + new_rx
                joints[i, 1] = py + new_ry
