"""BodyMesh — Skeleton pose rendering from joint rotation parameters.

Renders a 23-joint humanoid skeleton from rotation angles using matplotlib.
No SMPL/SOMA model file needed.
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

_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (0, 15), (15, 16), (16, 17),
    (5, 18), (5, 19), (5, 20), (5, 21), (5, 22),
]

_DEFAULT_POSE = np.array([
    [0.0, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.24, 0.0],
    [0.0, 0.36, 0.0], [0.0, 0.42, 0.0], [0.0, 0.50, 0.0],
    [0.08, 0.36, 0.0], [0.22, 0.30, 0.0], [0.36, 0.24, 0.0],
    [-0.08, 0.36, 0.0], [-0.22, 0.30, 0.0], [-0.36, 0.24, 0.0],
    [0.04, -0.10, 0.0], [0.04, -0.36, 0.0], [0.04, -0.62, 0.0],
    [-0.04, -0.10, 0.0], [-0.04, -0.36, 0.0], [-0.04, -0.62, 0.0],
    [0.02, 0.52, 0.0], [-0.02, 0.52, 0.0], [0.0, 0.48, 0.02],
    [0.04, 0.50, 0.0], [-0.04, 0.50, 0.0],
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

        try:
            rots = json.loads(rotations) if isinstance(rotations, str) else rotations
        except (json.JSONDecodeError, TypeError):
            rots = []

        joints = _DEFAULT_POSE.copy()

        if model_rotation_y:
            angle = math.radians(model_rotation_y)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in range(len(joints)):
                x, z = joints[i, 0], joints[i, 2]
                joints[i, 0] = x * cos_a - z * sin_a
                joints[i, 2] = x * sin_a + z * cos_a

        if rots:
            self._apply_rotations(joints, rots)

        fig, ax = plt.subplots(figsize=(5, 8), dpi=96)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        for parent, child in _BONE_PAIRS:
            p1, p2 = joints[parent], joints[child]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e94560",
                    linewidth=3, alpha=0.8, solid_capstyle="round")

        for i, (x, y, z) in enumerate(joints):
            size = 40 if i in (5, 8, 11, 14, 17) else 25
            alpha_val = 0.9 if i == 5 else 0.7
            ax.scatter(x, y, s=size, c="#0f3460", edgecolors="#e94560",
                       linewidths=1.5, zorder=5, alpha=alpha_val)

        margin = 0.15
        xs, ys = joints[:, 0], joints[:, 1]
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
        for i, angle_deg in enumerate(rots):
            if i >= len(joints):
                break
            angle = math.radians(float(angle_deg))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            parent_idx = None
            for p, c in _BONE_PAIRS:
                if c == i:
                    parent_idx = p
                    break
            if parent_idx is not None:
                px, py = joints[parent_idx, 0], joints[parent_idx, 1]
                rx, ry = joints[i, 0] - px, joints[i, 1] - py
                new_rx = rx * cos_a - ry * sin_a
                new_ry = rx * sin_a + ry * cos_a
                joints[i, 0] = px + new_rx
                joints[i, 1] = py + new_ry
