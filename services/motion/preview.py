"""Motion preview renderer — skeleton animation to MP4.

Renders posed joint arrays (T, J, 3) as a stick figure animation using PIL
for frame rasterization and imageio-ffmpeg for MP4 encoding.
"""
import io
import logging

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# SMPL-H kinematic tree (52 joints). Bones are (parent, child) pairs.
# Only major body bones — skip fingers for clean preview.
SMPLH_BODY_BONES = [
    (0, 1), (0, 2), (0, 3),           # pelvis -> hips, spine1
    (1, 4), (4, 7), (7, 10),          # left leg
    (2, 5), (5, 8), (8, 11),          # right leg
    (3, 6), (6, 9),                   # spine
    (9, 12),                          # spine3 -> neck
    (9, 13), (9, 14),                 # collars
    (12, 15),                         # neck -> head
    (13, 16), (16, 18), (18, 20),     # left arm
    (14, 17), (17, 19), (19, 21),     # right arm
]

# Bone colors (RGB) — left/right limb differentiation
BONE_COLORS = {
    "spine": (100, 200, 255),
    "left": (255, 130, 90),
    "right": (90, 180, 255),
    "head": (200, 200, 255),
}

_BONE_COLOR_MAP = {
    0: "spine", 1: "left", 2: "right", 3: "spine",
    4: "left", 5: "right", 6: "spine",
    7: "left", 8: "right", 9: "spine",
    10: "left", 11: "right",
    12: "spine", 13: "left", 14: "right",
    15: "head",
    16: "left", 17: "right",
    18: "left", 19: "right",
    20: "left", 21: "right",
}


def _project_3d_to_2d(joints_3d, width, height, elev_deg=20, azim_deg=30):
    """Orthographic projection with rotation for 3D effect."""
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)

    # Rotation matrix (elevation around X, azimuth around Y)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    R = Ry @ Rx

    projected = joints_3d @ R.T
    xy = projected[:, :2]

    # Normalize to image coords with padding
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    pad = 0.1
    scale = min(width * (1 - 2 * pad) / span[0], height * (1 - 2 * pad) / span[1])
    centered = (xy - (xy_min + xy_max) / 2) * scale
    px = (centered[:, 0] + width / 2).astype(int)
    py = (height / 2 - centered[:, 1]).astype(int)  # flip Y
    return px, py


def render_skeleton_frame(joints_3d, width=512, height=512, bg_color=(15, 15, 25)):
    """Render a single skeleton frame as PIL Image."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    px, py = _project_3d_to_2d(joints_3d, width, height)

    # Draw bones
    for parent, child in SMPLH_BODY_BONES:
        if parent >= len(px) or child >= len(px):
            continue
        side = _BONE_COLOR_MAP.get(parent, "spine")
        color = BONE_COLORS[side]
        draw.line([(px[parent], py[parent]), (px[child], py[child])],
                  fill=color, width=3)

    # Draw joints
    for i in range(len(px)):
        if i >= len(px):
            break
        side = _BONE_COLOR_MAP.get(i, "spine")
        color = BONE_COLORS[side]
        r = 4 if i in (0, 15) else 3
        draw.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=color)

    return img


def render_motion_to_mp4(posed_joints, fps=30, width=512, height=512):
    """Render posed joints (T, J, 3) as MP4 video, return base64 string.

    Args:
        posed_joints: numpy array of shape (T, J, 3) — joint positions per frame.
        fps: frames per second.
        width: output video width.
        height: output video height.

    Returns:
        bytes of MP4 video.
    """
    import imageio.v3 as iio

    if posed_joints.ndim != 3 or posed_joints.shape[2] != 3:
        raise ValueError(f"Expected (T, J, 3) array, got shape {posed_joints.shape}")

    frames = []
    for t in range(posed_joints.shape[0]):
        img = render_skeleton_frame(posed_joints[t], width, height)
        frames.append(np.array(img))

    buf = io.BytesIO()
    iio.imwrite(buf, frames, codec="libx264", fps=fps,
                extension=".mp4", output_params=["-pix_fmt", "yuv420p"])
    return buf.getvalue()


def render_motion_to_mp4_b64(posed_joints, fps=30):
    """Render motion to MP4 and return base64-encoded string."""
    import base64
    mp4_bytes = render_motion_to_mp4(posed_joints, fps=fps)
    return base64.b64encode(mp4_bytes).decode()
