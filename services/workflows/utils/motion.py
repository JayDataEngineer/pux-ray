"""Motion utilities — HY-Motion NPZ converter + shared motion helpers.

HY-Motion outputs 22-joint 6D rotation representations in Y-up convention.
This converts them to Anny-compatible per-frame Euler angle dicts in Z-up.

Usage:
    from services.workflows.utils.motion import extract_keyframes, npz_bytes_to_keyframes

    # From NPZ file:
    keyframes = extract_keyframes("path/to/motion.npz", num_keyframes=6)

    # From NPZ bytes (e.g., from Wan2GPService response):
    keyframes = npz_bytes_to_keyframes(npz_bytes, num_keyframes=6)

    # Each keyframe is: {"right_shoulder": [x, y, z], "left_shoulder": [...], ...}
    # Each keyframe can be passed directly to render_pose() or render_pose_b64().
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

# HY-Motion joint index → Anny bone name
HYMOTION_TO_ANNY: dict[int, str] = {
    0: "pelvis",
    3: "spine",
    6: "spine",
    9: "spine_03",
    12: "neck",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
    1: "left_hip",
    2: "right_hip",
    4: "left_knee",
    5: "right_knee",
    7: "left_ankle",
    8: "right_ankle",
}

SPINE_INDICES = [3, 6]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation → 3x3 matrix via Gram-Schmidt (Zhou et al., CVPR 2019)."""
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """3x3 matrix → XYZ Euler degrees. R = Rz * Ry * Rx convention."""
    sy = -R[..., 2, 0]
    cy = np.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2 + 1e-10)
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y = np.arctan2(sy, cy)
    z = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.degrees(np.stack([x, y, z], axis=-1))


def convert_yup_to_zup(euler_xyz: np.ndarray) -> np.ndarray:
    """Swap Y↔Z to convert from Y-up to Z-up convention."""
    return np.stack([euler_xyz[..., 0], euler_xyz[..., 2], euler_xyz[..., 1]], axis=-1)


def _rot6d_to_keyframes(rot6d: np.ndarray, num_keyframes: int) -> list[dict[str, list[float]]]:
    """Convert (frames, 22, 6) rotation array to list of Anny rotation dicts."""
    total_frames = rot6d.shape[0]
    if num_keyframes >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int).tolist()

    keyframes = []
    for fi in indices:
        frame_rot6d = rot6d[fi]
        frame_matrices = rot6d_to_matrix(frame_rot6d)
        rotations: dict[str, list[float]] = {}

        for hym_idx, anny_name in HYMOTION_TO_ANNY.items():
            R = frame_matrices[hym_idx]
            euler = matrix_to_euler_xyz(R)
            euler_zup = convert_yup_to_zup(euler)

            if anny_name in rotations:
                existing = np.array(rotations[anny_name])
                rotations[anny_name] = ((existing + euler_zup) / 2).tolist()
            else:
                rotations[anny_name] = euler_zup.tolist()

        keyframes.append(rotations)

    return keyframes


def extract_keyframes(npz_path: str | Path, num_keyframes: int = 6) -> list[dict[str, list[float]]]:
    """Extract keyframe rotations from HY-Motion NPZ file.

    Args:
        npz_path: Path to HY-Motion NPZ file.
        num_keyframes: Number of evenly-spaced keyframes to extract.

    Returns:
        List of rotation dicts, each compatible with render_pose().
    """
    data = np.load(str(npz_path), allow_pickle=True)
    return _rot6d_to_keyframes(data["rot6d"], num_keyframes)


def npz_bytes_to_keyframes(npz_bytes: bytes, num_keyframes: int = 6) -> list[dict[str, list[float]]]:
    """Extract keyframes from in-memory NPZ bytes.

    Args:
        npz_bytes: Raw NPZ file bytes (e.g., from Wan2GPService response).
        num_keyframes: Number of evenly-spaced keyframes to extract.

    Returns:
        List of rotation dicts, each compatible with render_pose().
    """
    buf = io.BytesIO(npz_bytes)
    data = np.load(buf, allow_pickle=True)
    return _rot6d_to_keyframes(data["rot6d"], num_keyframes)


def npz_b64_to_keyframes(npz_b64: str, num_keyframes: int = 6) -> list[dict[str, list[float]]]:
    """Extract keyframes from base64-encoded NPZ bytes.

    Args:
        npz_b64: Base64-encoded NPZ string.
        num_keyframes: Number of evenly-spaced keyframes to extract.

    Returns:
        List of rotation dicts, each compatible with render_pose().
    """
    import base64
    return npz_bytes_to_keyframes(base64.b64decode(npz_b64), num_keyframes)
