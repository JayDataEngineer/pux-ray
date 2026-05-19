"""Pose description generator — converts SMPL joint angles to text prompts.

FluxRT has no pose conditioning API. This module analyzes SMPL body_pose
parameters and generates natural language descriptions of each frame's
body position for prompt-based rendering.
"""
from __future__ import annotations

import torch


def describe_poses(
    body_pose: torch.Tensor,
    emotion: str = "",
    stride: int = 1,
) -> list[str]:
    """Generate per-frame text descriptions of body pose.

    Args:
        body_pose: (L, 69) SMPL body pose parameters (axis-angle).
        emotion: Optional emotion tag to include.
        stride: Generate description every N frames, repeat for others.

    Returns:
        List of L pose description strings.
    """
    bp = body_pose
    if bp.is_cuda:
        bp = bp.cpu()
    bp = bp.numpy()
    L = len(bp)

    descriptions = []
    for i in range(L):
        descriptions.append(_describe_frame(bp[i], emotion))

    return descriptions


def _describe_frame(joints: "numpy.ndarray", emotion: str) -> str:
    """Describe a single frame's pose from its 69-dim body_pose vector."""
    parts = ["person"]

    if emotion:
        parts.append(f"looking {emotion}")

    # SMPL body_pose joint groups (23 joints, 3 axis-angle each):
    # 0-2: left up arm, 3-5: left low arm
    # 6-8: right up arm, 9-11: right low arm
    # 12-14: spine (1-3), 15-17: neck, 18-20: left leg up, 21-23: left leg low
    # etc.

    left_shoulder_pitch = joints[1]
    right_shoulder_pitch = joints[7]
    left_elbow = joints[4]
    right_elbow = joints[10]
    spine_1 = joints[13]

    # Arms
    if left_shoulder_pitch > 0.8 or right_shoulder_pitch > 0.8:
        parts.append("arms raised high")
    elif left_shoulder_pitch > 0.3:
        parts.append("left arm raised")
    elif right_shoulder_pitch > 0.3:
        parts.append("right arm raised")

    if left_shoulder_pitch < -0.5 and right_shoulder_pitch < -0.5:
        parts.append("arms at sides")

    # Elbows
    if abs(left_elbow) > 0.8:
        parts.append("left arm bent")
    if abs(right_elbow) > 0.8:
        parts.append("right arm bent")

    # Spine
    if spine_1 > 0.5:
        parts.append("leaning forward")
    elif spine_1 < -0.5:
        parts.append("leaning backward")

    return ", ".join(parts)
