"""Pose description generator — converts SOMA joint positions to text prompts.

FluxRT has no pose conditioning API. This module analyzes SOMA 77-joint
positions and generates natural language descriptions of each frame's
body position for prompt-based rendering.

Two input modes:
1. SOMA posed_joints (T, 77, 3) — from Kimodo (preferred)
2. SMPL body_pose (T, 69) — from GEM (legacy fallback)
"""
from __future__ import annotations

import numpy as np
import torch

# SOMA 77-joint skeleton key joint indices (from SOMASkeleton77)
# These are the major body joints used for pose description
_SOMA_PELVIS = 0
_SOMA_SPINE_1 = 1
_SOMA_SPINE_2 = 2
_SOMA_SPINE_3 = 3
_SOMA_NECK = 12
_SOMA_HEAD = 15
_SOMA_LEFT_SHOULDER = 16
_SOMA_LEFT_ELBOW = 18
_SOMA_LEFT_WRIST = 20
_SOMA_LEFT_HAND = 22
_SOMA_RIGHT_SHOULDER = 24
_SOMA_RIGHT_ELBOW = 26
_SOMA_RIGHT_WRIST = 28
_SOMA_RIGHT_HAND = 30
_SOMA_LEFT_HIP = 32
_SOMA_LEFT_KNEE = 34
_SOMA_LEFT_ANKLE = 36
_SOMA_LEFT_FOOT = 38
_SOMA_RIGHT_HIP = 40
_SOMA_RIGHT_KNEE = 42
_SOMA_RIGHT_ANKLE = 44
_SOMA_RIGHT_FOOT = 46


def describe_poses(
    posed_joints: "np.ndarray | torch.Tensor | None" = None,
    body_pose: "np.ndarray | torch.Tensor | None" = None,
    foot_contacts: "np.ndarray | None" = None,
    emotion: str = "",
    stride: int = 1,
) -> list[str]:
    """Generate per-frame text descriptions of body pose.

    Accepts either SOMA posed_joints (preferred) or SMPL body_pose (legacy).

    Args:
        posed_joints: (T, 77, 3) SOMA global joint positions (from Kimodo).
        body_pose: (T, 69) SMPL body pose parameters (from GEM, legacy).
        foot_contacts: (T, 4) foot contact booleans [L_heel, L_toe, R_heel, R_toe].
        emotion: Optional emotion tag to include.
        stride: Generate description every N frames, repeat for others.

    Returns:
        List of T pose description strings.
    """
    if posed_joints is not None:
        return _describe_from_soma(posed_joints, foot_contacts, emotion, stride)
    elif body_pose is not None:
        return _describe_from_smpl(body_pose, emotion, stride)
    else:
        raise ValueError("Must provide either posed_joints or body_pose")


def _describe_from_soma(
    posed_joints: "np.ndarray | torch.Tensor",
    foot_contacts: "np.ndarray | None",
    emotion: str,
    stride: int,
) -> list[str]:
    """Describe poses from SOMA 77-joint global positions."""
    if isinstance(posed_joints, torch.Tensor):
        posed_joints = posed_joints.cpu().numpy()
    if foot_contacts is not None and isinstance(foot_contacts, torch.Tensor):
        foot_contacts = foot_contacts.cpu().numpy()

    L = len(posed_joints)
    descriptions = []

    for i in range(L):
        if i % stride != 0 and descriptions:
            descriptions.append(descriptions[-1])
        else:
            descriptions.append(
                _describe_soma_frame(posed_joints[i], foot_contacts, emotion)
            )

    return descriptions


def _describe_soma_frame(
    joints: "np.ndarray",
    foot_contacts: "np.ndarray | None",
    emotion: str,
) -> str:
    """Describe a single frame from SOMA 77-joint positions."""
    parts = ["person"]

    if emotion:
        parts.append(f"looking {emotion}")

    pelvis = joints[_SOMA_PELVIS]
    head = joints[_SOMA_HEAD]
    l_hand = joints[_SOMA_LEFT_HAND]
    r_hand = joints[_SOMA_RIGHT_HAND]
    l_shoulder = joints[_SOMA_LEFT_SHOULDER]
    r_shoulder = joints[_SOMA_RIGHT_SHOULDER]
    l_elbow = joints[_SOMA_LEFT_ELBOW]
    r_elbow = joints[_SOMA_RIGHT_ELBOW]

    # Arm height relative to shoulder
    l_arm_height = l_hand[1] - l_shoulder[1]  # positive = above shoulder
    r_arm_height = r_hand[1] - r_shoulder[1]

    if l_arm_height > 0.2 and r_arm_height > 0.2:
        parts.append("arms raised high")
    elif l_arm_height > 0.1:
        parts.append("left arm raised")
    elif r_arm_height > 0.1:
        parts.append("right arm raised")
    elif l_arm_height < -0.3 and r_arm_height < -0.3:
        parts.append("arms at sides")

    # Elbow bend (hand forward relative to elbow)
    l_foreward = l_hand[2] - l_elbow[2]
    r_foreward = r_hand[2] - r_elbow[2]
    if abs(l_foreward) > 0.15:
        parts.append("left arm bent")
    if abs(r_foreward) > 0.15:
        parts.append("right arm bent")

    # Head tilt / lean
    head_offset = head - pelvis
    if head_offset[2] > 0.15:
        parts.append("leaning forward")
    elif head_offset[2] < -0.15:
        parts.append("leaning backward")

    # Foot contacts
    if foot_contacts is not None:
        fc = foot_contacts
        if fc[0] < 0.5 and fc[2] < 0.5:
            parts.append("both feet off ground")

    return ", ".join(parts)


def _describe_from_smpl(
    body_pose: "np.ndarray | torch.Tensor",
    emotion: str,
    stride: int,
) -> list[str]:
    """Legacy: describe poses from SMPL body_pose (69-dim axis-angle)."""
    if isinstance(body_pose, torch.Tensor):
        body_pose = body_pose.cpu().numpy()

    L = len(body_pose)
    descriptions = []
    for i in range(L):
        if i % stride != 0 and descriptions:
            descriptions.append(descriptions[-1])
        else:
            descriptions.append(_describe_smpl_frame(body_pose[i], emotion))
    return descriptions


def _describe_smpl_frame(joints: "np.ndarray", emotion: str) -> str:
    """Legacy: describe a single frame from 69-dim SMPL body_pose."""
    parts = ["person"]

    if emotion:
        parts.append(f"looking {emotion}")

    left_shoulder_pitch = joints[1]
    right_shoulder_pitch = joints[7]
    left_elbow = joints[4]
    right_elbow = joints[10]
    spine_1 = joints[13]

    if left_shoulder_pitch > 0.8 or right_shoulder_pitch > 0.8:
        parts.append("arms raised high")
    elif left_shoulder_pitch > 0.3:
        parts.append("left arm raised")
    elif right_shoulder_pitch > 0.3:
        parts.append("right arm raised")

    if left_shoulder_pitch < -0.5 and right_shoulder_pitch < -0.5:
        parts.append("arms at sides")

    if abs(left_elbow) > 0.8:
        parts.append("left arm bent")
    if abs(right_elbow) > 0.8:
        parts.append("right arm bent")

    if spine_1 > 0.5:
        parts.append("leaning forward")
    elif spine_1 < -0.5:
        parts.append("leaning backward")

    return ", ".join(parts)
