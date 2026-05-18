"""Standalone BodyMeshRenderer — deterministic 3D body mesh → 2D image.

Extracted from comfyui-pose-director custom nodes. Renders Anny body meshes
as gray silhouettes on white backgrounds using pyrender (GPU/OpenGL) or PIL
(CPU-only fallback).

Usage:
    from services.workflows.utils.body_mesh import render_pose

    image = render_pose({"right_shoulder": [0, 0, 90]})
    image_b64 = base64.b64encode(image).decode()
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

# ─── Anny Joint Name Mapping ──────────────────────────────────────────────────
# Maps user-facing joint names to Anny's 163-bone indices.
# Unlisted bones default to identity (T-pose).

JOINT_MAP: dict[str, int] = {
    "spine": 44, "spine01": 44, "spine02": 44, "spine03": 47,
    "spine04": 47, "spine05": 47, "spine_01": 44, "spine_02": 44,
    "spine_03": 47, "pelvis": 0,
    "neck": 102, "neck01": 100, "neck02": 101, "neck03": 102, "neck_01": 102,
    "head": 103,
    "right_shoulder": 51, "right_upper_arm": 51, "right_upperarm": 51,
    "upperarm_r": 51, "right_elbow": 52, "right_forearm": 52, "lowerarm_r": 52,
    "right_wrist": 53, "hand_r": 53,
    "left_shoulder": 77, "left_upper_arm": 77, "left_upperarm": 77,
    "upperarm_l": 77, "left_elbow": 78, "left_forearm": 78, "lowerarm_l": 78,
    "left_wrist": 79, "hand_l": 79,
    "right_hip": 1, "right_thigh": 1, "right_upper_leg": 1,
    "right_upperleg": 1, "thigh_r": 1, "right_knee": 4, "right_shin": 4,
    "calf_r": 4, "right_ankle": 5, "foot_r": 5,
    "left_hip": 21, "left_thigh": 21, "left_upper_leg": 21,
    "left_upperleg": 21, "thigh_l": 21, "left_knee": 24, "left_shin": 24,
    "calf_l": 24, "left_ankle": 25, "foot_l": 25,
}

DEFAULT_PHENOTYPE = {"height": 1.7, "weight": 70.0, "age": 25, "gender": 0, "muscle": 0.5}

_model = None


def _get_model():
    global _model
    if _model is None:
        import anny
        _model = anny.create_fullbody_model().float()
    return _model


def _euler_to_rotation_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Euler XYZ (degrees) → 3x3 rotation matrix. R = Rz * Ry * Rx."""
    x, y, z = math.radians(x_deg), math.radians(y_deg), math.radians(z_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx],
    ], dtype=np.float32)


def _pose_tensor(rotations: dict[str, list[float]], bone_count: int = 163) -> np.ndarray:
    """Named rotations → (1, bone_count, 4, 4) pose array."""
    pose = np.tile(np.eye(4, dtype=np.float32), (1, bone_count, 1, 1))
    for joint, angles in rotations.items():
        idx = JOINT_MAP.get(joint)
        if idx is None or len(angles) != 3:
            continue
        pose[0, idx, :3, :3] = _euler_to_rotation_matrix(*angles)
    return pose


def _forward_anny(rotations: dict[str, list[float]],
                   phenotype: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Anny forward pass. Returns dict with 'vertices' (1, 13718, 3)."""
    model = _get_model()
    if phenotype is None:
        phenotype = dict(DEFAULT_PHENOTYPE)
    import torch
    pose = torch.from_numpy(_pose_tensor(rotations))
    return model.forward(pose_parameters=pose, phenotype_kwargs=phenotype)


def _get_faces() -> np.ndarray:
    """Get Anny triangular face indices (27420, 3)."""
    return _get_model().get_triangular_faces().cpu().numpy()


# ─── Mesh → 2D Rendering ─────────────────────────────────────────────────────

def _render_pyrender(vertices: np.ndarray, faces: np.ndarray,
                     width: int, height: int) -> np.ndarray:
    """GPU-accelerated render via pyrender."""
    import trimesh
    import pyrender

    bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    verts = vertices - bbox_center
    scale = max(np.abs(verts[:, 2]).max() * 1.4, np.abs(verts[:, 0]).max() * 1.1)
    if scale > 0:
        verts /= scale
    verts[:, 2] -= 0.05

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=np.full((len(verts), 4), [180, 180, 180, 255], dtype=np.uint8))

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    camera = pyrender.OrthographicCamera(xmag=1.2, ymag=1.2)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 3.0
    rot = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    scene.add(camera, pose=rot @ cam_pose)

    key = np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=key)
    fill = np.array([[1, 0, 0, -2], [0, -1, 0, -1], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=1.5), pose=fill)

    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)
    r.delete()
    return color


def _render_pil(vertices: np.ndarray, faces: np.ndarray,
                width: int, height: int) -> np.ndarray:
    """CPU fallback render via PIL (painter's algorithm)."""
    from PIL import Image, ImageDraw

    bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    verts = vertices - bbox_center
    scale = max(np.abs(verts[:, 2]).max() * 1.4, np.abs(verts[:, 0]).max() * 1.1)
    if scale > 0:
        verts /= scale
    verts[:, 2] -= 0.05

    sx = (verts[:, 0] * (width / 2) + width / 2).astype(int)
    sy = (-verts[:, 2] * (height / 2) + height / 2).astype(int)
    depths = verts[:, 1]
    order = np.argsort(-depths[faces].mean(axis=1))

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    face_depths = depths[faces].mean(axis=1)
    for fi in order:
        tri = faces[fi]
        pts = [(int(sx[tri[0]]), int(sy[tri[0]])),
               (int(sx[tri[1]]), int(sy[tri[1]])),
               (int(sx[tri[2]]), int(sy[tri[2]]))]
        shade = max(120, min(220, int(170 + face_depths[fi] * 30)))
        draw.polygon(pts, fill=(shade, shade, shade))
    return np.array(img)


def _rotate_y(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vertices around Y-axis. 0=front, 90=right, 180=back, 270=left."""
    if abs(angle_deg) < 0.01:
        return vertices
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return vertices @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64).T


def render_pose(rotations: dict[str, list[float]],
                width: int = 1024, height: int = 1024,
                model_rotation_y: float = 0.0,
                backend: str = "auto",
                phenotype: dict[str, Any] | None = None) -> np.ndarray:
    """Render a 3D body pose to a 2D image.

    Args:
        rotations: Joint name → [x_deg, y_deg, z_deg] (Anny T-pose rest).
        width, height: Output image dimensions.
        model_rotation_y: Whole-body rotation. 0=front, 90=right, 180=back, 270=left.
        backend: "pyrender" (GPU), "pil" (CPU), or "auto".
        phenotype: Optional body shape overrides (height, weight, age, gender, muscle).

    Returns:
        (H, W, 3) uint8 numpy array — gray body on white background.
    """
    result = _forward_anny(rotations, phenotype)
    vertices = result["vertices"][0].detach().cpu().numpy()
    faces = _get_faces()

    vertices = _rotate_y(vertices, model_rotation_y)

    if backend == "auto":
        try:
            return _render_pyrender(vertices, faces, width, height)
        except Exception:
            return _render_pil(vertices, faces, width, height)
    elif backend == "pyrender":
        return _render_pyrender(vertices, faces, width, height)
    else:
        return _render_pil(vertices, faces, width, height)


def render_pose_b64(rotations: dict[str, list[float]],
                    width: int = 1024, height: int = 1024,
                    model_rotation_y: float = 0.0,
                    backend: str = "auto") -> str:
    """Like render_pose but returns base64-encoded PNG bytes."""
    import base64, io
    from PIL import Image
    arr = render_pose(rotations, width, height, model_rotation_y, backend)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
