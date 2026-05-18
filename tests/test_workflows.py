"""Tests for the workflow orchestration system — vnccs, wdc, tech_noir.

Tests pure logic (math, data flow) without GPU/Wan2GP dependencies.
The Wan2GPService is mocked — we test orchestration, not model inference.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import base64
import io

import numpy as np
import pytest
from PIL import Image


# Small valid PNG for test payloads (1x1 white pixel)
_VALID_PNG_B64: str = ""
_buf = io.BytesIO()
Image.new("RGB", (1, 1), (255, 255, 255)).save(_buf, format="PNG")
_VALID_PNG_B64 = base64.b64encode(_buf.getvalue()).decode()

# 512x512 gray image for larger tests
_VALID_512_B64: str = ""
_buf2 = io.BytesIO()
Image.new("RGB", (512, 512), (200, 200, 200)).save(_buf2, format="PNG")
_VALID_512_B64 = base64.b64encode(_buf2.getvalue()).decode()

# Module-level flags for conditional skips
_ANNY_AVAILABLE = False
try:
    import anny  # noqa: F401
    _ANNY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Route registration
# =============================================================================

class TestWorkflowRegistration:
    """All 22 workflows must be registered and discoverable via the route module."""

    def test_all_registered(self):
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        assert len(_WORKFLOW_REGISTRY) == 22

    def test_vnccs_workflows_present(self):
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        vnccs_ids = [k for k in _WORKFLOW_REGISTRY if k.startswith("vnccs/")]
        assert len(vnccs_ids) == 6
        assert "vnccs/char-sheet" in vnccs_ids
        assert "vnccs/emotions" in vnccs_ids
        assert "vnccs/sprite" in vnccs_ids
        assert "vnccs/pose-edit" in vnccs_ids
        assert "vnccs/clone" in vnccs_ids
        assert "vnccs/detailer" in vnccs_ids

    def test_wdc_workflows_present(self):
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        wdc_ids = [k for k in _WORKFLOW_REGISTRY if k.startswith("wdc/")]
        assert len(wdc_ids) == 4

    def test_tech_noir_workflows_present(self):
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        tn_ids = [k for k in _WORKFLOW_REGISTRY if k.startswith("tech-noir/")]
        assert len(tn_ids) == 12

    def test_each_registered_function_has_docstring(self):
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        for wf_id, fn in _WORKFLOW_REGISTRY.items():
            assert fn.__doc__, f"{wf_id} has no docstring"

    def test_workflow_list_returns_all(self):
        from services.workflows.vnccs import get_workflows as vnccs_wfs
        from services.workflows.wdc import get_workflows as wdc_wfs
        from services.workflows.tech_noir import get_workflows as tn_wfs
        total = len(vnccs_wfs()) + len(wdc_wfs()) + len(tn_wfs())
        assert total == 22


# =============================================================================
# Workflow functions (mocked Wan2GPService)
# =============================================================================

class MockWan2GPService:
    """Simulates Wan2GPService.load()/.infer() without GPU."""

    def __init__(self):
        self.loaded_model = None
        self._loaded_model = None  # real Wan2GPService uses this name
        self.load_calls = []
        self.infer_calls = []
        self._registry = {}

    def load(self, model_name: str) -> None:
        self.load_calls.append(model_name)
        self.loaded_model = model_name
        self._loaded_model = model_name

    def infer(self, payload: dict) -> dict:
        self.infer_calls.append(payload)
        return {
            "status": "ok",
            "data": "dGVzdF9kYXRh",  # base64 "test_data"
            "media_type": "image/png",
        }

    def available_models(self):
        return list(self._registry.keys())

    def status(self):
        return {"loaded": self.loaded_model}


@pytest.fixture
def mock_svc():
    """Replace get_service in all workflow modules with a mock."""
    svc = MockWan2GPService()
    patches = [
        patch("services.workflows.base.get_service", return_value=svc),
        patch("services.workflows.vnccs.get_service", return_value=svc),
        patch("services.workflows.wdc.get_service", return_value=svc),
        patch("services.workflows.tech_noir.get_service", return_value=svc),
    ]
    for p in patches:
        p.start()
    yield svc
    for p in patches:
        p.stop()


class TestVNCCSWorkflows:
    """VNCCS workflow functions should construct correct payloads."""

    def test_char_sheet_calls_z_image_then_qwen(self, mock_svc):
        from services.workflows.vnccs import char_sheet
        result = char_sheet(prompt="test character", seed=42, quality="turbo")
        assert result["status"] == "ok"
        assert mock_svc.load_calls == ["z_image", "qwen-image-edit"]
        assert len(mock_svc.infer_calls) == 2
        assert mock_svc.infer_calls[0]["input_prompt"] == "test character"
        assert mock_svc.infer_calls[0]["seed"] == 42
        assert mock_svc.infer_calls[0]["sampling_steps"] == 8
        assert mock_svc.infer_calls[1]["input_prompt"] == "Draw character from image2"

    def test_char_sheet_base_quality(self, mock_svc):
        from services.workflows.vnccs import char_sheet
        char_sheet(prompt="test", seed=0, quality="base")
        assert mock_svc.infer_calls[0]["sampling_steps"] == 50

    def test_emotions_loops_over_emotions(self, mock_svc):
        from services.workflows.vnccs import emotions
        result = emotions(
            sheet_image_b64=_VALID_PNG_B64,
            emotions_list=["happy", "sad", "angry"],
            costumes=None, seed=42,
        )
        assert result["total"] == 3
        assert result["results"][0]["emotion"] == "happy"
        assert result["results"][1]["emotion"] == "sad"
        assert result["results"][2]["emotion"] == "angry"
        assert len(mock_svc.infer_calls) == 3

    def test_emotions_with_costumes(self, mock_svc):
        from services.workflows.vnccs import emotions
        result = emotions(
            sheet_image_b64=_VALID_PNG_B64,
            emotions_list=["happy"], costumes=["casual", "formal"], seed=42,
        )
        assert result["total"] == 2

    def test_clone_passthrough_attributes(self, mock_svc):
        from services.workflows.vnccs import clone
        clone(reference_image_b64=_VALID_PNG_B64, character_def={"age": 25, "hair": "blue"}, seed=42)
        assert mock_svc.infer_calls[0]["age"] == 25
        assert mock_svc.infer_calls[0]["hair"] == "blue"

    def test_detailer_passthrough_prompt(self, mock_svc):
        from services.workflows.vnccs import detailer
        detailer(image_b64=_VALID_PNG_B64, region_prompt="fix eyes", seed=42)
        assert mock_svc.infer_calls[0]["input_prompt"] == "fix eyes"

    @pytest.mark.skipif(not _ANNY_AVAILABLE, reason="requires anny package")
    def test_sprite_uses_reference_images(self, mock_svc):
        """sprite() should pass reference_images list (not manual composite)."""
        from services.workflows.vnccs import sprite
        result = sprite(
            sheet_image_b64=_VALID_PNG_B64,
            poses=[{"right_shoulder": [0, 0, 90]}],
            directions=[0.0],
            seed=42, backend="pil",
        )
        assert result["total"] == 1
        call = mock_svc.infer_calls[0]
        assert "reference_images" in call
        assert len(call["reference_images"]) == 3


class TestWDCWorkflows:
    """WDC workflow functions should construct correct payloads."""

    def test_ltx_fflf_2stage_includes_first_frame(self, mock_svc):
        from services.workflows.wdc import ltx_fflf_2stage
        ltx_fflf_2stage(prompt="test video", first_frame_b64="dGVzdF9mcmFtZQ==", seed=42)
        assert mock_svc.load_calls == ["ltx2"]
        assert mock_svc.infer_calls[0]["image_b64"] == "dGVzdF9mcmFtZQ=="

    def test_ltx_fflf_2stage_with_last_frame(self, mock_svc):
        from services.workflows.wdc import ltx_fflf_2stage
        ltx_fflf_2stage(prompt="test", first_frame_b64="dGVzdA==", last_frame_b64="bGFzdA==", seed=42)
        assert mock_svc.infer_calls[0]["image_end_b64"] == "bGFzdA=="

    def test_ltx_audio_includes_audio(self, mock_svc):
        from services.workflows.wdc import ltx_audio
        ltx_audio(prompt="test", first_frame_b64="dGVzdA==", audio_b64="YXVkaW8=", seed=42)
        assert mock_svc.infer_calls[0]["audio_b64"] == "YXVkaW8="

    def test_timeline_generates_per_segment(self, mock_svc):
        from services.workflows.wdc import timeline
        segments = [{"prompt": "shot one", "frames": 48}, {"prompt": "shot two", "frames": 48}]
        result = timeline(segments=segments, seed=42)
        assert len(result["segments"]) == 2
        assert mock_svc.infer_calls[0]["seed"] == 42
        assert mock_svc.infer_calls[1]["seed"] == 43


class TestTechNoirWorkflows:
    """Tech Noir build stage workflow functions."""

    def test_generate_passes_prompt_and_seed(self, mock_svc):
        from services.workflows.tech_noir import generate
        generate(prompt="test character", seed=42, quality="turbo")
        assert mock_svc.load_calls == ["z_image"]
        assert mock_svc.infer_calls[0]["input_prompt"] == "test character"
        assert mock_svc.infer_calls[0]["sampling_steps"] == 8

    def test_sheet_passes_image_b64(self, mock_svc):
        from services.workflows.tech_noir import sheet
        sheet(character_image_b64=_VALID_PNG_B64, seed=42)
        assert mock_svc.load_calls == ["qwen-image-edit"]
        assert mock_svc.infer_calls[0]["image_b64"] == _VALID_PNG_B64

    def test_emotions_delegates_to_vnccs(self, mock_svc):
        from services.workflows.tech_noir import emotions
        emotions(sheet_image_b64=_VALID_PNG_B64, emotions_list=["happy", "sad"], seed=42)
        assert len(mock_svc.infer_calls) == 2

    def test_trellis_passes_params(self, mock_svc):
        from services.workflows.tech_noir import trellis
        trellis(image_b64=_VALID_PNG_B64, seed=1, steps=12, guidance=7.5)
        assert mock_svc.load_calls == ["trellis"]
        assert mock_svc.infer_calls[0]["sampling_steps"] == 12

    def test_video_passes_params(self, mock_svc):
        from services.workflows.tech_noir import video
        video(image_b64=_VALID_PNG_B64, prompt="test", seed=42, fps=24, frames=97)
        assert mock_svc.load_calls == ["ltx2"]
        assert mock_svc.infer_calls[0]["frame_num"] == 97

    def test_face_detailer_detects_face(self, mock_svc):
        """face_detailer calls QWEN fallback on blank image (no face detected)."""
        from services.workflows.tech_noir import face_detailer
        face_detailer(image_b64=_VALID_512_B64, seed=42)
        assert len(mock_svc.infer_calls) >= 1


# =============================================================================
# Utility: Motion converter (pure math, no GPU)
# =============================================================================

class TestMotionConverter:
    """HY-Motion NPZ → Anny rotation dict conversion."""

    def test_rot6d_to_matrix_identity(self):
        from services.workflows.utils.motion import rot6d_to_matrix
        r6d = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        R = rot6d_to_matrix(r6d)
        assert R.shape == (1, 3, 3)
        I = np.eye(3)
        assert np.allclose(R[0] @ R[0].T, I, atol=1e-6)
        assert np.allclose(np.linalg.det(R[0]), 1.0, atol=1e-3)

    def test_matrix_to_euler_identity(self):
        from services.workflows.utils.motion import matrix_to_euler_xyz, rot6d_to_matrix
        r6d = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        R = rot6d_to_matrix(r6d)
        euler = matrix_to_euler_xyz(R)
        assert np.allclose(euler[0], [0, 0, 0], atol=1e-4)

    def test_rot6d_matrix_is_orthonormal(self):
        """After Gram-Schmidt, columns should be orthonormal."""
        from services.workflows.utils.motion import rot6d_to_matrix
        rng = np.random.default_rng(42)
        r6d = rng.random((3, 6))
        R = rot6d_to_matrix(r6d)
        for i in range(3):
            I = np.eye(3)
            assert np.allclose(R[i] @ R[i].T, I, atol=1e-6)

    def test_yup_to_zup_swaps_yz(self):
        from services.workflows.utils.motion import convert_yup_to_zup
        euler = np.array([[10.0, 20.0, 30.0]])
        zup = convert_yup_to_zup(euler)
        assert np.allclose(zup[0], [10.0, 30.0, 20.0])

    def test_extract_keyframes_from_real_npz(self):
        from services.workflows.utils.motion import extract_keyframes
        npz_path = "/home/user/Documents/programs/tech-noir-studio/departments/art/motions/shared/idle.npz"
        kfs = extract_keyframes(npz_path, num_keyframes=4)
        assert len(kfs) == 4
        assert "right_shoulder" in kfs[0]
        assert "left_shoulder" in kfs[0]
        assert "spine" in kfs[0]
        for angles in kfs[0].values():
            assert len(angles) == 3
            assert all(isinstance(v, float) for v in angles)

    def test_npz_bytes_roundtrip(self):
        from services.workflows.utils.motion import npz_bytes_to_keyframes
        npz_path = "/home/user/Documents/programs/tech-noir-studio/departments/art/motions/shared/idle.npz"
        with open(npz_path, "rb") as f:
            kfs = npz_bytes_to_keyframes(f.read(), num_keyframes=2)
        assert len(kfs) == 2

    def test_npz_b64_roundtrip(self):
        from services.workflows.utils.motion import npz_b64_to_keyframes
        import base64
        npz_path = "/home/user/Documents/programs/tech-noir-studio/departments/art/motions/shared/idle.npz"
        with open(npz_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        kfs = npz_b64_to_keyframes(b64, num_keyframes=3)
        assert len(kfs) == 3


# =============================================================================
# Utility: Body mesh rotation math (pure numpy, no anny/pyrender)
# =============================================================================

class TestBodyMeshRotationMath:
    """Body mesh rotation utilities that can be tested without anny."""

    def test_euler_to_matrix_identity(self):
        from services.workflows.utils.body_mesh import _euler_to_rotation_matrix
        R = _euler_to_rotation_matrix(0, 0, 0)
        I = np.eye(3)
        assert np.allclose(R, I, atol=1e-6)

    def test_euler_to_matrix_90z(self):
        from services.workflows.utils.body_mesh import _euler_to_rotation_matrix
        R = _euler_to_rotation_matrix(0, 0, 90)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        assert np.allclose(R, expected, atol=1e-5)

    def test_euler_to_matrix_90x(self):
        from services.workflows.utils.body_mesh import _euler_to_rotation_matrix
        R = _euler_to_rotation_matrix(90, 0, 0)
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        assert np.allclose(R, expected, atol=1e-5)

    def test_rotate_y_identity(self):
        from services.workflows.utils.body_mesh import _rotate_y
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = _rotate_y(v, 0)
        assert np.allclose(result, v)

    def test_rotate_y_90_degrees(self):
        from services.workflows.utils.body_mesh import _rotate_y
        v = np.array([[1, 0, 0]])
        result = _rotate_y(v, 90)
        assert np.allclose(result, [[0, 0, -1]], atol=1e-5)

    def test_pose_tensor_within_bone_count(self):
        from services.workflows.utils.body_mesh import _pose_tensor
        rotations = {"right_shoulder": [0, 0, 90], "head": [10, 0, 0]}
        pose = _pose_tensor(rotations, bone_count=163)
        assert pose.shape == (1, 163, 4, 4)
        I = np.eye(4, dtype=np.float32)
        assert np.allclose(pose[0, 0], I)  # pelvis identity
        # head (bone 103) should be rotated
        assert not np.allclose(pose[0, 103], I)

    def test_pose_tensor_unknown_joint_ignored(self):
        from services.workflows.utils.body_mesh import _pose_tensor
        rotations = {"nonexistent_joint": [10, 20, 30]}
        pose = _pose_tensor(rotations, bone_count=10)
        I = np.eye(4, dtype=np.float32)
        assert np.allclose(pose[0, 0], I)


# =============================================================================
# Utility: FaceDetailer
# =============================================================================

class TestFaceDetailer:
    """Face detection via DWPose keypoints + mask creation."""

    def test_face_bbox_from_blank_image_returns_none(self):
        from services.workflows.utils.detailer import _detect_face_bbox
        img = np.full((512, 512, 3), 200, dtype=np.uint8)
        bbox = _detect_face_bbox(img)
        assert bbox is None

    def test_refine_faces_falls_back_on_no_face(self):
        """refine_faces on a blank image should fall back to full-image QWEN."""
        from services.workflows.utils.detailer import refine_faces
        svc = MockWan2GPService()
        with patch("services.workflows.utils.detailer.get_service", return_value=svc):
            result = refine_faces(_VALID_512_B64, prompt="fix face", seed=42)
        # No face on blank → falls back to full-image QWEN
        assert svc.load_calls == ["qwen-image-edit"]
        assert svc.infer_calls[0]["input_prompt"] == "fix face"


# =============================================================================
# Utility: DWPose (no actual model inference, just shape/type checks)
# =============================================================================

class TestDWPose:
    """DWPose skeleton extraction."""

    def test_skeleton_from_blank_image_returns_blank(self):
        from services.workflows.utils.dwpose import skeleton_from_image
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        skeleton = skeleton_from_image(img, 512, 512)
        assert skeleton.shape == (512, 512, 3)
        assert skeleton.dtype == np.uint8

    def test_skeleton_b64_roundtrip(self):
        from services.workflows.utils.dwpose import skeleton_from_image_b64
        result = skeleton_from_image_b64(_VALID_512_B64, 512, 512)
        assert isinstance(result, str)
        assert len(base64.b64decode(result)) > 0

    def test_detect_poses_returns_2d_array(self):
        from services.workflows.utils.dwpose import detect_poses
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        kps = detect_poses(img)
        assert kps.ndim == 3
        assert kps.shape[-1] == 2


# =============================================================================
# Base helpers
# =============================================================================

class TestBaseHelpers:
    """encode_output, error_response, and related utilities."""

    def test_encode_output_creates_base64(self):
        import base64
        from services.workflows.base import encode_output
        result = encode_output(b"test data", "image/png")
        assert result["status"] == "ok"
        assert base64.b64decode(result["data"]) == b"test data"
        assert result["media_type"] == "image/png"

    def test_error_response_format(self):
        from services.workflows.base import error_response
        result = error_response("something went wrong")
        assert result["status"] == "error"
        assert result["error"] == "something went wrong"

    def test_workflow_listing_consistency(self):
        """Each get_workflows() entry should have a matching registered function."""
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        from services.workflows.vnccs import get_workflows as vnccs_wfs
        from services.workflows.wdc import get_workflows as wdc_wfs
        from services.workflows.tech_noir import get_workflows as tn_wfs
        all_ids = set(_WORKFLOW_REGISTRY.keys())
        for wf in vnccs_wfs() + wdc_wfs() + tn_wfs():
            assert wf["id"] in all_ids
        assert len(all_ids) == 22


# =============================================================================
# Tech Noir VNCCS constants (shared across build stages)
# =============================================================================

class TestTechNoirConstants:
    """VNCCS constants used in workflow orchestration."""

    def test_instruction_format(self):
        from services.workflows.tech_noir import VNCCS_INSTRUCTION
        assert "Picture 1" in VNCCS_INSTRUCTION
        assert "Picture 2" in VNCCS_INSTRUCTION
        assert "Picture 3" in VNCCS_INSTRUCTION

    def test_direction_rotations(self):
        from services.workflows.tech_noir import DIRECTION_ROTATIONS
        assert DIRECTION_ROTATIONS["front"] == 0.0
        assert DIRECTION_ROTATIONS["right"] == 90.0
        assert DIRECTION_ROTATIONS["back"] == 180.0
        assert DIRECTION_ROTATIONS["left"] == 270.0
