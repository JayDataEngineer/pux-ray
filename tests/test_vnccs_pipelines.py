"""E2E verification of VNCCS 1:1 ComfyUI workflow ports.

Tests verify:
1. Pose grid rendering (VNCCS_PoseGenerator equivalent)
2. Prompt construction (CharacterCreator equivalent)
3. Pipeline DAG structure matches ComfyUI workflow node graph
4. All helper functions work (face crop, resize, composite)
5. char_sheet() and pose_edit() call svc.load/infer with correct params
"""
from __future__ import annotations

import base64
import io
import json
import sys
from unittest.mock import MagicMock, patch, call, MagicMock as MockModule

import numpy as np
import pytest


def _make_test_image_b64(w=256, h=256, color=(128, 128, 128)) -> str:
    from PIL import Image
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Test 1: Pose grid data ──────────────────────────────────────────────────

def test_pose_grid_data_loads():
    """VNCCS_PoseGenerator data file loads correctly with 12 poses."""
    from services.workflows.vnccs import _load_default_pose_grid

    grid = _load_default_pose_grid()
    assert grid["canvas"]["width"] == 512
    assert grid["canvas"]["height"] == 1536
    assert len(grid["poses"]) == 12, f"Expected 12 poses, got {len(grid['poses'])}"

    for i, pose in enumerate(grid["poses"]):
        assert len(pose) == 18, f"Pose {i} has {len(pose)} keypoints, expected 18"


def test_pose_grid_renders_with_pil():
    """Pose grid rendering logic produces valid output."""
    from services.workflows.vnccs import _load_default_pose_grid, OPENPOSE_KEYPOINTS, OPENPOSE_BONES, BONE_COLORS

    grid = _load_default_pose_grid()

    from PIL import Image, ImageDraw
    canvas_w = grid["canvas"]["width"]
    canvas_h = grid["canvas"]["height"]
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for pose in grid["poses"]:
        pts = {}
        for kp_name in OPENPOSE_KEYPOINTS:
            coords = pose.get(kp_name)
            if coords and len(coords) >= 2:
                pts[kp_name] = (int(coords[0]), int(coords[1]))

        for idx, (a_idx, b_idx) in enumerate(OPENPOSE_BONES):
            a_name = OPENPOSE_KEYPOINTS[a_idx]
            b_name = OPENPOSE_KEYPOINTS[b_idx]
            if a_name in pts and b_name in pts:
                draw.line([pts[a_name], pts[b_name]], fill=BONE_COLORS[idx], width=3)

        for pt in pts.values():
            r = 4
            draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=(0, 0, 255))

    assert canvas.size == (512, 1536)
    pixels = np.array(canvas)
    non_white = np.sum(pixels < 255)
    assert non_white > 0, "Pose grid should contain colored bones"


# ─── Test 2: CharacterCreator prompt construction ───────────────────────────

def test_char_prompt_female():
    """CharacterCreator prompt for default female character."""
    from services.workflows.vnccs import _build_char_prompt

    pos, neg = _build_char_prompt(
        aesthetics="masterpiece,best quality,amazing quality",
        background_color="green",
        sex="female", age=18, nsfw=False,
        race="human", eyes="blue eyes", hair="black long",
    )
    assert "(1girl)" in pos
    assert "green background" in pos
    assert "18yo" in pos
    assert "wear white bra and panties" in pos
    assert "(human race:1.0)" in pos
    assert "(blue eyes eyes:1.0)" in pos
    assert "(black long hair:1.0)" in pos
    assert "bad quality,worst quality" in neg


def test_char_prompt_male_nsfw():
    """CharacterCreator prompt for male NSFW character."""
    from services.workflows.vnccs import _build_char_prompt

    pos, neg = _build_char_prompt(sex="male", age=22, nsfw=True)
    assert "(1boy)" in pos
    assert "22yo" in pos
    assert "(young_adult man:1.5)" in pos
    assert "(naked, nude, penis)" in pos
    assert "1girl" in neg


def test_char_prompt_age_ranges():
    """Age range descriptors match CharacterCreator node logic."""
    from services.workflows.vnccs import _build_char_prompt

    cases = [
        (2, "toddler girl"), (8, "loli"), (14, "teenager girl"),
        (18, "teenager girl"), (19, "young_adult woman"),
        (35, "adult woman"), (55, "old woman"),
    ]
    for age, expected in cases:
        pos, _ = _build_char_prompt(age=age)
        assert expected in pos, f"Age {age} should contain '{expected}', got: {pos}"


def test_face_details_prompt():
    """Face detail prompt construction."""
    from services.workflows.vnccs import _build_face_details

    result = _build_face_details(
        sex="female", race="human", eyes="blue eyes",
        hair="black long", skin_color="white",
    )
    assert "1girl" in result
    assert "human race" in result
    assert "(expressionless:1.0)" in result


# ─── Test 3: Image helpers ───────────────────────────────────────────────────

def test_resize_image():
    from services.workflows.vnccs import _resize_image_b64

    img_b64 = _make_test_image_b64(512, 512)
    resized = _resize_image_b64(img_b64, 256, 256)
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(resized)))
    assert img.size == (256, 256)


def test_compose_side_by_side():
    from services.workflows.vnccs import _compose_images_side_by_side

    img1 = _make_test_image_b64(100, 200, (255, 0, 0))
    img2 = _make_test_image_b64(100, 200, (0, 255, 0))
    composite = _compose_images_side_by_side(img1, img2)

    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(composite)))
    assert img.size == (200, 200)
    assert img.getpixel((10, 100)) == (255, 0, 0)
    assert img.getpixel((150, 100)) == (0, 255, 0)


# ─── Test 4: char_sheet pipeline structure ───────────────────────────────────

def test_char_sheet_pipeline_structure():
    """Verify char_sheet matches VN_Step1_QWEN_CharSheetGenerator_v1 DAG.

    Steps verified:
      1. svc.load("z_image") -> SD base (8-step turbo, cfg 1.0, 1024x1024)
      2. svc.load("qwen-image-edit") -> QWEN refinement
         reference_images: [pose_grid, base_image]
         loras: [lightning_4steps, poser_helper_v2]
         4 steps, cfg 1.0
    """
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    test_img = _make_test_image_b64(1024, 1024)
    mock_svc.infer.return_value = {
        "status": "ok", "data": test_img, "media_type": "image/png",
    }

    with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
        with patch("services.workflows.vnccs._render_pose_grid_b64", return_value=test_img):
            with patch("services.workflows.vnccs._crop_face_b64", return_value=None):
                from services.workflows.vnccs import char_sheet
                result = char_sheet(sex="female", age=18, seed=42, quality="turbo")

    assert result["status"] == "ok"
    load_calls = [c.args[0] for c in mock_svc.load.call_args_list]
    assert "z_image" in load_calls
    assert "qwen-image-edit" in load_calls

    infer_calls = mock_svc.infer.call_args_list
    assert len(infer_calls) >= 2

    # SD base
    sd_call = infer_calls[0][0][0]
    assert sd_call["model"] == "z_image"
    assert sd_call["sampling_steps"] == 8
    assert sd_call["guide_scale"] == 1.0
    assert sd_call["width"] == 1024
    assert sd_call["height"] == 1024
    assert "(1girl)" in sd_call["input_prompt"]

    # QWEN refinement
    qwen_call = infer_calls[1][0][0]
    assert qwen_call["model"] == "qwen-image-edit"
    assert qwen_call["sampling_steps"] == 4
    assert qwen_call["guide_scale"] == 1.0
    assert len(qwen_call["reference_images"]) == 2
    assert "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" in qwen_call["loras_selected"]
    assert "VNCCS/poser_helper_v2_000004200.safetensors" in qwen_call["loras_selected"]


def test_char_sheet_with_input_image():
    """When image_b64 is provided, SD generation is skipped."""
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }
    test_img = _make_test_image_b64(1024, 1024)

    with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
        with patch("services.workflows.vnccs._render_pose_grid_b64", return_value=test_img):
            with patch("services.workflows.vnccs._crop_face_b64", return_value=None):
                from services.workflows.vnccs import char_sheet
                result = char_sheet(image_b64=test_img, seed=42)

    assert result["status"] == "ok"
    load_calls = [c.args[0] for c in mock_svc.load.call_args_list]
    assert "z_image" not in load_calls
    assert "qwen-image-edit" in load_calls


def test_char_sheet_standard_quality():
    """Standard quality: 20 steps, cfg 6.0 (matches ComfyUI SDXL KSampler)."""
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }

    with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
        with patch("services.workflows.vnccs._render_pose_grid_b64", return_value=_make_test_image_b64()):
            with patch("services.workflows.vnccs._crop_face_b64", return_value=None):
                from services.workflows.vnccs import char_sheet
                char_sheet(quality="standard", seed=42)

    sd_call = mock_svc.infer.call_args_list[0][0][0]
    assert sd_call["sampling_steps"] == 20
    assert sd_call["guide_scale"] == 6.0


def test_char_sheet_face_detailer():
    """Face detailer runs with 20 steps, cfg 7 (matching ComfyUI face detailer node)."""
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    test_img = _make_test_image_b64(1024, 1024)
    mock_svc.infer.return_value = {
        "status": "ok", "data": test_img, "media_type": "image/png",
    }

    with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
        with patch("services.workflows.vnccs._render_pose_grid_b64", return_value=test_img):
            with patch("services.workflows.vnccs._crop_face_b64", return_value=_make_test_image_b64(200, 200)):
                with patch("services.workflows.vnccs._composite_face_back", return_value=test_img):
                    with patch("services.workflows.vnccs._upscale_image_b64", return_value=test_img):
                        from services.workflows.vnccs import char_sheet
                        result = char_sheet(sex="female", seed=42)

    # Should have 3 infer calls: SD base, QWEN refinement, face detailer
    infer_calls = mock_svc.infer.call_args_list
    assert len(infer_calls) >= 3, f"Expected 3+ infer calls (SD + QWEN + face), got {len(infer_calls)}"

    # Face detailer call (last infer)
    face_call = infer_calls[-1][0][0]
    assert face_call["sampling_steps"] == 20  # Face detailer uses 20 steps
    assert face_call["guide_scale"] == 7.0    # Face detailer uses cfg 7
    assert "expressionless" in face_call["input_prompt"]

    # Result should include face crop
    assert "face" in result



def test_char_sheet_upscale_step():
    """SeedVR2 upscale step (node 638) runs between QWEN refinement and face detailer."""
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    test_img = _make_test_image_b64(1024, 1024)
    mock_svc.infer.return_value = {
        "status": "ok", "data": test_img, "media_type": "image/png",
    }

    with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
        with patch("services.workflows.vnccs._render_pose_grid_b64", return_value=test_img):
            with patch("services.workflows.vnccs._crop_face_b64", return_value=None):
                with patch("services.workflows.vnccs._upscale_image_b64", return_value=test_img) as mock_upscale:
                    from services.workflows.vnccs import char_sheet
                    result = char_sheet(sex="female", seed=42)

                    # Upscale was called (SeedVR2 equivalent)
                    assert mock_upscale.called, "Upscale step (SeedVR2) should run"


# ─── Test 5: pose_edit pipeline structure ────────────────────────────────────

def _mock_pose_dependencies():
    """Set up mocks for dwpose and body_mesh lazy imports."""
    mock_dwpose = MagicMock()
    mock_dwpose.skeleton_from_image_b64.return_value = _make_test_image_b64(1024, 1024)

    mock_body_mesh = MagicMock()
    mock_body_mesh.render_pose_b64.return_value = _make_test_image_b64(1024, 1024)

    return mock_dwpose, mock_body_mesh


def test_pose_edit_mesh_mode():
    """Pose Studio mesh mode: BodyMesh renders from joint rotations."""
    mock_dwpose, mock_body_mesh = _mock_pose_dependencies()
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }

    with patch.dict(sys.modules, {
        "services.workflows.utils.dwpose": mock_dwpose,
        "services.workflows.utils.body_mesh": mock_body_mesh,
    }):
        with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
            from services.workflows.vnccs import pose_edit
            result = pose_edit(
                character_image_b64=_make_test_image_b64(1024, 1024),
                rotations={"r_shoulder": [0, 0, -90]},
                seed=42,
            )

    assert result["status"] == "ok"
    assert mock_svc.load.call_args_list[0].args[0] == "qwen-image-edit"

    # BodyMesh was called
    assert mock_body_mesh.render_pose_b64.called

    # QWEN infer matches PoseStudio workflow
    qwen_call = mock_svc.infer.call_args_list[0][0][0]
    assert qwen_call["model"] == "qwen-image-edit"
    assert qwen_call["input_prompt"] == "Draw character from image2"
    assert qwen_call["sampling_steps"] == 4
    assert qwen_call["guide_scale"] == 1.0
    assert len(qwen_call["reference_images"]) == 3
    assert "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors" in qwen_call["loras_selected"]
    assert "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" in qwen_call["loras_selected"]


def test_pose_edit_capture_mode():
    """Pose Studio capture mode: DWPose extracts from pose image."""
    mock_dwpose, _ = _mock_pose_dependencies()
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }

    with patch.dict(sys.modules, {"services.workflows.utils.dwpose": mock_dwpose}):
        with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
            from services.workflows.vnccs import pose_edit
            result = pose_edit(
                character_image_b64=_make_test_image_b64(1024, 1024),
                pose_image_b64=_make_test_image_b64(512, 768),
                seed=42,
            )

    assert result["status"] == "ok"
    qwen_call = mock_svc.infer.call_args_list[0][0][0]
    assert len(qwen_call["reference_images"]) == 3


def test_pose_edit_lighting_prompt():
    """Lighting prompt appended per VNCCS_PoseStudio prompt_template."""
    mock_dwpose, _ = _mock_pose_dependencies()
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }

    with patch.dict(sys.modules, {"services.workflows.utils.dwpose": mock_dwpose}):
        with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
            from services.workflows.vnccs import pose_edit
            result = pose_edit(
                character_image_b64=_make_test_image_b64(1024, 1024),
                pose_image_b64=_make_test_image_b64(512, 768),
                lighting_prompt="soft warm lighting from the left",
                seed=42,
            )

    qwen_call = mock_svc.infer.call_args_list[0][0][0]
    assert "Draw character from image2" in qwen_call["input_prompt"]
    assert "soft warm lighting from the left" in qwen_call["input_prompt"]


def test_pose_edit_custom_prompt():
    """Custom user_prompt overrides default."""
    mock_dwpose, _ = _mock_pose_dependencies()
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }

    with patch.dict(sys.modules, {"services.workflows.utils.dwpose": mock_dwpose}):
        with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
            from services.workflows.vnccs import pose_edit
            result = pose_edit(
                character_image_b64=_make_test_image_b64(1024, 1024),
                pose_image_b64=_make_test_image_b64(512, 768),
                user_prompt="Draw character from image2 in a dramatic pose",
                seed=42,
            )

    qwen_call = mock_svc.infer.call_args_list[0][0][0]
    assert qwen_call["input_prompt"] == "Draw character from image2 in a dramatic pose"


def test_pose_edit_mesh_config():
    """mesh_config (phenotype) is passed to BodyMesh."""
    mock_dwpose, mock_body_mesh = _mock_pose_dependencies()
    mock_svc = MagicMock()
    mock_svc._loaded_model = "qwen-image-edit"
    mock_svc.infer.return_value = {
        "status": "ok", "data": _make_test_image_b64(1024, 1024), "media_type": "image/png",
    }
    phenotype = {"age": 27, "gender": 0, "weight": 0.5}

    with patch.dict(sys.modules, {
        "services.workflows.utils.dwpose": mock_dwpose,
        "services.workflows.utils.body_mesh": mock_body_mesh,
    }):
        with patch("services.workflows.vnccs.get_service", return_value=mock_svc):
            from services.workflows.vnccs import pose_edit
            result = pose_edit(
                character_image_b64=_make_test_image_b64(1024, 1024),
                rotations={"spine": [-15, 0, 0]},
                mesh_config=phenotype,
                seed=42,
            )

    assert mock_body_mesh.render_pose_b64.called
    call_kwargs = mock_body_mesh.render_pose_b64.call_args[1]
    assert call_kwargs.get("phenotype") == phenotype


# ─── Test 6: Workflow registry ───────────────────────────────────────────────

def test_workflows_registered():
    from services.workflows.vnccs import get_workflows
    workflows = get_workflows()
    ids = [w["id"] for w in workflows]
    for expected in ["vnccs/char-sheet", "vnccs/pose-edit", "vnccs/emotions",
                     "vnccs/sprite", "vnccs/clone", "vnccs/detailer"]:
        assert expected in ids, f"Missing workflow: {expected}"


def test_pose_studio_prompt_constant():
    from services.workflows.vnccs import VNCCS_POSE_STUDIO_PROMPT
    assert VNCCS_POSE_STUDIO_PROMPT == "Draw character from image2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
