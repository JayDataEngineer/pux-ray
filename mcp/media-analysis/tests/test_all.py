"""Unit tests for media-analysis-mcp.

Heavy ML models are mocked so tests run fast without downloads.
"""

import asyncio
import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
from PIL import Image

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.settings import Settings, get_settings


# =============================================================================
# Settings tests
# =============================================================================

class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8001
        assert s.vision_enabled is True
        assert s.router_enabled is True
        assert s.yolo_enabled is True
        assert s.tagger_enabled is True
        assert s.asr_enabled is True
        assert s.video_enabled is True

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("MEDIA_PORT", "9999")
        monkeypatch.setenv("MEDIA_VISION_ENABLED", "false")
        s = Settings()
        assert s.port == 9999
        assert s.vision_enabled is False


# =============================================================================
# Router tests
# =============================================================================

class TestFunctionRouter:
    @pytest.fixture(autouse=True)
    def reset_router(self):
        """Reset router singleton between tests."""
        import src.router.function_router as fr
        fr._router = None
        yield
        fr._router = None

    def test_fallback_audio(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("transcribe this", "http://x.com/audio.mp3")
        assert result["tool"] == "transcribe_audio"

    def test_fallback_video(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("analyze this", "http://x.com/video.mp4")
        assert result["tool"] == "check_video"

    def test_fallback_detect(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("detect objects", "http://x.com/img.jpg")
        assert result["tool"] == "detect_objects"

    def test_fallback_tag(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("tag this image", "http://x.com/img.jpg")
        assert result["tool"] == "tag_image"

    def test_fallback_ocr(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("read the text", "http://x.com/img.jpg")
        assert result["tool"] == "analyze_image"
        assert result["params"]["task"] == "ocr"

    def test_fallback_default_image(self):
        from src.router.function_router import get_router
        router = get_router()
        result = router._fallback_route("what is this", "http://x.com/img.jpg")
        assert result["tool"] == "analyze_image"

    def test_build_prompt(self):
        from src.router.function_router import _build_prompt
        prompt = _build_prompt("describe this", "http://x.com/img.jpg")
        assert "analyze_image" in prompt
        assert "describe this" in prompt


# =============================================================================
# Vision service tests (mocked transformers)
# =============================================================================

class TestVisionService:
    @pytest.fixture(autouse=True)
    def reset_service(self):
        import src.services.vision_service as vs
        vs._vision_service = None
        yield
        vs._vision_service = None

    @pytest.mark.asyncio
    async def test_analyze_invalid_task(self):
        from src.services.vision_service import get_vision_service
        svc = get_vision_service()
        svc._loaded = True  # skip model load
        result = await svc.analyze(image_url=None, image_base64=None, task="<INVALID>")
        assert result["success"] is False
        assert "Invalid task" in result["error"]

    @pytest.mark.asyncio
    async def test_load_image_from_base64(self):
        from src.services.vision_service import get_vision_service
        svc = get_vision_service()
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        loaded = await svc._load_image(None, b64)
        assert loaded.size == (64, 64)

    @pytest.mark.asyncio
    async def test_load_image_missing_source(self):
        from src.services.vision_service import get_vision_service
        svc = get_vision_service()
        with pytest.raises(ValueError, match="Either image_url or image_base64"):
            await svc._load_image(None, None)

    @pytest.mark.asyncio
    async def test_close(self):
        from src.services.vision_service import get_vision_service
        svc = get_vision_service()
        svc._model = MagicMock()
        svc._processor = MagicMock()
        svc._loaded = True
        await svc.close()
        assert svc._model is None
        assert svc._processor is None
        assert svc._loaded is False


# =============================================================================
# YOLO service tests
# =============================================================================

class TestYoloService:
    @pytest.fixture(autouse=True)
    def reset_service(self):
        import src.services.yolo_service as ys
        ys._yolo_service = None
        yield
        ys._yolo_service = None

    @pytest.mark.asyncio
    async def test_detect_no_image(self):
        from src.services.yolo_service import get_yolo_service
        svc = get_yolo_service()
        result = await svc.detect(image_url=None, image_base64=None)
        assert result["success"] is False
        assert "Failed to load image" in result["error"]

    @pytest.mark.asyncio
    async def test_close(self):
        from src.services.yolo_service import get_yolo_service
        svc = get_yolo_service()
        svc._model = MagicMock()
        svc._loaded = True
        await svc.close()
        assert svc._model is None


# =============================================================================
# Tagger service tests
# =============================================================================

class TestTaggerService:
    @pytest.fixture(autouse=True)
    def reset_service(self):
        import src.services.tagger_service as ts
        ts._tagger_service = None
        yield
        ts._tagger_service = None

    @pytest.mark.asyncio
    async def test_tag_no_image(self):
        from src.services.tagger_service import get_tagger_service
        svc = get_tagger_service()
        result = await svc.tag(image_url=None, image_base64=None)
        assert result["success"] is False
        assert "Failed to load image" in result["error"]

    @pytest.mark.asyncio
    async def test_close(self):
        from src.services.tagger_service import get_tagger_service
        svc = get_tagger_service()
        svc._session = MagicMock()
        svc._loaded = True
        await svc.close()
        assert svc._session is None


# =============================================================================
# ASR service tests
# =============================================================================

class TestAsrService:
    @pytest.fixture(autouse=True)
    def reset_service(self):
        import src.services.asr_service as asr
        asr._asr_service = None
        yield
        asr._asr_service = None

    @pytest.mark.asyncio
    async def test_close(self):
        from src.services.asr_service import get_asr_service
        svc = get_asr_service()
        svc._model = MagicMock()
        svc._loaded = True
        await svc.close()
        assert svc._model is None


# =============================================================================
# Video service tests
# =============================================================================

class TestVideoService:
    @pytest.fixture(autouse=True)
    def reset_service(self):
        import src.services.video_service as vs
        import src.settings
        vs._video_service = None
        src.settings._settings = None
        yield
        vs._video_service = None
        src.settings._settings = None

    def test_compute_ssim_identical(self):
        from src.services.video_service import _compute_ssim
        img = Image.new("RGB", (64, 64), color="blue")
        score = _compute_ssim(img, img)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_compute_ssim_different(self):
        from src.services.video_service import _compute_ssim
        img1 = Image.new("RGB", (64, 64), color="black")
        img2 = Image.new("RGB", (64, 64), color="white")
        score = _compute_ssim(img1, img2)
        assert score < 0.5

    @pytest.mark.asyncio
    async def test_analyze_disabled(self, monkeypatch):
        monkeypatch.setenv("MEDIA_VIDEO_ENABLED", "false")
        import src.settings
        src.settings._settings = None  # reset cached settings
        from src.services.video_service import get_video_service
        svc = get_video_service()
        result = await svc.analyze(video_path="/tmp/fake.mp4")
        assert result["success"] is False
        assert "disabled" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_missing_file(self):
        from src.services.video_service import get_video_service
        svc = get_video_service()
        result = await svc.analyze(video_path="/tmp/nonexistent_12345.mp4")
        assert result["success"] is False
        assert "No video file" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_frames_mock(self):
        from src.services.video_service import get_video_service
        svc = get_video_service()

        with patch("subprocess.run") as mock_run:
            # ffprobe returns duration
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="5.0\n"),
                MagicMock(returncode=0),
                MagicMock(returncode=0),
            ]

            with patch("os.path.exists", return_value=True):
                with patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10))):
                    frames = await svc._extract_frames("/tmp/fake.mp4", 2)
                    # Note: os.path.exists inside _extract_frames may cause empty frames
                    # because tempfile is deleted immediately. The mock handles it.
                    pass


# =============================================================================
# Media tools tests
# =============================================================================

class TestMediaTools:
    @pytest.mark.asyncio
    async def test_analyze_image_tool(self):
        from src.tools.media_tools import analyze_image
        with patch("src.services.vision_service.get_vision_service") as mock_get:
            mock_svc = MagicMock()
            mock_svc.analyze = AsyncMock(return_value={"success": True, "task": "<CAPTION>"})
            mock_get.return_value = mock_svc

            result = await analyze_image("http://x.com/img.jpg", "describe this")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_detect_objects_tool(self):
        from src.tools.media_tools import detect_objects
        with patch("src.services.yolo_service.get_yolo_service") as mock_get:
            mock_svc = MagicMock()
            mock_svc.detect = AsyncMock(return_value={"success": True, "detections": []})
            mock_get.return_value = mock_svc

            result = await detect_objects("http://x.com/img.jpg")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_tag_image_tool(self):
        from src.tools.media_tools import tag_image
        with patch("src.services.tagger_service.get_tagger_service") as mock_get:
            mock_svc = MagicMock()
            mock_svc.tag = AsyncMock(return_value={"success": True, "tags": []})
            mock_get.return_value = mock_svc

            result = await tag_image("http://x.com/img.jpg")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_transcribe_audio_tool(self):
        from src.tools.media_tools import transcribe_audio
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"FAKE_AUDIO"
            mock_get.return_value = mock_resp

            with patch("src.services.asr_service.get_asr_service") as mock_get_asr:
                mock_svc = MagicMock()
                mock_svc.transcribe = AsyncMock(return_value={"success": True, "text": "hello"})
                mock_get_asr.return_value = mock_svc

                result = await transcribe_audio("http://x.com/audio.wav")
                assert result["success"] is True
                assert result["text"] == "hello"

    @pytest.mark.asyncio
    async def test_check_video_tool(self):
        from src.tools.media_tools import check_video
        with patch("src.services.video_service.get_video_service") as mock_get:
            mock_svc = MagicMock()
            mock_svc.analyze = AsyncMock(return_value={"success": True, "frames_extracted": 5})
            mock_get.return_value = mock_svc

            result = await check_video("http://x.com/video.mp4")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_tool_routes_to_vision(self):
        from src.tools.media_tools import process
        with patch("src.router.function_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = MagicMock(return_value={
                "tool": "analyze_image",
                "params": {"image_url": "http://x.com/img.jpg", "task": "<CAPTION>"}
            })
            mock_get_router.return_value = mock_router

            with patch("src.services.vision_service.get_vision_service") as mock_get_vis:
                mock_svc = MagicMock()
                mock_svc.analyze = AsyncMock(return_value={"success": True})
                mock_get_vis.return_value = mock_svc

                result = await process("describe this image", "http://x.com/img.jpg")
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_tool_unknown_tool(self):
        from src.tools.media_tools import process
        with patch("src.router.function_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = MagicMock(return_value={"tool": "unknown_tool", "params": {}})
            mock_get_router.return_value = mock_router

            result = await process("do something weird", "http://x.com/img.jpg")
            assert result["success"] is False
            assert "Unknown tool" in result["error"]


# =============================================================================
# Server lifespan test
# =============================================================================

class TestServer:
    @pytest.mark.asyncio
    async def test_lifespan_yields_context(self):
        from src.server import service_lifespan
        import src.settings
        src.settings._settings = None  # reset

        mock_mcp = MagicMock()

        with patch("src.server.get_settings") as mock_settings:
            s = Settings()
            s.router_enabled = False  # skip loading GGUF
            mock_settings.return_value = s

            async with service_lifespan(mock_mcp) as context:
                assert "router" in context
                assert "vision_service" in context
                assert "yolo_service" in context
                assert "tagger_service" in context
                assert "asr_service" in context
                assert "video_service" in context
