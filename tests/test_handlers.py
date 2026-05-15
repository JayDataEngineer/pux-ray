"""CPU handler tests — espeak and faster_whisper generate() with mocked deps.

Tests _Pipeline.generate() directly for CPU-only handlers that don't need
GPU, model files, or heavy dependencies. Subprocess/network calls are mocked.
"""
from __future__ import annotations

import base64
import importlib
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_CUSTOM_MODELS_DIR = Path(__file__).resolve().parent.parent / "services" / "wan2gp" / "custom_models"


def _import_handler(import_path: str):
    """Import handler with pip package collision handling."""
    try:
        mod = importlib.import_module(import_path)
        mod_file = getattr(mod, "__file__", "") or ""
        if "custom_models" in mod_file:
            return mod
    except (ImportError, ModuleNotFoundError):
        pass
    parts = import_path.split(".")
    mod_file = _CUSTOM_MODELS_DIR.joinpath(*parts[:-1]) / (parts[-1] + ".py")
    if not mod_file.exists():
        raise ImportError(f"Handler file not found: {mod_file}")
    spec = importlib.util.spec_from_file_location(import_path, str(mod_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Espeak Handler ─────────────────────────────────────────────────────────


class TestEspeakGenerate:

    @pytest.fixture
    def espeak_pipeline(self):
        from espeak.espeak_handler import _Pipeline
        return _Pipeline("espeak-ng")

    @pytest.mark.unit
    @pytest.mark.handler
    def test_synthesize_returns_audio(self, espeak_pipeline):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            result = espeak_pipeline.generate(input_prompt="hello")
            assert result["status"] == "success"
            assert result["media_type"] == "audio/wav"
            audio = base64.b64decode(result["data"])
            assert audio[:4] == b"RIFF"

    @pytest.mark.unit
    @pytest.mark.handler
    def test_voice_param_forwarded(self, espeak_pipeline):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            espeak_pipeline.generate(input_prompt="hello", voice="en-us")
            cmd = mock_run.call_args[0][0]
            assert "en-us" in cmd

    @pytest.mark.unit
    @pytest.mark.handler
    def test_speed_param_forwarded(self, espeak_pipeline):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            espeak_pipeline.generate(input_prompt="hello", speed=200)
            cmd = mock_run.call_args[0][0]
            assert "200" in cmd

    @pytest.mark.unit
    @pytest.mark.handler
    def test_text_from_kw_arg(self, espeak_pipeline):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            espeak_pipeline.generate(text="from kwarg")
            cmd = mock_run.call_args[0][0]
            assert "from kwarg" in cmd

    @pytest.mark.unit
    @pytest.mark.handler
    def test_empty_text_raises(self, espeak_pipeline):
        with pytest.raises(ValueError, match="text required"):
            espeak_pipeline.generate()

    @pytest.mark.unit
    @pytest.mark.handler
    def test_subprocess_failure_raises(self, espeak_pipeline):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "espeak-ng", stderr="fatal error"
            )
            with pytest.raises(subprocess.CalledProcessError):
                espeak_pipeline.generate(input_prompt="hello")

    @pytest.mark.unit
    @pytest.mark.handler
    def test_temp_file_cleanup_on_success(self, espeak_pipeline):
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            espeak_pipeline.generate(input_prompt="hello")

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        leaked = files_after - files_before
        assert leaked == set(), f"Temp files leaked: {leaked}"

    @pytest.mark.unit
    @pytest.mark.handler
    def test_temp_file_cleanup_on_failure(self, espeak_pipeline):
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("espeak-ng", 30)
            with pytest.raises(subprocess.TimeoutExpired):
                espeak_pipeline.generate(input_prompt="hello")

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        leaked = files_after - files_before
        assert leaked == set(), f"Temp files leaked on failure: {leaked}"


# ─── Faster-Whisper Handler ────────────────────────────────────────────────


class TestFasterWhisperContract:
    """Test faster_whisper handler contract without loading the model."""

    @pytest.mark.unit
    @pytest.mark.handler
    def test_handler_contract(self):
        mod = _import_handler("faster_whisper.faster_whisper_handler")
        fh = mod.family_handler
        types = fh.query_supported_types()
        assert "faster_whisper" in types
        family = fh.query_model_family()
        assert family == "faster_whisper"
        infos = fh.query_family_infos()
        assert "faster_whisper" in infos
        assert isinstance(infos["faster_whisper"][0], int)

    @pytest.mark.unit
    @pytest.mark.handler
    def test_pipeline_has_generate(self):
        mod = _import_handler("faster_whisper.faster_whisper_handler")
        pipeline_cls = mod._Pipeline
        assert hasattr(pipeline_cls, "generate")
        assert callable(getattr(pipeline_cls, "generate"))
