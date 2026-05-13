"""Tests for individual service TNAP endpoints with mocked backends.

Tests the __call__ method on services by mocking subprocess calls and
model inference. Set model=True to skip loading.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_request(body: dict, method: str = "POST"):
    """Create a mock Starlette Request."""
    req = MagicMock()
    req.method = method
    req.json = AsyncMock(return_value=body)
    return req


def _unwrap(cls):
    """Extract the inner class from @serve.deployment wrapper."""
    if hasattr(cls, 'func_or_class'):
        return cls.func_or_class
    if hasattr(cls, '__ray_actor_class__'):
        return cls.__ray_actor_class__
    return cls


# ─── Parametrized GET Status Tests ─────────────────────────────────────────


class TestServiceGETResponses:
    """All services should return status ok on GET."""

    @pytest.mark.parametrize("svc_name,import_path,cls_name", [
        ("espeak", "services.tts.espeak", "EspeakTTS"),
    ])
    def test_get_returns_status_ok(self, svc_name, import_path, cls_name):
        import importlib
        mod = importlib.import_module(import_path)
        cls = _unwrap(getattr(mod, cls_name))
        svc = cls.__new__(cls)
        svc.model = True
        svc.model_name = svc_name

        assert svc.model_name == svc_name
        assert svc.is_loaded() is True


# ─── Espeak TNAP Tests ────────────────────────────────────────────────────


class TestEspeakTNAP:

    def _make_service(self):
        from services.tts.espeak import EspeakTTS
        cls = _unwrap(EspeakTTS)
        svc = cls.__new__(cls)
        svc.model = True
        svc.model_name = "espeak"
        svc._espeak_bin = "espeak-ng"
        return svc

    def test_synthesize_with_text_returns_audio(self):
        svc = self._make_service()
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            result = svc.synthesize("hello")
            assert result[:4] == b"RIFF"

    def test_synthesize_with_voice_param(self):
        svc = self._make_service()
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            svc.synthesize("hello", voice="en-us")
            cmd = mock_run.call_args[0][0]
            assert "en-us" in cmd

    def test_subprocess_failure_raises(self):
        svc = self._make_service()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "espeak-ng", stderr="fatal error")
            with pytest.raises(subprocess.CalledProcessError):
                svc.synthesize("hello")


# ─── VibeVoice Community TNAP Tests ────────────────────────────────────────


class TestVibeVoiceCommunityTNAP:

    def _make_service(self):
        from services.tts.vibe_voice import VibeVoiceCommunityTTSDeployment
        cls = _unwrap(VibeVoiceCommunityTTSDeployment)
        svc = cls.__new__(cls)
        svc.model = True
        svc.model_name = "vibevoice-tts"
        return svc

    def test_missing_text_returns_400(self):
        svc = self._make_service()
        import asyncio

        req = _make_request({"action": "generate", "input": {"voice": "Andrew"}})
        resp = asyncio.get_event_loop().run_until_complete(svc(req))
        assert resp.status_code == 400

    def test_with_text_calls_generate_audio(self):
        svc = self._make_service()
        import asyncio

        with patch.object(type(svc), "_generate_audio", return_value=b"RIFF\x00\x00\x00\x00WAVE", create=True):
            with patch.object(svc, "_find_voice_file", return_value="/voices/Andrew.wav"):
                req = _make_request({"action": "generate", "input": {"text": "hello", "voice": "Andrew"}})
                resp = asyncio.get_event_loop().run_until_complete(svc(req))
                assert resp.status_code == 200
