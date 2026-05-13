"""Tests for service handlers through the unified model_engine system.

Tests the orchestrator __call__ method on handlers by mocking subprocess calls
and model inference.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_espeak_orchestrator():
    from services.model_engine.handlers.espeak.orchestrator import EspeakOrchestrator
    return EspeakOrchestrator()


# ─── Espeak Handler Tests ─────────────────────────────────────────────────


class TestEspeakHandler:

    def test_synthesize_with_text_returns_audio(self):
        orch = _make_espeak_orchestrator()
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            result = orch({"text": "hello"})
            assert result["status"] == "success"
            import base64
            assert base64.b64decode(result["data"])[:4] == b"RIFF"

    def test_synthesize_with_voice_param(self):
        orch = _make_espeak_orchestrator()
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            orch({"text": "hello", "voice": "en-us"})
            cmd = mock_run.call_args[0][0]
            assert "en-us" in cmd

    def test_subprocess_failure_raises(self):
        orch = _make_espeak_orchestrator()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "espeak-ng", stderr="fatal error")
            with pytest.raises(subprocess.CalledProcessError):
                orch({"text": "hello"})
