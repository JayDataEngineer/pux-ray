"""Tests for temp file cleanup — validates try/finally in espeak orchestrator.

Ensures NamedTemporaryFile(delete=False) files are always unlinked,
even when subprocess calls fail or timeout.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEspeakTempCleanup:

    def _make_orchestrator(self):
        from services.model_engine.handlers.espeak.orchestrator import EspeakOrchestrator
        orch = EspeakOrchestrator()
        return orch

    def test_temp_file_removed_on_success(self):
        orch = self._make_orchestrator()
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))

        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            orch({"text": "hello"})

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Temp files leaked: {new_files}"

    def test_temp_file_removed_on_subprocess_failure(self):
        orch = self._make_orchestrator()
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("espeak-ng", 30)
            with pytest.raises(subprocess.TimeoutExpired):
                orch({"text": "hello"})

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Temp files leaked on failure: {new_files}"
