"""Tests for temp file cleanup — validates try/finally in espeak handler.

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

    def _make_pipeline(self):
        from models.espeak.espeak_handler import _Pipeline
        return _Pipeline("espeak-ng")

    def test_temp_file_removed_on_success(self):
        pipeline = self._make_pipeline()
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))

        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")

            mock_run.side_effect = write_output
            pipeline.generate(input_prompt="hello")

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Temp files leaked: {new_files}"

    def test_temp_file_removed_on_subprocess_failure(self):
        pipeline = self._make_pipeline()
        files_before = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("espeak-ng", 30)
            with pytest.raises(subprocess.TimeoutExpired):
                pipeline.generate(input_prompt="hello")

        files_after = set(Path(tempfile.gettempdir()).glob("tmp*.wav"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Temp files leaked on failure: {new_files}"
