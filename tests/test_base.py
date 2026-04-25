"""Unit tests for CLIToolMixin — relative path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest
from unittest import mock

from registry.config import Config


class TestPathResolution:
    """Test _resolve_path in services.base (static, no Ray needed)."""

    @staticmethod
    def _resolve_path(raw: str, project_root: Path) -> Path:
        """Replicate services.base.CLIToolMixin._resolve_path."""
        # Import is deliberately local to avoid Ray import during unit tests
        from services.base import CLIToolMixin
        return CLIToolMixin._resolve_path(raw, project_root)

    def test_absolute_path_passes_through(self):
        result = self._resolve_path("/absolute/path/to/python", Path("/home/user"))
        assert str(result) == "/absolute/path/to/python"

    def test_relative_path_resolves_to_project_root(self):
        result = self._resolve_path("infra/repos/TRELLIS.2/.venv/bin/python",
                                    Path("/home/user/ray"))
        assert str(result) == "/home/user/ray/infra/repos/TRELLIS.2/.venv/bin/python"

    def test_relative_with_dots(self):
        result = self._resolve_path("../tools/wrapper.py", Path("/home/user/ray"))
        assert str(result) == "/home/user/tools/wrapper.py"

    def test_relative_config_paths(self):
        root = Path("/opt/tech-noir")
        for tool in ["trellis", "anigen", "ace_step"]:
            path = f"infra/repos/{tool.upper()}/.venv/bin/python"
            resolved = self._resolve_path(path, root)
            assert str(resolved) == f"/opt/tech-noir/infra/repos/{tool.upper()}/.venv/bin/python"


class TestSubprocessRun:
    """Test _run_cli subprocess execution (mocked)."""

    def test_run_cli_basic(self):
        from services.base import CLIToolMixin
        import subprocess

        mixin = CLIToolMixin()
        # Set up required attrs (normally done by _init_cli)
        mixin._venv_python = "/usr/bin/python3"
        mixin._script = "-c"
        mixin._working_dir = "/tmp"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr="",
            )
            result = mixin._run_cli(["print('hello')"], timeout=30)
            mock_run.assert_called_once()
            assert result.stdout == "ok"

    def test_run_cli_with_extra_env(self):
        from services.base import CLIToolMixin
        import subprocess
        import os

        mixin = CLIToolMixin()
        mixin._venv_python = "/usr/bin/python3"
        mixin._script = "-c"
        mixin._working_dir = "/tmp"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            )
            mixin._run_cli(["pass"], timeout=30, extra_env={"ACESTEP_CHECKPOINTS_DIR": "/models"})
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"] is not None
            assert call_kwargs["env"]["ACESTEP_CHECKPOINTS_DIR"] == "/models"

    def test_run_cli_failure_raises(self):
        from services.base import CLIToolMixin
        import subprocess

        mixin = CLIToolMixin()
        mixin._venv_python = "/usr/bin/python3"
        mixin._script = "-c"
        mixin._working_dir = "/tmp"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="fatal error",
            )
            with pytest.raises(RuntimeError, match="CLI failed"):
                mixin._run_cli(["raise"])
