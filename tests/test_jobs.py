"""Unit tests for gateway.jobs — JobManager and task functions."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from gateway.jobs import _resolve_config_paths, _run_tool
from registry.config import Config


class TestJobConfigResolution:
    def setup_method(self):
        Config().reload()

    def teardown_method(self):
        Config().reload()

    def test_resolves_see_through_paths(self, tmp_path):
        fake_yaml = {
            "services": {
                "creative": {
                    "see_through": {
                        "venv_python": "infra/repos/see-through/.venv/bin/python",
                        "script": "infra/repos/see-through/inference/scripts/inference_psd.py",
                        "working_dir": "infra/repos/see-through",
                    }
                }
            }
        }
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("unused: true\n")
        Config._instance._data = fake_yaml

        venv, script, cwd = _resolve_config_paths("services.creative.see_through")
        root = Config().project_root
        assert (root / "infra/repos/see-through/.venv/bin/python").resolve() == venv.resolve()

    def test_resolves_ace_step_paths(self, tmp_path):
        fake_yaml = {
            "services": {
                "creative": {
                    "ace_step": {
                        "venv_python": "infra/repos/ACE-Step-1.5/.venv/bin/python",
                        "script": "infra/repos/ACE-Step-1.5/cli.py",
                        "working_dir": "infra/repos/ACE-Step-1.5",
                    }
                }
            }
        }
        Config._instance._data = fake_yaml

        venv, script, cwd = _resolve_config_paths("services.creative.ace_step")
        root = Config().project_root
        assert (root / "infra/repos/ACE-Step-1.5/.venv/bin/python").resolve() == venv.resolve()
        assert (root / "infra/repos/ACE-Step-1.5/cli.py").resolve() == script.resolve()


class TestRunTool:
    def test_run_tool_success(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"output bytes", stderr=b"",
            )
            result = _run_tool(
                Path("/fake/venv/bin/python"),
                Path("/fake/script.py"),
                Path("/fake/cwd"),
                ["--flag", "value"],
            )
            assert result == b"output bytes"
            mock_run.assert_called_once()

    def test_run_tool_failure_raises(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"something broke",
            )
            with pytest.raises(RuntimeError, match="Tool failed"):
                _run_tool(
                    Path("/fake/bin/python"),
                    Path("/fake/script.py"),
                    Path("/fake"),
                    [],
                )
