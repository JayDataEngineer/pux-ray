"""Integration tests for setup pipeline — validates task setup works end-to-end.

These tests verify the LOCAL setup module (infra.setup) works correctly.
No GPU, no Ray needed. Run with: task test -- tests/test_infra_setup.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RAY_ROOT = Path(__file__).resolve().parent.parent
INFRA_DIR = RAY_ROOT / "infra"
REPOS_DIR = INFRA_DIR / "repos"
VENV_PYTHON = RAY_ROOT / ".venv" / "bin" / "python"


class TestSetupModule:
    """Python setup module works correctly."""

    def test_clone_module_importable(self):
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", "from infra.setup.clone import REPOS; print(len(REPOS))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert int(result.stdout.strip()) >= 4

    def test_venvs_module_importable(self):
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", "from infra.setup.venvs import TOOLS; print(len(TOOLS))"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert int(result.stdout.strip()) >= 4

    def test_clone_module_lists_repos(self):
        from infra.setup.clone import REPOS
        # Bare-metal tools only (Docker tools clone inside Dockerfile)
        expected = {"ace-step", "see-through", "qwen", "gpt-sovits", "comfyui", "llama"}
        assert set(REPOS.keys()) == expected

    def test_venvs_module_lists_tools(self):
        from infra.setup.venvs import TOOLS
        # Bare-metal tools only (Docker tools built via `python -m infra.setup docker`)
        expected = {"ace-step", "see-through", "gpt-sovits", "qwen", "comfyui", "llama"}
        assert set(TOOLS.keys()) == expected

    def test_all_repos_have_valid_urls(self):
        from infra.setup.clone import REPOS
        for name, (url, dest) in REPOS.items():
            assert url.startswith("https://github.com/"), f"{name}: invalid URL {url}"
            assert len(dest) > 0, f"{name}: empty dest"


class TestNoShellScriptDependencies:
    """Verify boot system doesn't depend on shell scripts."""

    def test_ray_cluster_start_is_python(self):
        """_start_ray should not call start_cluster.sh."""
        code = Path(RAY_ROOT / "boot" / "services.py").read_text()
        assert "start_cluster.sh" not in code, "boot/services.py still references start_cluster.sh"

    def test_taskfile_uses_python(self):
        """Taskfile should not call bash scripts for setup."""
        code = Path(RAY_ROOT / "Taskfile.yml").read_text()
        assert "setup_venvs.sh" not in code, "Taskfile still references setup_venvs.sh"
        assert "clone_repos.sh" not in code, "Taskfile still references clone_repos.sh"
