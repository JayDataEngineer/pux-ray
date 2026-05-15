"""E2E setup verification — validates the entire infrastructure is ready.

Run ON the GPU server (or any machine with models + repos):
    task test -- tests/test_e2e_setup.py

Checks:
1. All models exist on disk with correct size
2. All cloned repos exist with their venvs
3. All creative tool venvs can import their core dependencies
4. llama.cpp is built and server binary exists
5. Config resolves correctly for all services

Skips gracefully if MODELS_ROOT or repos don't exist (local dev PC).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from registry.config import Config
from registry.models import ModelRegistry

RAY_ROOT = Path(__file__).resolve().parent.parent
INFRA_DIR = RAY_ROOT / "infra"
REPOS_DIR = INFRA_DIR / "repos"
VENV_PYTHON = RAY_ROOT / ".venv" / "bin" / "python"


def _models_root() -> Path | None:
    try:
        root = Path(Config().models_root)
        return root if root.exists() else None
    except Exception:
        return None


def _has_repos() -> bool:
    return REPOS_DIR.is_dir() and any(REPOS_DIR.iterdir())


# ---------------------------------------------------------------------------
# Models — every downloadable model exists on disk
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _models_root(), reason="MODELS_ROOT not available")
class TestAllModelsOnDisk:
    """Every downloadable model must exist with correct size."""

    def test_all_downloadable_models_exist(self):
        registry = ModelRegistry()
        missing = []
        for cat, models in registry.data.items():
            if not isinstance(models, dict):
                continue
            for name, meta in models.items():
                if not isinstance(meta, dict):
                    continue
                download = meta.get("download", "")
                source = meta.get("source") or ""
                if download == "skip" or (not source and download not in ("file", "snapshot", "civitai")):
                    continue
                try:
                    path = registry.get_path(cat, name)
                    if not path.exists():
                        missing.append(f"{cat}/{name} -> {path}")
                except Exception as e:
                    missing.append(f"{cat}/{name}: {e}")
        assert missing == [], f"Missing models:\n" + "\n".join(missing)

    def test_all_models_minimum_size(self):
        """Downloaded models should be at least 30% of expected size."""
        registry = ModelRegistry()
        small = []
        for cat, models in registry.data.items():
            if not isinstance(models, dict):
                continue
            for name, meta in models.items():
                if not isinstance(meta, dict):
                    continue
                download = meta.get("download", "")
                source = meta.get("source") or ""
                if download == "skip" or (not source and download not in ("file", "snapshot", "civitai")):
                    continue
                try:
                    path = registry.get_path(cat, name)
                    if not path.exists():
                        continue
                    expected_gb = meta.get("size_gb", 0)
                    if path.is_file():
                        actual_gb = path.stat().st_size / 1e9
                    elif path.is_dir():
                        actual_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
                    else:
                        actual_gb = 0
                    if expected_gb and actual_gb < expected_gb * 0.3:
                        small.append(f"{cat}/{name}: {actual_gb:.1f}GB / {expected_gb}GB expected")
                except Exception:
                    pass
        assert small == [], f"Undersized models (possible partial download):\n" + "\n".join(small)

    def test_models_verify_command_passes(self):
        """The CLI verify command should exit 0 (all models present)."""
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "registry.cli", "models", "verify"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"models verify returned {result.returncode}:\n{result.stdout}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Repos — all GitHub repos cloned
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_repos(), reason="infra/repos/ not available")
class TestReposCloned:
    """All creative tool repos must be cloned."""

    EXPECTED_REPOS = {
        "TRELLIS.2": {"setup.sh", "README.md"},
        "AniGen": {"README.md"},
        "ACE-Step-1.5": {"pyproject.toml"},
        "see-through": {"requirements.txt"},
        "qwen_img_expert": {"pyproject.toml"},
        "llama.cpp": {"CMakeLists.txt"},
    }

    def test_all_repos_exist(self):
        missing = []
        for repo, _ in self.EXPECTED_REPOS.items():
            repo_dir = REPOS_DIR / repo
            if not repo_dir.is_dir():
                missing.append(repo)
        assert missing == [], f"Missing repos: {missing}"

    def test_repos_have_key_files(self):
        missing = []
        for repo, key_files in self.EXPECTED_REPOS.items():
            repo_dir = REPOS_DIR / repo
            if not repo_dir.is_dir():
                continue
            for f in key_files:
                if not (repo_dir / f).exists():
                    missing.append(f"{repo}/{f}")
        assert missing == [], f"Missing key files: {missing}"


# ---------------------------------------------------------------------------
# Venvs — all creative tool venvs work
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_repos(), reason="infra/repos/ not available")
class TestVenvsWork:
    """All creative tool venvs must be functional."""

    def test_trellis_venv(self):
        venv_py = REPOS_DIR / "TRELLIS.2" / ".venv" / "bin" / "python"
        if not venv_py.exists():
            pytest.skip("TRELLIS.2 venv not built yet")
        result = subprocess.run(
            [str(venv_py), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"TRELLIS.2 venv broken: {result.stderr}"

    def test_anigen_venv(self):
        venv_py = REPOS_DIR / "AniGen" / ".venv" / "bin" / "python"
        if not venv_py.exists():
            pytest.skip("AniGen venv not built yet")
        result = subprocess.run(
            [str(venv_py), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"AniGen venv broken: {result.stderr}"

    def test_ace_step_venv(self):
        venv_py = REPOS_DIR / "ACE-Step-1.5" / ".venv" / "bin" / "python"
        if not venv_py.exists():
            pytest.skip("ACE-Step venv not built yet")
        result = subprocess.run(
            [str(venv_py), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"ACE-Step venv broken: {result.stderr}"

    def test_llama_cpp_built(self):
        llama_server = REPOS_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
        if not (REPOS_DIR / "llama.cpp").is_dir():
            pytest.skip("llama.cpp not cloned yet")
        if not llama_server.exists():
            pytest.skip("llama.cpp not built yet")
        assert llama_server.is_file(), f"llama-server not a file: {llama_server}"


# ---------------------------------------------------------------------------
# Config — all service paths resolve
# ---------------------------------------------------------------------------

class TestConfigResolves:
    """Config must resolve all service paths without errors."""

    def test_models_root_set(self):
        root = Config().models_root
        assert root, "models_root not configured"

    def test_llama_server_path(self):
        path = Config().get("binaries.llama_server", "")
        assert path, "binaries.llama_server not configured"

    def test_all_creative_tools_configured(self):
        config = Config()
        tools = ["trellis", "anigen", "ace_step", "see_through"]
        missing = []
        for tool in tools:
            section = config.get(f"services.creative.{tool}", {})
            # Tools running through Wan2GP may not need venv/script config
            # They only need them if configured as standalone subprocesses
            if section.get("venv_python") and not section.get("script"):
                missing.append(f"{tool}.script (has venv but no script)")
            if section.get("script") and not section.get("venv_python"):
                missing.append(f"{tool}.venv_python (has script but no venv)")
        assert missing == [], f"Inconsistent config: {missing}"
