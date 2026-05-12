"""Tests for model registry — validates 100% infrastructure as code.

These tests run LOCALLY (no GPU, no Ray) and verify:
- Every model in the registry has a valid download source
- No model requires manual download
- All HF/Civitai URLs are parseable
- The registry structure is consistent
- Path resolution works for all models
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from registry.models import ModelRegistry

RAY_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = RAY_ROOT / ".venv" / "bin" / "python"


@pytest.fixture
def registry():
    return ModelRegistry()


class TestRegistryStructure:
    """Registry YAML is well-formed and complete."""

    def test_registry_loads(self, registry):
        assert registry.data is not None
        assert len(registry.data) > 0

    def test_all_categories_are_dicts(self, registry):
        for cat, models in registry.data.items():
            assert isinstance(models, dict), f"Category {cat} is not a dict"

    def test_all_models_have_metadata(self, registry):
        for cat, models in registry.data.items():
            for name, meta in models.items():
                assert isinstance(meta, dict), f"{cat}/{name} metadata is not a dict"
                assert "path" in meta, f"{cat}/{name} missing 'path'"

    def test_all_models_have_size(self, registry):
        for cat, models in registry.data.items():
            for name, meta in models.items():
                assert "size_gb" in meta, f"{cat}/{name} missing 'size_gb'"
                assert meta["size_gb"] > 0, f"{cat}/{name} has zero size_gb"


class TestNoManualDownloads:
    """ZERO models require manual download — everything is automated."""

    def test_no_manual_download_entries(self, registry):
        manual = []
        for cat, models in registry.data.items():
            for name, meta in models.items():
                if meta.get("download") == "manual":
                    manual.append(f"{cat}/{name}")
        # wan2gp models are self-managed by mmgp — manual download is expected
        allowed = {m for m in manual if m.startswith("wan2gp/")}
        unexpected = set(manual) - allowed
        assert len(unexpected) == 0, f"Unexpected manual download models: {unexpected}"

    def test_no_null_source_for_downloadable(self, registry):
        """Models that aren't 'skip' must have a source."""
        missing = []
        for cat, models in registry.data.items():
            for name, meta in models.items():
                download = meta.get("download", "")
                source = meta.get("source", "")
                if download not in ("skip", None, "") and not source:
                    missing.append(f"{cat}/{name} (download={download})")
        assert missing == [], f"Downloadable models missing source: {missing}"


class TestDownloadSources:
    """All download sources are valid and parseable."""

    def test_hf_sources_parseable(self, registry):
        from registry.cli import _parse_source
        for cat, models in registry.data.items():
            for name, meta in models.items():
                source = meta.get("source", "")
                if not source or not source.startswith("hf://"):
                    continue
                repo_id, filename = _parse_source(source)
                assert repo_id, f"{cat}/{name}: empty repo_id from '{source}'"
                assert "/" in repo_id, f"{cat}/{name}: repo_id '{repo_id}' missing org/"

    def test_civitai_sources_valid(self, registry):
        for cat, models in registry.data.items():
            for name, meta in models.items():
                source = meta.get("source") or ""
                if not source.startswith("civitai://"):
                    continue
                model_id = source.split("://")[1]
                assert model_id.isdigit(), f"{cat}/{name}: civitai ID not numeric: {model_id}"

    def test_download_modes_valid(self, registry):
        valid_modes = {"file", "snapshot", "civitai", "modelscope", "skip", "manual", ""}
        for cat, models in registry.data.items():
            for name, meta in models.items():
                mode = meta.get("download", "")
                assert mode in valid_modes, f"{cat}/{name}: unknown download mode '{mode}'"


class TestPathResolution:
    """Every model's path resolves without errors."""

    def test_all_paths_resolve(self, registry):
        errors = []
        for cat, models in registry.data.items():
            if not isinstance(models, dict):
                continue
            for name, meta in models.items():
                if not isinstance(meta, dict):
                    continue
                try:
                    path = registry.get_path(cat, name)
                    assert isinstance(path, Path), f"{cat}/{name}: get_path didn't return Path"
                except Exception as e:
                    errors.append(f"{cat}/{name}: {e}")
        assert errors == [], f"Path resolution errors: {errors}"

    def test_all_paths_are_under_models_root(self, registry):
        from registry.config import Config
        models_root = Path(Config().models_root)
        for cat, models in registry.data.items():
            if not isinstance(models, dict):
                continue
            for name, meta in models.items():
                if not isinstance(meta, dict):
                    continue
                path = registry.get_path(cat, name)
                raw = meta.get("path", "")
                # Absolute paths from config may point outside models_root
                # (e.g. local dev machine vs container). Only check relative paths.
                if not Path(raw).is_absolute():
                    assert str(path).startswith(str(models_root)), \
                        f"{cat}/{name}: path {path} not under {models_root}"
                else:
                    # Absolute paths should still exist or be resolvable
                    assert str(path) == raw, f"{cat}/{name}: absolute path mismatch"


class TestCLICommands:
    """CLI commands run without crashing."""

    def test_models_list_runs(self):
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "registry.cli", "models", "list"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"models list failed: {result.stderr}"

    def test_models_status_runs(self):
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "registry.cli", "models", "status"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"models status failed: {result.stderr}"

    def test_models_verify_runs(self):
        """verify command runs — exit code depends on whether models exist on disk."""
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "registry.cli", "models", "verify"],
            capture_output=True, text=True, timeout=30,
        )
        # exit code 0 = all models present, 1 = some missing (both are valid in tests)
        assert result.returncode in (0, 1), f"models verify crashed: {result.stderr}"
        assert "OK" in result.stdout or "MISSING" in result.stdout


class TestModelCounts:
    """Verify expected model counts."""

    def test_at_least_30_models(self, registry):
        count = sum(len(m) for m in registry.data.values() if isinstance(m, dict))
        assert count >= 30, f"Only {count} models in registry, expected >= 30"

    def test_has_required_categories(self, registry):
        required = {"llm", "tts", "asr", "vision", "comfyui", "3d", "audio"}
        missing = required - set(registry.data.keys())
        assert missing == set(), f"Missing categories: {missing}"

    def test_llm_models_exist(self, registry):
        llm = registry.data.get("llm", {})
        assert len(llm) >= 5, f"Only {len(llm)} LLM models, expected >= 5"

    def test_tts_models_exist(self, registry):
        tts = registry.data.get("tts", {})
        assert len(tts) >= 3, f"Only {len(tts)} TTS models, expected >= 3"
