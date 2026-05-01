"""Tests for model registry — validates 100% infrastructure as code.

These tests run LOCALLY (no GPU, no Ray) and verify:
- Every model in the registry has a valid download source
- No model requires manual download
- All HF/Civitai URLs are parseable
- The registry structure is consistent
"""

from __future__ import annotations

import pytest

from registry.models import ModelRegistry


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
        assert manual == [], f"Models with download=manual: {manual}"

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
        valid_modes = {"file", "snapshot", "civitai", "skip", ""}
        for cat, models in registry.data.items():
            for name, meta in models.items():
                mode = meta.get("download", "")
                assert mode in valid_modes, f"{cat}/{name}: unknown download mode '{mode}'"


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
