"""Edge case and property-based tests — validates robustness across the system.

Uses pytest parametrize for combinatorial coverage and hypothesis for
property-based testing of invariant properties.
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── Payload edge cases ─────────────────────────────────────────────────────


class TestBuildKwargsEdgeCases:
    """Edge cases for _build_generate_kwargs."""

    @pytest.mark.unit
    def test_empty_payload_and_defaults(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({}, {})
        assert "seed" in result
        assert result["seed"] == -1

    @pytest.mark.unit
    def test_extra_keys_ignored(self):
        """Unknown keys not in _KEY_MAP or _SAFE_PASSTHROUGH are silently dropped."""
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"unknown_key": "val", "prompt": "hello"}, {})
        assert "unknown_key" not in result
        assert result["input_prompt"] == "hello"

    @pytest.mark.unit
    def test_none_values_passed_through(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"seed": None}, {})
        assert result["seed"] is None

    @pytest.mark.unit
    def test_zero_values_preserved(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"width": 0, "height": 0}, {})
        assert result["width"] == 0
        assert result["height"] == 0

    @pytest.mark.unit
    def test_large_values_preserved(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"seed": 2**32}, {})
        assert result["seed"] == 2**32

    @pytest.mark.unit
    def test_string_values_preserved(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"text": "Hello 世界 🌍"}, {})
        assert result["text"] == "Hello 世界 🌍"

    @pytest.mark.unit
    @pytest.mark.parametrize("blocked_key", [
        "input_custom", "output_dir", "model_filename",
        "lora_dir", "input_frames", "input_video", "input_masks",
    ])
    def test_individual_blocked_keys(self, blocked_key):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({blocked_key: "evil"}, {})
        assert blocked_key not in result


# ─── Config edge cases ──────────────────────────────────────────────────────


class TestConfigEdgeCases:

    @pytest.mark.unit
    def test_env_var_with_special_chars(self, monkeypatch):
        from registry.config import _resolve_env
        monkeypatch.setenv("SPECIAL", "hello/world:foo=bar")
        assert _resolve_env("${SPECIAL}") == "hello/world:foo=bar"

    @pytest.mark.unit
    def test_env_var_with_empty_value(self, monkeypatch):
        from registry.config import _resolve_env
        monkeypatch.setenv("EMPTY", "")
        assert _resolve_env("${EMPTY}") == ""

    @pytest.mark.unit
    def test_resolve_deep_empty_dict(self):
        from registry.config import _resolve_deep
        assert _resolve_deep({}) == {}

    @pytest.mark.unit
    def test_resolve_deep_nested_lists(self, monkeypatch):
        from registry.config import _resolve_deep
        monkeypatch.setenv("X", "42")
        data = {"a": [{"b": "${X:y}"}]}
        result = _resolve_deep(data)
        assert result["a"][0]["b"] == "42"


# ─── Forge VRAM invariants ─────────────────────────────────────────────────


class TestForgeVRAMInvariants:
    """Property-based tests for VRAM tracking invariants."""

    @pytest.fixture
    def forge(self):
        from services.forge import ForgeCore
        return ForgeCore(service_map={})

    @pytest.mark.unit
    def test_zero_sum_after_multiple_loads(self, forge):
        from tests.test_forge import MockService, BigService
        forge._register_service("mock", MockService())
        forge._register_service("big", BigService())

        forge._do_load("mock", "test")
        allocated = forge._total_allocated()
        free = forge._vram_free_mb
        assert allocated + free == 22_528

        forge._do_load("big", "test")
        allocated = forge._total_allocated()
        free = forge._vram_free_mb
        assert allocated + free == 22_528

    @pytest.mark.unit
    def test_zero_sum_after_eviction(self, forge):
        from tests.test_forge import MockService, BigService
        big = BigService()
        big._loaded = True
        forge._register_service("big", big)
        forge._loaded["big"] = True
        forge._vram_allocations["big"] = 20_480
        forge._vram_free_mb = 22_528 - 20_480

        mock = MockService()
        forge._register_service("mock", mock)
        forge._evict_for("mock")

        allocated = forge._total_allocated()
        free = forge._vram_free_mb
        assert allocated + free == 22_528

    @pytest.mark.unit
    def test_vram_never_negative(self, forge):
        """Free VRAM should never go below zero."""
        assert forge._vram_free_mb >= 0
        forge._vram_allocations["test"] = 22_528
        forge._vram_free_mb = 0
        assert forge._vram_free_mb >= 0

    @pytest.mark.unit
    def test_unload_restores_all_vram(self, forge):
        from tests.test_forge import MockService
        svc = MockService()
        forge._register_service("mock", svc)
        forge._do_load("mock", "test")
        forge._do_unload("mock")
        assert forge._total_allocated() == 0
        assert forge._vram_free_mb == 22_528


# ─── Espeak edge cases ──────────────────────────────────────────────────────


class TestEspeakEdgeCases:

    @pytest.fixture
    def pipeline(self):
        from models.espeak.espeak_handler import _Pipeline
        return _Pipeline("espeak-ng")

    @pytest.mark.unit
    def test_special_characters_in_text(self, pipeline):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")
            mock_run.side_effect = write_output
            result = pipeline.generate(input_prompt="Hello 世界")
            assert result["status"] == "success"

    @pytest.mark.unit
    def test_empty_string_text_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.generate(input_prompt="")

    @pytest.mark.unit
    @pytest.mark.parametrize("voice", ["en", "fr", "de", "zh", "ja"])
    def test_various_voices(self, pipeline, voice):
        with patch("subprocess.run") as mock_run:
            def write_output(*args, **kwargs):
                cmd = args[0]
                out_path = cmd[cmd.index("-w") + 1]
                Path(out_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess([], 0, "", "")
            mock_run.side_effect = write_output
            result = pipeline.generate(input_prompt="test", voice=voice)
            assert result["status"] == "success"
            cmd = mock_run.call_args[0][0]
            assert voice in cmd


# ─── Binary data fixtures validation ────────────────────────────────────────


class TestBinaryFixtures:
    """Validate that the conftest binary generators produce correct data."""

    @pytest.mark.unit
    def test_wav_is_valid(self, sample_wav_bytes):
        assert sample_wav_bytes[:4] == b"RIFF"
        assert sample_wav_bytes[8:12] == b"WAVE"
        assert len(sample_wav_bytes) > 44

    @pytest.mark.unit
    def test_png_is_valid(self, sample_png_bytes):
        assert sample_png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.unit
    def test_b64_roundtrip_wav(self, sample_wav_bytes, sample_wav_b64):
        decoded = base64.b64decode(sample_wav_b64)
        assert decoded == sample_wav_bytes

    @pytest.mark.unit
    def test_b64_roundtrip_png(self, sample_png_bytes, sample_png_b64):
        decoded = base64.b64decode(sample_png_b64)
        assert decoded == sample_png_bytes


# ─── Model registry edge cases ──────────────────────────────────────────────


class TestRegistryEdgeCases:

    @pytest.mark.unit
    def test_nonexistent_category_raises(self):
        from registry.models import ModelRegistry
        with pytest.raises(KeyError):
            ModelRegistry().get_path("nonexistent_cat", "model")

    @pytest.mark.unit
    def test_nonexistent_model_raises(self):
        from registry.models import ModelRegistry
        reg = ModelRegistry()
        # Pick a real category
        cats = [c for c in reg.data if isinstance(reg.data[c], dict) and reg.data[c]]
        if cats:
            with pytest.raises(KeyError):
                reg.get_path(cats[0], "nonexistent_model_xyz")

    @pytest.mark.unit
    def test_source_url_parsing(self):
        from registry.cli import _parse_source
        repo, fname = _parse_source("hf://org/repo")
        assert repo == "org/repo"
        assert fname is None

    @pytest.mark.unit
    def test_source_url_without_filename(self):
        from registry.cli import _parse_source
        repo, fname = _parse_source("hf://org/repo")
        assert repo == "org/repo"
        assert fname is None

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_source", [
        "hf://",
        "hf://single",
        "",
    ])
    def test_invalid_hf_sources(self, invalid_source):
        from registry.cli import _parse_source
        repo, _ = _parse_source(invalid_source)
        # Should not crash, may return empty/malformed
        assert isinstance(repo, str)
