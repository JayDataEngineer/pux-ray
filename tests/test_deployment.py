"""Unit tests for Wan2GP deployment module — discovery, payload security, key mapping.

Tests the Wan2GP service layer WITHOUT loading actual models or needing GPU.
All model loading is mocked; only the discovery/routing/security logic is tested.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── CUSTOM_HANDLERS Completeness ───────────────────────────────────────────


class TestCustomHandlersList:

    @pytest.mark.unit
    def test_all_11_handlers_registered(self):
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        assert len(CUSTOM_HANDLERS) == 11

    @pytest.mark.unit
    def test_no_duplicate_handlers(self):
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        assert len(CUSTOM_HANDLERS) == len(set(CUSTOM_HANDLERS))

    @pytest.mark.unit
    def test_all_handler_modules_exist(self):
        """Every import path in CUSTOM_HANDLERS maps to a real file in the fork."""
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        project_root = Path(__file__).resolve().parent.parent
        fork_models = project_root / "opt" / "wan2gp" / "models"
        for handler_path in CUSTOM_HANDLERS:
            # Strip "models." prefix to get the path within models/
            rel = handler_path
            if rel.startswith("models."):
                rel = rel[len("models."):]
            mod_file = fork_models / (rel.replace(".", "/") + ".py")
            assert mod_file.exists(), f"Handler file missing: {mod_file} (from {handler_path})"


# ─── _derive_key() ─────────────────────────────────────────────────────────


class TestDeriveKey:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_type,handler_path,expected",
        [
            ("t2v-14B", "models.wan.wan_handler", "wan/t2v-14B"),
            ("ace-step-v1", "models.TTS.ace_step_handler", "tts/ace-step-v1"),
            ("faster_whisper", "models.faster_whisper.faster_whisper_handler", "faster_whisper/faster_whisper"),
            ("hunyuan-t2v", "models.hyvideo.hunyuan_handler", "hunyuan/hunyuan-t2v"),
            ("flux-dev", "models.flux.flux_handler", "flux/flux-dev"),
            ("i2v-14B", "models.wan.ovi_handler", "wan/i2v-14B"),
        ],
        ids=lambda x: str(x).split("/")[-1] if "/" in str(x) else str(x),
    )
    def test_derive_key(self, model_type, handler_path, expected):
        from services.wan2gp.deployment import _derive_key
        assert _derive_key(model_type, handler_path) == expected


# ─── _build_generate_kwargs() ───────────────────────────────────────────────


class TestBuildGenerateKwargs:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "payload,expected_key,expected_val",
        [
            ({"prompt": "hello"}, "input_prompt", "hello"),
            ({"negative_prompt": "blurry"}, "n_prompt", "blurry"),
            ({"steps": 10}, "sampling_steps", 10),
            ({"guidance": 7.5}, "guide_scale", 7.5),
            ({"frames": 60}, "frame_num", 60),
        ],
        ids=["prompt", "negative", "steps", "guidance", "frames"],
    )
    def test_key_mapping(self, payload, expected_key, expected_val):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs(payload, {})
        assert result[expected_key] == expected_val

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "key,value",
        [
            ("seed", 42),
            ("width", 512),
            ("height", 512),
            ("fps", 24),
            ("temperature", 0.7),
            ("top_p", 0.9),
            ("text", "hello"),
            ("voice", "af_bella"),
            ("language", "en"),
        ],
    )
    def test_safe_passthrough_keys(self, key, value):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({key: value}, {})
        assert result[key] == value

    @pytest.mark.unit
    def test_defaults_applied(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({}, {"steps": 20, "guidance": 5.0})
        assert result["steps"] == 20
        assert result["guidance"] == 5.0

    @pytest.mark.unit
    def test_payload_overrides_defaults(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"steps": 50}, {"steps": 20})
        assert result["steps"] == 50

    @pytest.mark.unit
    def test_default_seed_minus_one(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({}, {})
        assert result["seed"] == -1

    @pytest.mark.unit
    def test_explicit_seed_preserved(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs({"seed": 123}, {})
        assert result["seed"] == 123

    @pytest.mark.unit
    def test_blocked_keys_not_forwarded(self, caplog):
        """Keys in _BLOCKED_KEYS must never appear in kwargs."""
        from services.wan2gp.deployment import _build_generate_kwargs, _BLOCKED_KEYS
        payload = {k: "evil" for k in _BLOCKED_KEYS}
        with caplog.at_level(logging.DEBUG):
            result = _build_generate_kwargs(payload, {})
        for key in _BLOCKED_KEYS:
            assert key not in result, f"Blocked key '{key}' leaked into kwargs"
        assert any("Blocked key" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_custom_key_map_override(self):
        from services.wan2gp.deployment import _build_generate_kwargs
        result = _build_generate_kwargs(
            {"custom_key": "value"},
            {},
            key_map={"custom_key": "mapped_key"},
        )
        assert result["mapped_key"] == "value"


# ─── Payload Security ───────────────────────────────────────────────────────


class TestPayloadSecurity:

    @pytest.mark.unit
    def test_blocked_keys_covers_dangerous_paths(self):
        from services.wan2gp.deployment import _BLOCKED_KEYS
        dangerous = {
            "input_custom", "output_dir", "model_filename",
            "lora_dir", "input_frames", "input_video",
        }
        assert dangerous.issubset(_BLOCKED_KEYS), (
            f"Missing dangerous keys: {dangerous - _BLOCKED_KEYS}"
        )

    @pytest.mark.unit
    def test_safe_keys_dont_overlap_blocked(self):
        from services.wan2gp.deployment import _SAFE_PASSTHROUGH, _BLOCKED_KEYS
        overlap = _SAFE_PASSTHROUGH & _BLOCKED_KEYS
        assert overlap == set(), f"Keys in both safe and blocked: {overlap}"


# ─── _CPU_ONLY_TYPES ───────────────────────────────────────────────────────


class TestCPUOnlyTypes:

    @pytest.mark.unit
    def test_cpu_types_are_custom_handlers(self):
        from services.wan2gp.deployment import _CPU_ONLY_TYPES
        assert "kokoro" in _CPU_ONLY_TYPES
        assert "espeak" in _CPU_ONLY_TYPES
        assert "faster_whisper" in _CPU_ONLY_TYPES

    @pytest.mark.unit
    def test_gpu_types_not_in_cpu_list(self):
        from services.wan2gp.deployment import _CPU_ONLY_TYPES
        gpu_types = {"moss-soundeffect"}
        assert not gpu_types & _CPU_ONLY_TYPES


# ─── _WEIGHT_SEARCH Registry Mapping ────────────────────────────────────────


class TestWeightSearchMapping:

    @pytest.mark.unit
    def test_all_gpu_custom_handlers_have_weight_search(self):
        from services.wan2gp.deployment import _WEIGHT_SEARCH, _CPU_ONLY_TYPES
        # All custom GPU handlers should have a weight search entry
        expected_gpu = {
            "moss-soundeffect",
            "vibevoice-asr", "vibevoice-tts",
        }
        missing = expected_gpu - set(_WEIGHT_SEARCH.keys())
        assert missing == set(), f"GPU handlers missing from _WEIGHT_SEARCH: {missing}"

    @pytest.mark.unit
    def test_weight_search_entries_are_tuples(self):
        from services.wan2gp.deployment import _WEIGHT_SEARCH
        for model_type, searches in _WEIGHT_SEARCH.items():
            assert isinstance(searches, list), f"{model_type}: not a list"
            for entry in searches:
                assert isinstance(entry, tuple), f"{model_type}: entry not a tuple"
                assert len(entry) == 2, f"{model_type}: tuple must be (category, name)"
                assert isinstance(entry[0], str), f"{model_type}: category not str"
                assert isinstance(entry[1], str), f"{model_type}: name not str"


# ─── _ensure_vendor_path() ─────────────────────────────────────────────────


class TestEnsureVendorPath:

    @pytest.mark.unit
    def test_adds_fork_to_sys_path(self):
        from services.wan2gp.deployment import _ensure_vendor_path
        # Reset so it actually runs
        import services.wan2gp.deployment as dep
        dep._ven_loaded = False
        _ensure_vendor_path()
        project_root = Path(__file__).resolve().parent.parent
        fork_root = str(project_root / "opt" / "wan2gp")
        assert fork_root in sys.path
