"""Unit tests for Config — env var resolution, relative path handling.

Uses monkeypatch for env var isolation, tmp_path for filesystem fixtures,
and parametrize for edge case coverage.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from registry.config import Config, _resolve_env, _resolve_deep, _PROJECT_ROOT


# ─── Env Var Resolution ─────────────────────────────────────────────────────


class TestEnvVarResolution:
    """Test ${VAR:default} token resolution."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_val,template,expected",
        [
            ({"FOO": "bar"}, "${FOO}", "bar"),
            ({"FOO": "bar"}, "${FOO:baz}", "bar"),
            ({}, "${MISSING:default_val}", "default_val"),
            ({}, "${MISSING}", "${MISSING}"),
            ({"A": "1", "B": "2"}, "a=${A} b=${B}", "a=1 b=2"),
        ],
        ids=["simple", "with_default_set", "with_default_unset", "no_default", "multiple"],
    )
    def test_resolve_env(self, monkeypatch, env_val, template, expected):
        for k, v in env_val.items():
            monkeypatch.setenv(k, v)
        # Clear any leftover env vars from other tests
        for var in ("FOO", "A", "B", "MISSING"):
            if var not in env_val:
                monkeypatch.delenv(var, raising=False)
        assert _resolve_env(template) == expected

    @pytest.mark.unit
    def test_plain_string_passthrough(self):
        assert _resolve_env("plain string") == "plain string"

    @pytest.mark.unit
    def test_empty_string(self):
        assert _resolve_env("") == ""


# ─── Deep Resolution ────────────────────────────────────────────────────────


class TestDeepResolution:
    """Test recursive env var resolution in nested structures."""

    @pytest.mark.unit
    def test_nested_dict(self, monkeypatch):
        monkeypatch.delenv("ENV", raising=False)
        data = {"a": {"b": "${ENV:42}"}}
        result = _resolve_deep(data)
        assert result["a"]["b"] == "42"

    @pytest.mark.unit
    def test_list(self, monkeypatch):
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        data = ["${A:x}", "${B:y}"]
        result = _resolve_deep(data)
        assert result == ["x", "y"]

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [123, True, None, 3.14, []])
    def test_non_string_passthrough(self, value):
        assert _resolve_deep(value) == value


# ─── Config Singleton ───────────────────────────────────────────────────────


class TestConfigSingleton:

    def setup_method(self):
        Config._data = None

    def teardown_method(self):
        Config._data = None

    @pytest.mark.unit
    def test_project_root(self):
        config = Config()
        assert config.project_root.exists()
        assert isinstance(config.project_root, Path)
        assert (config.project_root / "config").exists()

    @pytest.mark.unit
    def test_singleton(self):
        assert Config() is Config()

    @pytest.mark.unit
    def test_get_nested(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("services:\n  comfyui:\n    port: 18465\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        assert Config().get("services.comfyui.port") == 18465

    @pytest.mark.unit
    def test_get_missing_returns_default(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("k: v\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        assert Config().get("nonexistent.key", "fallback") == "fallback"

    @pytest.mark.unit
    def test_require_raises_on_missing(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("k: v\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        with pytest.raises(KeyError):
            Config().require("nonexistent.key")

    @pytest.mark.unit
    def test_models_root_default(self, tmp_path):
        fake = tmp_path / "fake.yaml"
        fake.write_text("models_root: /custom/models\n")
        Config().reload()
        Config._instance._data = {"models_root": "/custom/models"}
        assert Config().models_root == "/custom/models"
        Config().reload()

    @pytest.mark.unit
    def test_env_var_overrides_config(self, monkeypatch, tmp_path):
        """TECH_NOIR_MODELS_ROOT env var should override config file."""
        monkeypatch.setenv("TECH_NOIR_MODELS_ROOT", "/env/models")
        Config().reload()
        assert Config().models_root == "/env/models"
        monkeypatch.delenv("TECH_NOIR_MODELS_ROOT")
        Config().reload()
