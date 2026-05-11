"""Unit tests for Config — env var resolution, relative path handling."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

from registry.config import Config, _resolve_env, _resolve_deep, _PROJECT_ROOT


class TestEnvVarResolution:
    """Test ${VAR:default} token resolution."""

    def test_simple_var(self):
        with mock.patch.dict(os.environ, {"FOO": "bar"}):
            assert _resolve_env("${FOO}") == "bar"

    def test_var_with_default_when_set(self):
        with mock.patch.dict(os.environ, {"FOO": "bar"}):
            assert _resolve_env("${FOO:baz}") == "bar"

    def test_var_with_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _resolve_env("${MISSING:default_val}") == "default_val"

    def test_var_without_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _resolve_env("${MISSING}") == "${MISSING}"

    def test_multiple_vars_in_string(self):
        with mock.patch.dict(os.environ, {"A": "1", "B": "2"}):
            assert _resolve_env("a=${A} b=${B}") == "a=1 b=2"

    def test_no_vars(self):
        assert _resolve_env("plain string") == "plain string"


class TestDeepResolution:
    """Test recursive env var resolution in nested structures."""

    def test_nested_dict(self):
        data = {"a": {"b": "${ENV:42}"}}
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _resolve_deep(data)
            assert result["a"]["b"] == "42"

    def test_list(self):
        data = ["${A:x}", "${B:y}"]
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _resolve_deep(data)
            assert result == ["x", "y"]

    def test_non_string_passthrough(self):
        assert _resolve_deep(123) == 123
        assert _resolve_deep(True) is True
        assert _resolve_deep(None) is None


class TestConfigSingleton:
    """Test Config singleton behavior and path access."""

    def setup_method(self):
        Config._data = None

    def teardown_method(self):
        Config._data = None

    def test_project_root(self):
        config = Config()
        assert config.project_root.exists()
        assert isinstance(config.project_root, Path)
        assert (config.project_root / "config").exists()

    def test_singleton(self):
        assert Config() is Config()

    def test_get_nested(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("services:\n  comfyui:\n    port: 18465\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        assert Config().get("services.comfyui.port") == 18465

    def test_get_missing_returns_default(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("k: v\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        assert Config().get("nonexistent.key", "fallback") == "fallback"

    def test_require_raises_on_missing(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("k: v\n")
        monkeypatch.setattr(Config, "_pick_path", lambda s: yaml_file)
        with pytest.raises(KeyError):
            Config().require("nonexistent.key")

    def test_models_root_default(self, tmp_path):
        fake = tmp_path / "fake.yaml"
        fake.write_text("models_root: /custom/models\n")
        Config().reload()
        Config._instance._data = {"models_root": "/custom/models"}
        assert Config().models_root == "/custom/models"
        Config().reload()
