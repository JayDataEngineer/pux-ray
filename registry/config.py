"""Configuration singleton - reads config/local.yaml with env-var overrides.

Resolves ``${ENV_VAR:default}`` syntax in YAML values so that
machine-specific paths can live in config/local.yaml (git-ignored)
while config/local.yaml.example ships safe defaults.

Usage:
    from registry.config import Config

    path = Config().get("binaries.llama_server")
    root = Config().models_root
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_YAML = _PROJECT_ROOT / "config" / "local.yaml"
_EXAMPLE_YAML = _PROJECT_ROOT / "config" / "local.yaml.example"

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(value: str) -> str:
    """Replace ``${VAR:default}`` and ``${VAR}`` tokens with env values."""

    def _sub(m: re.Match) -> str:
        token = m.group(1)
        if ":" in token:
            var, default = token.split(":", 1)
            return os.environ.get(var.strip(), default)
        return os.environ.get(token.strip(), m.group(0))

    return _ENV_VAR_RE.sub(_sub, value)


def _resolve_deep(obj: Any) -> Any:
    """Recursively resolve env-var placeholders in nested structures."""
    if isinstance(obj, str):
        return _resolve_env(obj)
    if isinstance(obj, dict):
        return {k: _resolve_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_deep(v) for v in obj]
    return obj


class Config:
    """Cached configuration singleton.

    Reads *config/local.yaml* when present; otherwise falls back to
    *config/local.yaml.example*.  All ``${ENV:default}`` tokens are
    resolved at load time against the real environment.
    """

    _instance: Config | None = None
    _data: dict[str, Any] | None = None

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # -- loading ----------------------------------------------------------

    @staticmethod
    def _pick_path() -> Path:
        if _LOCAL_YAML.exists():
            return _LOCAL_YAML
        if _EXAMPLE_YAML.exists():
            return _EXAMPLE_YAML
        raise FileNotFoundError(
            f"No config found. Expected one of:\n"
            f"  {_LOCAL_YAML}\n  {_EXAMPLE_YAML}"
        )

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            path = self._pick_path()
            logger.debug("Loading config from %s", path)
            with open(path) as fh:
                raw = yaml.safe_load(fh)
            self._data = _resolve_deep(raw) if raw else {}
        return self._data

    def reload(self) -> None:
        """Drop cached data so the next access re-reads the file."""
        self._data = None

    # -- dict-like access -------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a nested value by dot-separated key.

        Examples::

            Config().get("binaries.llama_server")
            Config().get("services.comfyui.port", 8188)
        """
        node: Any = self.data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, dotted_key: str) -> Any:
        """Like :meth:`get` but raises ``KeyError`` when missing."""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Required config key missing: {dotted_key}")
        return value

    # -- convenience properties -------------------------------------------

    @property
    def models_root(self) -> str:
        """Root directory for all model files.

        Must be set via ``TECH_NOIR_MODELS_ROOT`` env var or
        ``models_root`` in ``config/local.yaml``.
        Falls back to ``<project_root>/models`` if unconfigured.
        """
        raw = self.get("models_root", "")
        if not raw or raw.startswith("${"):
            default = _PROJECT_ROOT / "models"
            default.mkdir(exist_ok=True)
            return str(default)
        return raw

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT
