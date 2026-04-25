"""Model registry - single source of truth for model file locations."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional

from registry.config import Config

_REGISTRY_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REGISTRY_DIR / "config" / "model_registry.yaml"
_MODELS_ROOT = Path(Config().models_root)


class ModelRegistry:
    """Reads model_registry.yaml and resolves model paths."""

    _instance: ModelRegistry | None = None
    _data: dict[str, Any] | None = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            with open(_CONFIG_PATH) as f:
                self._data = yaml.safe_load(f)
        return self._data

    def reload(self) -> None:
        self._data = None

    def get_path(self, service_type: str, model_name: str) -> Path:
        """Get absolute path to a model's files."""
        entry = self.data[service_type][model_name]
        raw = entry.get("path") or entry.get("directory")
        p = Path(raw)
        if not p.is_absolute():
            p = _MODELS_ROOT / p
        return p

    def get_metadata(self, service_type: str, model_name: str) -> dict[str, Any]:
        return self.data[service_type][model_name]

    def list_models(self, service_type: Optional[str] = None) -> dict[str, Any]:
        if service_type:
            return {service_type: self.data.get(service_type, {})}
        return self.data

    def validate_paths(self) -> list[str]:
        """Return list of model entries whose paths don't exist on disk."""
        missing = []
        for stype, models in self.data.items():
            if not isinstance(models, dict):
                continue
            for mname, meta in models.items():
                if not isinstance(meta, dict):
                    continue
                try:
                    p = self.get_path(stype, mname)
                    if not p.exists():
                        missing.append(f"{stype}/{mname}: {p}")
                except (KeyError, TypeError):
                    pass
        return missing
