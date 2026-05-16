"""Module quant variant resolver.

Reads ``config/model_specs.yaml`` and resolves per-module weight paths
for a requested quant level. Used by Wan2GP handlers and the Forge.

Usage::

    from registry.specs import resolve

    spec = resolve("moss", quant="bf16")
    # {"modules": {"language_model": "/abs/path/.../bf16/", ...},
    #  "co_tenants": {"language_model": ["emb_ext"]},
    #  "quant": "bf16"}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_SPECS_PATH = Path(__file__).resolve().parent.parent / "config" / "model_specs.yaml"
_cache: dict[str, Any] | None = None


def _load_specs() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_SPECS_PATH) as f:
            _cache = yaml.safe_load(f) or {}
    return _cache


def reload() -> None:
    """Drop cached specs so the next call re-reads the file."""
    global _cache
    _cache = None


def _models_root() -> str:
    from registry.config import Config
    return Config().models_root


def resolve(
    model_name: str,
    quant: str | None = None,
    models_root: str | None = None,
) -> dict:
    """Resolve module paths for a model at a given quant level.

    Returns::

        {
            "modules": {name: absolute_path, ...},
            "bundled": {name: parent_module_name, ...},
            "co_tenants": {name: [tenants], ...},
            "quant": "bf16",
        }

    Resolution logic:
      1. If quant specified, use it.
      2. Else use ``quant_default`` from spec.
      3. For each module with variants, pick the requested quant.
         If the requested quant isn't available, pick the first available.
      4. Bundled modules get the same path as their parent (the first
         non-bundled module found before them that shares weights).
      5. Paths are resolved relative to MODELS_ROOT.
    """
    specs = _load_specs()

    if model_name not in specs:
        raise ValueError(
            f"No spec for model '{model_name}'. "
            f"Available: {sorted(specs.keys())}"
        )

    spec = specs[model_name]
    resolved_quant = quant or spec.get("quant_default", "bf16")
    root = models_root or _models_root()

    modules: dict[str, str] = {}
    bundled: dict[str, str] = {}
    co_tenants = spec.get("co_tenants", {})

    # Find the "primary" module for bundled resolution — first non-bundled module
    primary_module: str | None = None

    for mod_name, mod_spec in spec.get("modules", {}).items():
        if mod_spec.get("bundled"):
            # Bundled modules share the primary module's path
            if primary_module:
                bundled[mod_name] = primary_module
                modules[mod_name] = modules[primary_module]
            continue

        primary_module = primary_module or mod_name

        variants = mod_spec.get("variants", {})
        if not variants:
            continue

        # Pick the requested quant, or fall back to first available
        if resolved_quant in variants:
            rel_path = variants[resolved_quant]
        else:
            # Fall back to the first variant listed
            first_key = next(iter(variants))
            logger.warning(
                "Quant '%s' not available for %s.%s, falling back to '%s'",
                resolved_quant, model_name, mod_name, first_key,
            )
            rel_path = variants[first_key]

        modules[mod_name] = str(Path(root) / rel_path)

    return {
        "modules": modules,
        "bundled": bundled,
        "co_tenants": co_tenants,
        "quant": resolved_quant,
    }


def list_models() -> list[str]:
    """Return all model names with specs."""
    return sorted(_load_specs().keys())


def model_info(model_name: str) -> dict:
    """Return raw spec for a model (for inspection/debugging)."""
    specs = _load_specs()
    if model_name not in specs:
        raise ValueError(f"No spec for model '{model_name}'")
    return specs[model_name]
