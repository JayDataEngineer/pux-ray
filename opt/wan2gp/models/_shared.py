"""Shared utilities for Wan2GP family handlers.

Spec-first path resolution, weight loading helpers, optional base class,
and vendor registration used across all custom handlers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_model_path(
    model_name: str,
    model_def_key: str,
    model_def: dict[str, Any] | None,
    *,
    category: str | None = None,
    registry_name: str | None = None,
    quant: str | None = None,
    spec_module: str = "model",
    check_file: str | None = None,
) -> Path:
    """Resolve model path using spec-first strategy.

    Resolution order:
      1. model_def[model_def_key] if it points to a valid directory
      2. registry.specs.resolve(model_name, quant) → spec[modules][spec_module]
      3. registry.get_path(category, registry_name or model_name)
      4. HF auto-download (caller handles this)

    Args:
        model_name: Name in model_specs.yaml (e.g. "kokoro", "vibevoice_asr").
        model_def_key: Key to check in model_def dict (e.g. "kokoro_path").
        model_def: The model_def dict passed to load_model().
        category: Registry category for get_path fallback (e.g. "3d", "tts").
        registry_name: Override name for get_path (defaults to model_name).
        quant: Quant level to request from spec resolver.
        spec_module: Module key in spec to extract (defaults to first available).
        check_file: If set, verify this file exists in the resolved path.

    Returns:
        Resolved Path, or empty Path if nothing found.
    """
    model_def = model_def or {}

    # 1. Check model_def first (explicit override from deployment.py)
    path = Path(model_def.get(model_def_key, ""))
    if check_file:
        if path.is_dir() and (path / check_file).exists():
            return path
    elif path.is_dir():
        return path

    # 2. Spec-first resolution
    try:
        from registry.specs import resolve
        spec = resolve(model_name, quant=quant)
        modules = spec.get("modules", {})
        if spec_module in modules:
            sp = Path(modules[spec_module])
            if check_file:
                if sp.is_dir() and (sp / check_file).exists():
                    return sp
            elif sp.is_dir():
                return sp
        # Try first non-empty module path
        for mod_path in modules.values():
            p = Path(mod_path)
            if p.is_dir():
                if check_file and not (p / check_file).exists():
                    continue
                return p
    except Exception:
        pass

    # 3. Registry get_path fallback
    if category:
        try:
            from registry.models import ModelRegistry
            reg = ModelRegistry()
            p = Path(reg.get_path(category, registry_name or model_name))
            if check_file:
                if p.is_dir() and (p / check_file).exists():
                    return p
            elif p.is_dir():
                return p
        except Exception:
            pass

    return Path("")



def load_safetensors(model_path: Path, pattern: str = "model*.safetensors") -> dict:
    """Load all safetensors files matching pattern into a flat state dict."""
    import safetensors.torch

    sd: dict = {}
    for sf_path in sorted(model_path.rglob(pattern)):
        sd.update(safetensors.torch.load_file(str(sf_path)))
    return sd


def load_prefix_into_module(
    sd: dict,
    prefix: str,
    module,
    dtype=None,
) -> dict:
    """Load weights matching *prefix* into *module*, return leftover keys.

    Strips the prefix, casts to dtype, loads with strict=False (ignores
    missing/unexpected keys).

    Returns:
        Remaining state dict with prefix-matched keys removed.
    """
    matched: dict = {}
    rest: dict = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            matched[k[len(prefix):].lstrip(".")] = v.to(dtype) if dtype else v
        else:
            rest[k] = v
    module.load_state_dict(matched, strict=False)
    return rest


def load_torch_checkpoint(path: Path, map_location: str = "cpu", weights_only: bool = True):
    """Load a PyTorch checkpoint file (.pt, .pth, .ckpt).

    Prefers weights_only=True for safety. Falls back to weights_only=False
    for legacy checkpoints that store non-tensor objects.
    """
    import torch

    try:
        return torch.load(str(path), map_location=map_location, weights_only=weights_only)
    except Exception:
        return torch.load(str(path), map_location=map_location, weights_only=False)


def register_vendor_package(name: str, path: Path):
    """Register a vendor package in sys.modules via importlib.

    Required when vendor code uses absolute imports (e.g. ``from modules import
    ...``) that would collide with Wan2GP's ``models`` package or other
    top-level names.

    Args:
        name: Module name to register (e.g. "anigen", "modules").
        path: Directory containing the vendor package.
    """
    import importlib.util
    import sys

    if name in sys.modules:
        return sys.modules[name]
    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
    spec = importlib.util.spec_from_file_location(
        name, str(init_file),
        submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BaseFamilyHandler:
    """Optional base class for Wan2GP family_handler.

    Provides sensible defaults for the 7-method contract so handlers only
    specify what's unique. Subclasses must define class attributes and can
    override any method.

    Required class attributes:
        FAMILY: str — family name (e.g. "pixal3d")
        FAMILY_ID: int — unique numeric ID (300-499)
        DISPLAY_NAME: str — human-readable name (e.g. "Pixal3D")
        SUPPORTED_TYPES: list[str] — model types (e.g. ["pixal3d"])
        AUDIO_ONLY: bool — True for TTS/ASR, False for image/3D
        UI_DEFAULTS: dict — default UI settings

    Example::

        class family_handler(BaseFamilyHandler):
            FAMILY = "pixal3d"
            FAMILY_ID = 404
            DISPLAY_NAME = "Pixal3D"
            SUPPORTED_TYPES = ["pixal3d"]
            AUDIO_ONLY = False
            UI_DEFAULTS = {"steps": 12, "guidance": 7.5}

            @staticmethod
            def load_model(model_filename, model_type, base_model_type,
                           model_def, **kwargs):
                ...

    Not mandatory — existing handlers that don't inherit from this will
    continue to work. New handlers should use it to reduce boilerplate.
    """

    FAMILY: str = ""
    FAMILY_ID: int = 0
    DISPLAY_NAME: str = ""
    SUPPORTED_TYPES: list[str] = []
    AUDIO_ONLY: bool = False
    UI_DEFAULTS: dict = {}

    @classmethod
    def query_supported_types(cls) -> list[str]:
        return list(cls.SUPPORTED_TYPES)

    @classmethod
    def query_family_maps(cls) -> tuple[dict, dict]:
        return {}, {}

    @classmethod
    def query_model_family(cls) -> str:
        return cls.FAMILY

    @classmethod
    def query_family_infos(cls) -> dict[str, tuple[int, str]]:
        return {cls.FAMILY: (cls.FAMILY_ID, cls.DISPLAY_NAME)}

    @classmethod
    def query_model_def(cls, base_model_type: str, model_def: dict) -> dict:
        return {
            "audio_only": cls.AUDIO_ONLY,
            "image_outputs": not cls.AUDIO_ONLY,
        }

    @classmethod
    def update_default_settings(cls, base_model_type: str, model_def: dict,
                                ui_defaults: dict) -> None:
        ui_defaults.update(cls.UI_DEFAULTS)

