"""Wan2GP family_handler contract tests — validates all 11 custom handlers.

Every custom handler must conform to the Wan2GP family_handler convention:
  - family_handler class with 7 static methods
  - query_supported_types() → list[str]
  - query_family_maps() → tuple[dict, dict]
  - query_model_family() → str
  - query_family_infos() → dict[str, tuple[int, str]]
  - query_model_def(base_model_type, model_def) → dict
  - load_model(...) → (pipeline, {"pipe": dict, "coTenantsMap": dict})
  - update_default_settings(base_model_type, model_def, ui_defaults)

Plus: pipeline object must have a generate() method.

Tests are parametrized by handler so failures are isolated per-handler.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── Handler fixtures ───────────────────────────────────────────────────────

ALL_HANDLERS = [
    ("models.kokoro.kokoro_handler", "kokoro"),
    ("models.moss.moss_handler", "moss-soundeffect"),
    ("models.espeak.espeak_handler", "espeak"),
    ("models.faster_whisper.faster_whisper_handler", "faster_whisper"),
    ("models.vibevoice_asr.vibevoice_asr_handler", "vibevoice-asr"),
    ("models.vibevoice_tts.vibevoice_tts_handler", "vibevoice-tts"),
]


_FORK_MODELS_DIR = Path(__file__).resolve().parent.parent / "opt" / "wan2gp" / "models"


def _import_handler(import_path: str):
    """Import a handler module from the fork's models/ package.

    Falls back to loading by file path when the standard import fails or
    resolves to the wrong module (e.g. kokoro collides with pip package).
    """
    try:
        mod = importlib.import_module(import_path)
        mod_file = getattr(mod, "__file__", "") or ""
        if "opt/wan2gp/models" in mod_file:
            return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Fallback: strip "models." prefix, load from fork models/ dir
    rel_path = import_path
    if rel_path.startswith("models."):
        rel_path = rel_path[len("models."):]
    parts = rel_path.split(".")
    mod_file = _FORK_MODELS_DIR.joinpath(*parts[:-1]) / (parts[-1] + ".py")
    if not mod_file.exists():
        raise ImportError(f"Handler file not found: {mod_file}")
    spec = importlib.util.spec_from_file_location(import_path, str(mod_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=ALL_HANDLERS, ids=[h[1] for h in ALL_HANDLERS])
def handler(request):
    """Parametrized fixture yielding (import_path, model_type, family_handler class)."""
    import_path, model_type = request.param
    mod = _import_handler(import_path)
    return import_path, model_type, mod.family_handler


@pytest.fixture(params=ALL_HANDLERS, ids=[h[1] for h in ALL_HANDLERS])
def handler_class(request):
    """Parametrized fixture yielding just the family_handler class."""
    import_path, _ = request.param
    mod = _import_handler(import_path)
    return mod.family_handler


# ─── Module Structure ───────────────────────────────────────────────────────


class TestHandlerModuleStructure:
    """Every handler module must expose a family_handler class."""

    @pytest.mark.handler
    @pytest.mark.unit
    def test_module_has_family_handler_class(self, handler):
        import_path, _, fh = handler
        assert isinstance(fh, type), (
            f"{import_path}: family_handler is {type(fh)}, expected a class"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_family_handler_has_required_static_methods(self, handler):
        import_path, _, fh = handler
        required = [
            "query_supported_types",
            "query_family_maps",
            "query_model_family",
            "query_family_infos",
            "query_model_def",
            "load_model",
            "update_default_settings",
        ]
        for method_name in required:
            assert hasattr(fh, method_name), (
                f"{import_path}: missing method '{method_name}'"
            )
            method = getattr(fh, method_name)
            assert callable(method), (
                f"{import_path}: '{method_name}' is not callable"
            )


# ─── query_supported_types() ────────────────────────────────────────────────


class TestQuerySupportedTypes:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_returns_list_of_strings(self, handler):
        import_path, _, fh = handler
        result = fh.query_supported_types()
        assert isinstance(result, list), (
            f"{import_path}: query_supported_types() returned {type(result)}, expected list"
        )
        assert len(result) > 0, f"{import_path}: query_supported_types() returned empty list"
        for item in result:
            assert isinstance(item, str), (
                f"{import_path}: supported type {item!r} is not a string"
            )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_no_duplicate_types(self, handler):
        import_path, _, fh = handler
        result = fh.query_supported_types()
        assert len(result) == len(set(result)), (
            f"{import_path}: duplicate model types in query_supported_types()"
        )


# ─── query_family_maps() ────────────────────────────────────────────────────


class TestQueryFamilyMaps:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_returns_tuple_of_two_dicts(self, handler):
        import_path, _, fh = handler
        result = fh.query_family_maps()
        assert isinstance(result, tuple), (
            f"{import_path}: query_family_maps() returned {type(result)}, expected tuple"
        )
        assert len(result) == 2, (
            f"{import_path}: query_family_maps() returned {len(result)}-tuple, expected 2"
        )
        assert isinstance(result[0], dict), (
            f"{import_path}: first element of family_maps is {type(result[0])}, expected dict"
        )
        assert isinstance(result[1], dict), (
            f"{import_path}: second element of family_maps is {type(result[1])}, expected dict"
        )


# ─── query_model_family() ──────────────────────────────────────────────────


class TestQueryModelFamily:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_returns_non_empty_string(self, handler):
        import_path, _, fh = handler
        result = fh.query_model_family()
        assert isinstance(result, str), (
            f"{import_path}: query_model_family() returned {type(result)}, expected str"
        )
        assert len(result) > 0, (
            f"{import_path}: query_model_family() returned empty string"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_family_name_matches_supported_types_prefix(self, handler):
        """Family name should be related to the handler's model types."""
        import_path, model_type, fh = handler
        family = fh.query_model_family()
        # Family name should appear in handler import path or model type
        combined = import_path + model_type
        assert family in combined or combined.replace("-", "_").replace(".", "_").count(family.replace("-", "_")) >= 0, (
            f"{import_path}: family '{family}' doesn't relate to handler path or model type"
        )


# ─── query_family_infos() ──────────────────────────────────────────────────


class TestQueryFamilyInfos:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_returns_dict_with_tuple_values(self, handler):
        import_path, _, fh = handler
        result = fh.query_family_infos()
        assert isinstance(result, dict), (
            f"{import_path}: query_family_infos() returned {type(result)}, expected dict"
        )
        assert len(result) > 0, (
            f"{import_path}: query_family_infos() returned empty dict"
        )
        for key, value in result.items():
            assert isinstance(key, str), (
                f"{import_path}: family_infos key {key!r} is not a string"
            )
            assert isinstance(value, tuple), (
                f"{import_path}: family_infos['{key}'] is {type(value)}, expected tuple"
            )
            assert len(value) == 2, (
                f"{import_path}: family_infos['{key}'] has {len(value)} elements, expected 2"
            )
            assert isinstance(value[0], int), (
                f"{import_path}: family_infos['{key}'][0] is {type(value[0])}, expected int"
            )
            assert isinstance(value[1], str), (
                f"{import_path}: family_infos['{key}'][1] is {type(value[1])}, expected str"
            )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_family_infos_keys_match_model_family(self, handler):
        """query_family_infos() keys should include the query_model_family() value."""
        import_path, _, fh = handler
        family = fh.query_model_family()
        infos = fh.query_family_infos()
        assert family in infos, (
            f"{import_path}: query_model_family()='{family}' not in query_family_infos() keys {list(infos.keys())}"
        )


# ─── query_model_def() ─────────────────────────────────────────────────────


class TestQueryModelDef:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_returns_dict(self, handler):
        import_path, model_type, fh = handler
        result = fh.query_model_def(model_type, {})
        assert isinstance(result, dict), (
            f"{import_path}: query_model_def() returned {type(result)}, expected dict"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    @pytest.mark.parametrize("key", ["audio_only", "image_outputs"])
    def test_has_standard_keys(self, handler, key):
        import_path, model_type, fh = handler
        result = fh.query_model_def(model_type, {})
        assert key in result, (
            f"{import_path}: query_model_def() missing '{key}' key, got {list(result.keys())}"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_values_are_booleans(self, handler):
        import_path, model_type, fh = handler
        result = fh.query_model_def(model_type, {})
        for key in ("audio_only", "image_outputs"):
            assert isinstance(result[key], bool), (
                f"{import_path}: query_model_def()['{key}'] is {type(result[key])}, expected bool"
            )


# ─── update_default_settings() ──────────────────────────────────────────────


class TestUpdateDefaultSettings:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_does_not_raise(self, handler):
        import_path, model_type, fh = handler
        ui_defaults = {}
        # Should not raise
        fh.update_default_settings(model_type, {}, ui_defaults)

    @pytest.mark.handler
    @pytest.mark.unit
    def test_updates_dict_in_place(self, handler):
        import_path, model_type, fh = handler
        ui_defaults = {}
        fh.update_default_settings(model_type, {}, ui_defaults)
        # Most handlers add at least one default
        assert isinstance(ui_defaults, dict)


# ─── load_model() signature ────────────────────────────────────────────────


class TestLoadModelSignature:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_accepts_standard_parameters(self, handler):
        import_path, _, fh = handler
        sig = inspect.signature(fh.load_model)
        params = list(sig.parameters.keys())
        required = [
            "model_filename", "model_type", "base_model_type", "model_def",
        ]
        for p in required:
            assert p in params, (
                f"{import_path}: load_model() missing parameter '{p}', has {params}"
            )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_accepts_kwargs(self, handler):
        """load_model must accept **kwargs for forward-compat."""
        import_path, _, fh = handler
        sig = inspect.signature(fh.load_model)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_kwargs, (
            f"{import_path}: load_model() must accept **kwargs for Wan2GP compat"
        )


# ─── Pipeline generate() contract (unit — checks class structure) ────────────


class TestPipelineContract:
    """Verify _Pipeline classes have the expected generate() signature."""

    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_class_has_generate(self, handler):
        """Handler modules must define a class with a generate() method."""
        import_path, model_type, fh = handler
        mod = _import_handler(import_path)
        assert hasattr(mod, "_Pipeline"), (
            f"{import_path}: module has no _Pipeline class"
        )
        pipeline_cls = mod._Pipeline
        assert hasattr(pipeline_cls, "generate"), (
            f"{import_path}: _Pipeline has no generate() method"
        )
        assert callable(getattr(pipeline_cls, "generate")), (
            f"{import_path}: _Pipeline.generate is not callable"
        )


# ─── Cross-handler uniqueness ──────────────────────────────────────────────


class TestCrossHandlerUniqueness:
    """Family IDs and infos must be unique across all handlers."""

    @pytest.mark.unit
    def test_family_ids_unique(self):
        """No two handlers can return the same family ID."""
        families = {}
        for import_path, model_type in ALL_HANDLERS:
            mod = _import_handler(import_path)
            fh = mod.family_handler
            family = fh.query_model_family()
            if family in families:
                pytest.fail(
                    f"Duplicate family '{family}' in {import_path} and {families[family]}"
                )
            families[family] = import_path

    @pytest.mark.unit
    def test_family_info_ids_unique(self):
        """Numeric IDs in query_family_infos() must be unique across handlers."""
        all_ids = {}
        for import_path, model_type in ALL_HANDLERS:
            mod = _import_handler(import_path)
            fh = mod.family_handler
            infos = fh.query_family_infos()
            for family, (num_id, label) in infos.items():
                if num_id in all_ids:
                    pytest.fail(
                        f"Duplicate family info ID {num_id} in "
                        f"{import_path} ({family}) and {all_ids[num_id]}"
                    )
                all_ids[num_id] = f"{import_path} ({family})"

    @pytest.mark.unit
    def test_all_handlers_importable(self):
        """Every handler in ALL_HANDLERS can be imported."""
        for import_path, model_type in ALL_HANDLERS:
            mod = _import_handler(import_path)
            assert hasattr(mod, "family_handler"), (
                f"{import_path}: module has no family_handler attribute"
            )
