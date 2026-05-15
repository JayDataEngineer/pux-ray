"""Cross-module consistency tests — verify registries, handlers, and config agree.

These tests catch mismatches between:
- SERVICE_REGISTRY and the handler discovery system
- Forge SERVICE_MAP and SERVICE_REGISTRY
- Custom handler files on disk and CUSTOM_HANDLERS list
- model_registry.yaml and handler _WEIGHT_SEARCH entries
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def custom_models_dir(project_root):
    return project_root / "services" / "wan2gp" / "custom_models"


# ─── Handler files on disk vs CUSTOM_HANDLERS ───────────────────────────────


class TestHandlerFilesMatchRegistry:

    @pytest.mark.unit
    def test_all_custom_handlers_have_files(self, custom_models_dir):
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        for handler_path in CUSTOM_HANDLERS:
            parts = handler_path.split(".")
            mod_file = custom_models_dir.joinpath(*parts[:-1]) / (parts[-1] + ".py")
            assert mod_file.exists(), (
                f"CUSTOM_HANDLERS entry '{handler_path}' has no file at {mod_file}"
            )

    @pytest.mark.unit
    def test_all_handler_files_in_custom_models_registered(self, custom_models_dir):
        """Every *_handler.py in custom_models/ should be in CUSTOM_HANDLERS."""
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        handler_files = set()
        for f in custom_models_dir.rglob("*_handler.py"):
            rel = f.relative_to(custom_models_dir)
            parts = list(rel.parts)
            parts[-1] = parts[-1].removesuffix(".py")
            handler_files.add(".".join(parts))

        unregistered = handler_files - set(CUSTOM_HANDLERS)
        assert unregistered == set(), (
            f"Handler files not in CUSTOM_HANDLERS: {unregistered}"
        )


# ─── SERVICE_REGISTRY vs Forge SERVICE_MAP ──────────────────────────────────


class TestServiceRegistryForgeConsistency:

    @pytest.mark.unit
    def test_all_forge_services_in_registry(self):
        from services.forge import SERVICE_MAP
        from services.registry import SERVICE_REGISTRY
        for service_name in SERVICE_MAP:
            assert service_name in SERVICE_REGISTRY, (
                f"SERVICE_MAP entry '{service_name}' not in SERVICE_REGISTRY"
            )

    @pytest.mark.unit
    def test_forge_services_marked_as_gpu(self):
        from services.forge import SERVICE_MAP
        from services.registry import SERVICE_REGISTRY
        for service_name in SERVICE_MAP:
            entry = SERVICE_REGISTRY[service_name]
            assert entry.needs_gpu is True, (
                f"SERVICE_MAP service '{service_name}' not marked as needs_gpu"
            )

    @pytest.mark.unit
    def test_gpu_registry_services_in_forge_or_wan2gp(self):
        """Every GPU service must be routeable via forge or wan2gp."""
        from services.registry import SERVICE_REGISTRY
        gpu_services = {name for name, e in SERVICE_REGISTRY.items() if e.needs_gpu}
        from services.forge import SERVICE_MAP
        forge_services = set(SERVICE_MAP.keys())
        # wan2gp handles many GPU models through Wan2GPForgeService
        wan2gp_handled = {"trellis", "anigen", "ace_step", "hy_motion",
                          "moss_soundeffect", "see_through", "index_tts",
                          "faster_qwen3_tts", "vibevoice_asr", "vibevoice_tts",
                          "wan2gp"}
        routed = forge_services | wan2gp_handled
        unrouted = gpu_services - routed
        assert unrouted == set(), (
            f"GPU services not routed via forge or wan2gp: {unrouted}"
        )


# ─── SERVICE_REGISTRY completeness ─────────────────────────────────────────


class TestServiceRegistryCompleteness:

    @pytest.mark.unit
    def test_all_custom_handlers_have_registry_entries(self):
        """Every custom handler should map to a SERVICE_REGISTRY entry."""
        from services.wan2gp.deployment import CUSTOM_HANDLERS
        from services.registry import SERVICE_REGISTRY

        # Handler family names → registry service names
        handler_to_service = {
            "trellis.trellis_handler": "trellis",
            "anigen_handler.anigen_handler": "anigen",
            "see_through.see_through_handler": "see_through",
            "hy_motion.hy_motion_handler": "hy_motion",
            "kokoro.kokoro_handler": "kokoro",
            "moss.moss_handler": "moss_soundeffect",
            "espeak.espeak_handler": "espeak",
            "faster_whisper.faster_whisper_handler": "faster_whisper",
            "vibevoice_asr.vibevoice_asr_handler": "vibevoice_asr",
            "vibevoice_tts.vibevoice_tts_handler": "vibevoice_tts",
            "faster_qwen3_tts.faster_qwen3_tts_handler": "faster_qwen3_tts",
        }

        for handler_path, service_name in handler_to_service.items():
            assert handler_path in CUSTOM_HANDLERS, (
                f"Handler {handler_path} not in CUSTOM_HANDLERS"
            )
            assert service_name in SERVICE_REGISTRY, (
                f"Handler {handler_path} → service '{service_name}' not in SERVICE_REGISTRY"
            )

    @pytest.mark.unit
    def test_all_registry_entries_have_deployment_target(self):
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment in ("forge", "wan2gp"), (
                f"{name}: unknown deployment '{entry.deployment}'"
            )


# ─── model_registry.yaml vs handler _WEIGHT_SEARCH ─────────────────────────


class TestModelRegistryWeightSearchConsistency:

    @pytest.mark.unit
    def test_weight_search_categories_in_model_registry(self):
        """All categories referenced in _WEIGHT_SEARCH must exist in model_registry.yaml."""
        from services.wan2gp.deployment import _WEIGHT_SEARCH
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        missing = []
        for model_type, searches in _WEIGHT_SEARCH.items():
            for category, name in searches:
                if category not in registry.data:
                    missing.append(f"{model_type}: category '{category}' not in registry")
                elif name not in registry.data.get(category, {}):
                    missing.append(f"{model_type}: '{category}/{name}' not in registry")
        assert missing == [], f"_WEIGHT_SEARCH references missing entries:\n" + "\n".join(missing)


# ─── Tier classification ────────────────────────────────────────────────────


class TestTierClassification:

    @pytest.mark.unit
    def test_tier1_services_have_complete_entries(self):
        """Tier 1 services must have all required ServiceEntry fields."""
        from services.registry import SERVICE_REGISTRY

        tier1 = {
            "kokoro", "espeak", "faster_whisper", "faster_qwen3_tts",
            "index_tts", "vibevoice_asr", "vibevoice_tts",
            "trellis", "ace_step", "comfyui", "hy_motion",
            "moss_soundeffect", "anigen", "see_through", "llm",
            "wan2gp",
        }
        for name in tier1:
            assert name in SERVICE_REGISTRY, f"Tier 1 service '{name}' not in SERVICE_REGISTRY"
            entry = SERVICE_REGISTRY[name]
            assert entry.deployment, f"{name}: missing deployment"
            assert entry.app, f"{name}: missing app"
            assert entry.label, f"{name}: missing label"
            assert entry.category, f"{name}: missing category"
            assert entry.default_model, f"{name}: missing default_model"
            assert entry.output_type, f"{name}: missing output_type"
