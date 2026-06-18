"""Cross-module consistency tests — verify registries and configs agree.

Note: wan2gp service has been removed. These tests validate the remaining
forge-based service tier. The 4-tier pool system is configured declaratively
in config/inference_pools.yaml (no Python handler discovery needed).
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


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
    def test_all_gpu_services_routed_via_forge(self):
        """Every GPU service in the registry must be routeable via forge."""
        from services.registry import SERVICE_REGISTRY
        gpu_services = {name for name, e in SERVICE_REGISTRY.items() if e.needs_gpu}
        from services.forge import SERVICE_MAP
        forge_services = set(SERVICE_MAP.keys())
        unrouted = gpu_services - forge_services
        assert unrouted == set(), (
            f"GPU services not routed via forge: {unrouted}"
        )


# ─── SERVICE_REGISTRY completeness ─────────────────────────────────────────


class TestServiceRegistryCompleteness:

    @pytest.mark.unit
    def test_all_registry_entries_have_deployment_target(self):
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment in ("forge",), (
                f"{name}: unknown deployment '{entry.deployment}'"
            )


# ─── Tier classification ────────────────────────────────────────────────────


class TestTierClassification:

    @pytest.mark.unit
    def test_tier1_services_have_complete_entries(self):
        """Tier 1 services must have all required ServiceEntry fields."""
        from services.registry import SERVICE_REGISTRY

        tier1 = {
            "kokoro", "espeak", "faster_whisper",
            "index_tts", "vibevoice_asr",
            "ace_step", "comfyui",
            "moss_soundeffect", "llm",
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
