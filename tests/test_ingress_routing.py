"""Tests for ingress routing — OpenAI route targets, admin, GPU coordination.

Uses Starlette TestClient where possible. For routes that call serve.get_deployment_handle,
tests verify routing logic by checking the service registry entries that the handler
would resolve.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient


def _make_client():
    """Create TestClient with no auth."""
    from gateway.ingress import create_app
    return TestClient(create_app())


# ─── OpenAI Route Target Resolution ────────────────────────────────────────


class TestOpenAIRouteTargets:
    """Verify each OpenAI route resolves to the correct deployment via registry."""

    def test_chat_completions_targets_forge(self):
        from services.registry import get_service
        from services.forge import SERVICE_MAP
        # LLM goes through the Forge
        assert get_service("llm") is not None
        assert get_service("llm").needs_gpu is True
        assert "llm" in SERVICE_MAP

    def test_audio_speech_default_model_resolves_kokoro(self):
        from services.registry import resolve_model, get_service
        key, entry = resolve_model("tts-01-kokoro")
        assert key == "kokoro"
        assert entry.deployment == "kokoro_tts"

    def test_audio_speech_espeak_alias(self):
        from services.registry import resolve_model
        key, entry = resolve_model("tts-01-espeak")
        assert key == "espeak"
        assert entry.deployment == "espeak_tts"

    def test_audio_speech_unknown_returns_none(self):
        from services.registry import resolve_model
        assert resolve_model("nonexistent-model") is None

    def test_audio_transcriptions_whisper_alias(self):
        from services.registry import resolve_model
        key, entry = resolve_model("whisper-1")
        assert key == "faster_whisper"
        assert entry.deployment == "faster_whisper"

    def test_all_tts_services_have_speech_aliases(self):
        from services.registry import SERVICE_REGISTRY
        for name in ["kokoro", "espeak", "index_tts", "vibevoice_cpp_gpu"]:
            entry = SERVICE_REGISTRY[name]
            assert entry.model_aliases, f"{name} missing model_aliases for /v1/audio/speech"


# ─── Admin Route Tests ─────────────────────────────────────────────────────


class TestIngressAdminRoutes:

    def test_admin_load_with_forge(self):
        with _make_client() as client, \
             patch("gateway.ingress._get_forge") as mock_forge:
            mock_handle = AsyncMock()
            mock_handle.preload.remote = AsyncMock(
                return_value={"status": "loaded", "service": "trellis"})
            mock_forge.return_value = mock_handle
            r = client.post("/admin/load", json={"service": "trellis"})
            assert r.status_code == 200
            assert r.json()["status"] == "loaded"

    def test_admin_load_no_forge_returns_503(self):
        with _make_client() as client, \
             patch("gateway.ingress._get_forge") as mock_forge:
            mock_forge.side_effect = Exception("no cluster")
            r = client.post("/admin/load", json={"service": "trellis"})
            assert r.status_code == 503

    def test_admin_unload_with_forge(self):
        with _make_client() as client, \
             patch("gateway.ingress._get_forge") as mock_forge:
            mock_handle = AsyncMock()
            mock_handle.release.remote = AsyncMock(
                return_value={"status": "released"})
            mock_forge.return_value = mock_handle
            r = client.post("/admin/unload")
            assert r.status_code == 200

    def test_admin_unload_no_forge_returns_503(self):
        with _make_client() as client, \
             patch("gateway.ingress._get_forge") as mock_forge:
            mock_forge.side_effect = Exception("no cluster")
            r = client.post("/admin/unload")
            assert r.status_code == 503


# ─── GPU Coordination Tests ────────────────────────────────────────────────


class TestIngressGPU:

    def test_gpu_services_flagged_correctly(self):
        from services.registry import SERVICE_REGISTRY
        gpu_services = {"trellis", "anigen", "ace_step", "comfyui", "llm",
                        "hy_motion", "see_through", "moss_soundeffect", "phi4mm",
                        "index_tts", "faster_qwen3_tts", "gpt_sovits",
                        "vibevoice_microsoft"}
        for name in gpu_services:
            assert SERVICE_REGISTRY[name].needs_gpu is True, f"{name} should need GPU"

    def test_cpu_services_skip_gpu(self):
        from services.registry import SERVICE_REGISTRY
        for name in ["kokoro", "espeak", "faster_whisper"]:
            assert SERVICE_REGISTRY[name].needs_gpu is False, f"{name} should be CPU"


# ─── Service Discovery Tests ──────────────────────────────────────────────


class TestIngressServiceDiscovery:

    def test_list_models_returns_model_list(self):
        with _make_client() as client:
            r = client.get("/v1/models")
            assert r.status_code == 200
            data = r.json()
            assert data["object"] == "list"
            assert isinstance(data["data"], list)

    def test_list_services(self):
        with _make_client() as client:
            r = client.get("/v1/services")
            assert r.status_code == 200
            assert len(r.json()) > 0

    def test_service_info_found(self):
        with _make_client() as client:
            r = client.get("/v1/services/kokoro")
            assert r.status_code == 200
            assert r.json()["name"] == "kokoro"

    def test_service_info_not_found(self):
        with _make_client() as client:
            r = client.get("/v1/services/nonexistent")
            assert r.status_code == 404
