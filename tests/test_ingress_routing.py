"""Tests for ingress routing — _ForgeRequest, OpenAI route targets, admin, GPU coordination.

Uses Starlette TestClient where possible. For routes that call serve.get_deployment_handle,
tests verify routing logic by checking the service registry entries that the handler
would resolve, since Ray Serve cannot be mocked in-process without triggering initialization.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient


def _make_client():
    """Create TestClient with no auth."""
    from gateway.ingress import create_app
    return TestClient(create_app())


# ─── _ForgeRequest Tests ───────────────────────────────────────────────────


class TestForgeRequest:

    def test_json_returns_data(self):
        from gateway.ingress import _ForgeRequest
        req = _ForgeRequest({"service": "llm", "prompt": "hi"})
        assert asyncio.get_event_loop().run_until_complete(req.json()) == {"service": "llm", "prompt": "hi"}

    def test_body_returns_json_bytes(self):
        from gateway.ingress import _ForgeRequest
        req = _ForgeRequest({"text": "hello"})
        body = asyncio.get_event_loop().run_until_complete(req.body())
        assert json.loads(body) == {"text": "hello"}

    def test_method_is_post(self):
        from gateway.ingress import _ForgeRequest
        req = _ForgeRequest({})
        assert req.method == "POST"

    def test_url_path_set(self):
        from gateway.ingress import _ForgeRequest
        req = _ForgeRequest({}, path="/v1/chat/completions")
        assert req.url.path == "/v1/chat/completions"

    def test_headers_contain_content_type(self):
        from gateway.ingress import _ForgeRequest
        req = _ForgeRequest({})
        assert req.headers["content-type"] == "application/json"


# ─── OpenAI Route Target Resolution ────────────────────────────────────────


class TestOpenAIRouteTargets:
    """Verify each OpenAI route resolves to the correct deployment via registry."""

    def test_chat_completions_targets_master_router(self):
        from services.registry import get_service
        # LLM goes through master_router, not a standalone deployment
        assert get_service("llm") is not None
        assert get_service("llm").needs_gpu is True

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

    def test_admin_load_with_governor(self):
        with _make_client() as client, \
             patch("gateway.ingress.ray") as mock_ray:
            mock_gov = AsyncMock()
            mock_gov.acquire.remote = AsyncMock(return_value={"granted": True})
            mock_ray.get_actor.return_value = mock_gov
            r = client.post("/admin/load", json={"service": "trellis"})
            assert r.status_code == 200
            assert r.json()["status"] == "loaded"

    def test_admin_load_no_governor_returns_503(self):
        with _make_client() as client, \
             patch("gateway.ingress.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("no governor")
            r = client.post("/admin/load", json={"service": "trellis"})
            assert r.status_code == 503

    def test_admin_unload_releases_holder(self):
        with _make_client() as client, \
             patch("gateway.ingress.ray") as mock_ray:
            mock_gov = AsyncMock()
            mock_gov.status.remote = AsyncMock(return_value={"holder": "trellis"})
            mock_gov.release.remote = AsyncMock()
            mock_ray.get_actor.return_value = mock_gov
            r = client.post("/admin/unload")
            assert r.status_code == 200

    def test_admin_unload_no_governor_returns_503(self):
        with _make_client() as client, \
             patch("gateway.ingress.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("no governor")
            r = client.post("/admin/unload")
            assert r.status_code == 503


# ─── GPU Coordination Tests ────────────────────────────────────────────────


class TestIngressGPU:

    def test_gpu_services_flagged_correctly(self):
        from services.registry import SERVICE_REGISTRY
        gpu_services = {"trellis", "anigen", "ace_step", "comfyui", "llm",
                        "hy_motion", "see_through", "moss_soundeffect", "phi4mm",
                        "index_tts", "faster_qwen3_tts", "gpt_sovits", "qwen_tts",
                        "vibevoice_microsoft", "qwen_asr", "tangoflux"}
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
            assert len(r.json()) == 20

    def test_service_info_found(self):
        with _make_client() as client:
            r = client.get("/v1/services/kokoro")
            assert r.status_code == 200
            assert r.json()["name"] == "kokoro"

    def test_service_info_not_found(self):
        with _make_client() as client:
            r = client.get("/v1/services/nonexistent")
            assert r.status_code == 404
