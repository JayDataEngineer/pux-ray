"""Tests for the unified service registry, ingress routing, and SDK client.

All services route through Forge → Wan2GP → model_engine pipeline.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_ray():
    with patch("gateway.ingress._get_forge") as mock_forge:
        mock_handle = AsyncMock()
        mock_handle.status.remote = AsyncMock(return_value={})
        mock_handle.invoke.remote = AsyncMock(return_value={"status": "success"})
        mock_forge.return_value = mock_handle
        yield


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    from gateway.ingress import create_app
    return TestClient(create_app())


# ─── Registry Tests ───────────────────────────────────────────────────────────

class TestServiceRegistry:

    def test_core_services_present(self):
        from services.registry import SERVICE_REGISTRY
        expected = {
            "kokoro", "espeak", "faster_qwen3_tts", "faster_whisper",
            "moss_soundeffect", "ace_step",
            "trellis", "anigen", "hy_motion", "see_through",
            "comfyui", "llm", "wan2gp",
            "index_tts", "vibevoice_asr", "vibevoice_tts",
        }
        assert expected.issubset(set(SERVICE_REGISTRY.keys())), \
            f"Missing: {expected - set(SERVICE_REGISTRY.keys())}"

    def test_all_route_through_forge(self):
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment == "forge", \
                f"{name} deployment is '{entry.deployment}', expected 'forge'"

    def test_get_service_found(self):
        from services.registry import get_service
        entry = get_service("kokoro")
        assert entry is not None
        assert entry.category == "tts"
        assert entry.needs_gpu is False
        assert entry.output_type == "audio"

    def test_get_service_not_found(self):
        from services.registry import get_service
        assert get_service("nonexistent") is None

    def test_resolve_model_aliases(self):
        from services.registry import resolve_model
        key, entry = resolve_model("tts-01-kokoro")
        assert key == "kokoro"

        key, entry = resolve_model("whisper-1")
        assert key == "faster_whisper"

    def test_resolve_model_not_found(self):
        from services.registry import resolve_model
        assert resolve_model("nonexistent-model") is None

    def test_categories_valid(self):
        from services.registry import SERVICE_REGISTRY
        valid = {"tts", "asr", "audio", "creative", "llm", "image"}
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.category in valid, f"{name}: bad category '{entry.category}'"

    def test_output_types_valid(self):
        from services.registry import SERVICE_REGISTRY
        valid = {"audio", "json", "model_3d", "image", "proxy", "video"}
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.output_type in valid, f"{name}: bad output_type '{entry.output_type}'"

    def test_cpu_services_marked_correctly(self):
        from services.registry import SERVICE_REGISTRY
        for name in ("kokoro", "espeak", "faster_whisper"):
            assert SERVICE_REGISTRY[name].needs_gpu is False, f"{name} should be CPU"

    def test_all_entries_have_required_fields(self):
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment, f"{name}: missing deployment"
            assert entry.app, f"{name}: missing app"
            assert entry.label, f"{name}: missing label"
            assert entry.default_model, f"{name}: missing default_model"
            assert entry.description, f"{name}: missing description"

    def test_model_aliases_unique(self):
        from services.registry import SERVICE_REGISTRY
        all_aliases = {}
        for name, entry in SERVICE_REGISTRY.items():
            for alias in entry.model_aliases:
                assert alias not in all_aliases, \
                    f"Duplicate alias '{alias}' in {name} and {all_aliases[alias]}"
                all_aliases[alias] = name


# ─── Ingress Route Tests ──────────────────────────────────────────────────────

class TestIngressRoutes:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_status(self, client):
        r = client.get("/status")
        assert r.status_code == 200

    def test_list_services(self, client):
        from services.registry import SERVICE_REGISTRY
        r = client.get("/v1/services")
        assert r.status_code == 200
        svcs = r.json()
        assert len(svcs) == len(SERVICE_REGISTRY)
        names = {s["name"] for s in svcs}
        assert names == set(SERVICE_REGISTRY.keys())

    def test_service_info_found(self, client):
        r = client.get("/v1/services/kokoro")
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "kokoro"
        assert d["output_type"] == "audio"

    def test_service_info_not_found(self, client):
        r = client.get("/v1/services/nonexistent")
        assert r.status_code == 404

    def test_tnap_generate_unknown_service(self, client):
        r = client.post("/v1/nonexistent/generate", json={"action": "generate"})
        assert r.status_code == 404

    def test_admin_load(self, client):
        with patch("gateway.ingress._get_forge") as mock_forge:
            mock_handle = AsyncMock()
            mock_handle.preload.remote = AsyncMock(
                return_value={"status": "loaded", "service": "trellis"})
            mock_forge.return_value = mock_handle
            r = client.post("/admin/load", json={"service": "trellis", "model": "trellis"})
            assert r.status_code == 200

    def test_admin_unload(self, client):
        with patch("gateway.ingress._get_forge") as mock_forge:
            mock_handle = AsyncMock()
            mock_handle.release.remote = AsyncMock(
                return_value={"status": "released"})
            mock_forge.return_value = mock_handle
            r = client.post("/admin/unload")
            assert r.status_code == 200


# ─── SDK Client Tests ─────────────────────────────────────────────────────────

class TestSDKClient:

    @pytest.fixture
    def client(self):
        from sdk.client import RayClient
        return RayClient(base_url="http://test")

    def test_has_all_methods(self, client):
        for m in ["generate", "generate_binary", "services", "service_info",
                   "chat", "transcribe", "synthesize", "status", "load", "unload"]:
            assert hasattr(client, m), f"Missing: {m}"

    @pytest.mark.asyncio
    async def test_generate_sends_correct_url_and_payload(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "output": {}, "metrics": {}}
        mock_resp.raise_for_status = MagicMock()
        client.client.post = AsyncMock(return_value=mock_resp)

        await client.generate("kokoro", input={"text": "hi"}, config={"precision": "fp16"})

        url = client.client.post.call_args[0][0]
        assert url == "/v1/kokoro/generate"

        payload = client.client.post.call_args[1]["json"]
        assert payload["action"] == "generate"
        assert payload["input"]["text"] == "hi"


# ─── Cross-module Consistency ─────────────────────────────────────────────────

class TestConsistency:

    def test_ingress_matches_registry(self):
        from starlette.testclient import TestClient
        from gateway.ingress import create_app
        from services.registry import SERVICE_REGISTRY

        client = TestClient(create_app())
        api_names = {s["name"] for s in client.get("/v1/services").json()}
        reg_names = set(SERVICE_REGISTRY.keys())
        assert api_names == reg_names
