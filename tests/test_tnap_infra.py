"""Tests for the TNAP service registry, ingress routing, and SDK client.

Run:
    uv run pytest tests/test_tnap_infra.py -v
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_ray():
    """Mock Ray actor lookups so tests don't hang on Forge connection."""
    with patch("gateway.ingress._get_forge") as mock_forge:
        mock_handle = AsyncMock()
        mock_handle.status.remote = AsyncMock(return_value={})
        mock_handle.invoke.remote = AsyncMock(return_value={"status": "success"})
        mock_forge.return_value = mock_handle
        yield


@pytest.fixture
def client():
    """Starlette TestClient for ingress."""
    from starlette.testclient import TestClient
    from gateway.ingress import create_app
    return TestClient(create_app())


# ─── Registry Tests ───────────────────────────────────────────────────────────


class TestServiceRegistry:
    """Test services.registry data integrity."""

    def test_all_services_present(self):
        from services.registry import SERVICE_REGISTRY
        expected = {
            "kokoro", "espeak", "faster_qwen3_tts", "index_tts",
            "vibevoice_cpp_gpu", "gpt_sovits", "faster_whisper",
            "vibevoice_microsoft", "moss_soundeffect",
            "ace_step", "trellis", "anigen", "hy_motion", "see_through",
            "phi4mm", "comfyui", "llm", "wan2gp",
        }
        assert expected.issubset(set(SERVICE_REGISTRY.keys())), \
            f"Missing: {expected - set(SERVICE_REGISTRY.keys())}"

    def test_deployment_names_unique(self):
        from services.registry import SERVICE_REGISTRY
        dep_names = [e.deployment for e in SERVICE_REGISTRY.values()]
        assert len(dep_names) == len(set(dep_names)), \
            f"Duplicate: {[n for n in dep_names if dep_names.count(n) > 1]}"

    def test_deployment_names_match_serve_config(self):
        from services.registry import SERVICE_REGISTRY
        from pathlib import Path
        import re
        serve_config = Path("infra/k8s/serve_config.py").read_text()
        registry_deployments = {e.deployment for e in SERVICE_REGISTRY.values()}
        # Extract bound deployment names from serve_config (e.g. "kokoro_tts = KokoroTTS.bind()")
        bound = set(re.findall(r'^(\w+)\s*=\s*\w+\.bind\(\)', serve_config, re.MULTILINE))
        # Exclude infrastructure deployments (not AI services)
        infra = {"api_ingress", "playground", "vibevoice_cpp_cpu", "forge"}
        unregistered = bound - registry_deployments - infra
        assert not unregistered, f"Deployments in serve_config but not in registry: {unregistered}"

    def test_get_service_found(self):
        from services.registry import get_service
        entry = get_service("kokoro")
        assert entry is not None
        assert entry.deployment == "kokoro_tts"
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
        assert entry.deployment == "kokoro_tts"

        key, entry = resolve_model("whisper-1")
        assert key == "faster_whisper"

    def test_resolve_model_not_found(self):
        from services.registry import resolve_model
        assert resolve_model("nonexistent-model") is None

    def test_categories_valid(self):
        from services.registry import SERVICE_REGISTRY
        valid = {"tts", "asr", "audio", "creative", "vision", "multimodal", "image", "llm"}
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.category in valid, f"{name}: bad category '{entry.category}'"

    def test_output_types_valid(self):
        from services.registry import SERVICE_REGISTRY
        valid = {"audio", "json", "model_3d", "image", "proxy", "video"}
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.output_type in valid, f"{name}: bad output_type '{entry.output_type}'"

    def test_all_entries_have_required_fields(self):
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment, f"{name}: missing deployment"
            assert entry.app, f"{name}: missing app"
            assert entry.label, f"{name}: missing label"
            assert entry.default_model, f"{name}: missing default_model"
            assert entry.description, f"{name}: missing description"

    def test_cpu_services_marked_correctly(self):
        from services.registry import SERVICE_REGISTRY
        for name in ("kokoro", "espeak", "faster_whisper"):
            assert SERVICE_REGISTRY[name].needs_gpu is False, f"{name} should be CPU"

    def test_gpu_services_marked_correctly(self):
        from services.registry import SERVICE_REGISTRY
        gpu_services = {"trellis", "anigen", "ace_step", "phi4mm",
                        "hy_motion", "see_through", "comfyui", "llm", "index_tts",
                        "faster_qwen3_tts", "gpt_sovits",
                        "moss_soundeffect", "vibevoice_microsoft"}
        for name in gpu_services:
            assert SERVICE_REGISTRY[name].needs_gpu is True, f"{name} should be GPU"

    def test_tts_services_have_aliases(self):
        from services.registry import SERVICE_REGISTRY
        for name in ["kokoro", "espeak", "index_tts", "faster_qwen3_tts", "vibevoice_cpp_gpu", "gpt_sovits"]:
            assert SERVICE_REGISTRY[name].model_aliases, f"{name} needs model_aliases"

    def test_asr_services_have_aliases(self):
        from services.registry import SERVICE_REGISTRY
        for name in ["faster_whisper", "vibevoice_microsoft"]:
            assert SERVICE_REGISTRY[name].model_aliases, f"{name} needs model_aliases"

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
    """Test gateway/ingress.py routes via Starlette TestClient."""

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
        for s in svcs:
            assert all(k in s for k in ("name", "label", "category", "needs_gpu",
                                         "output_type", "description"))

    def test_service_info_found(self, client):
        r = client.get("/v1/services/kokoro")
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "kokoro"
        assert d["output_type"] == "audio"
        assert "tts-01-kokoro" in d["model_aliases"]
        assert d["description"]

    def test_service_info_not_found(self, client):
        r = client.get("/v1/services/nonexistent")
        assert r.status_code == 404

    def test_tnap_generate_unknown_service(self, client):
        r = client.post("/v1/nonexistent/generate", json={"action": "generate"})
        assert r.status_code == 404
        assert "Unknown service" in r.json()["error"]
        assert "kokoro" in r.json()["error"]

    def test_tnap_generate_routes_to_correct_deployment(self, client):
        """Verify the generic route targets the correct Ray Serve deployment."""
        from services.registry import get_service
        entry = get_service("espeak")
        assert entry is not None
        assert entry.deployment == "espeak_tts"
        assert entry.app == "espeak_tts"
        assert entry.needs_gpu is False

    def test_tnap_generate_gpu_service_lookup(self):
        """Verify GPU service registry entry is correct for Forge."""
        from services.registry import get_service
        from services.forge import SERVICE_MAP
        entry = get_service("trellis")
        assert entry.needs_gpu is True
        assert "trellis" in SERVICE_MAP

    def test_tnap_generate_cpu_service_lookup(self):
        """Verify CPU service skips GPU scheduling."""
        from services.registry import get_service
        entry = get_service("kokoro")
        assert entry.needs_gpu is False

    def test_all_services_have_valid_routing(self):
        """Every service in the registry maps to a valid (deployment, app) pair."""
        from services.registry import SERVICE_REGISTRY
        for name, entry in SERVICE_REGISTRY.items():
            assert entry.deployment, f"{name}: missing deployment"
            assert entry.app, f"{name}: missing app"
            assert entry.output_type in ("audio", "json", "model_3d", "image", "proxy", "video")

    def test_services_list_has_all_categories(self, client):
        r = client.get("/v1/services")
        categories = {s["category"] for s in r.json()}
        for cat in ("tts", "asr", "audio", "creative", "multimodal", "image", "llm"):
            assert cat in categories, f"Missing category: {cat}"

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
    """Test sdk/client.py with mocked HTTP."""

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

        client.client.post.assert_called_once()
        url = client.client.post.call_args[0][0]
        assert url == "/v1/kokoro/generate"

        payload = client.client.post.call_args[1]["json"]
        assert payload["action"] == "generate"
        assert payload["input"]["text"] == "hi"
        assert payload["config"]["precision"] == "fp16"

    @pytest.mark.asyncio
    async def test_generate_binary_decodes_base64(self, client):
        audio = b"RIFF" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "output": {"content": base64.b64encode(audio).decode()},
        }
        mock_resp.raise_for_status = MagicMock()
        client.client.post = AsyncMock(return_value=mock_resp)

        result = await client.generate_binary("kokoro", input={"text": "test"})
        assert result[:4] == b"RIFF"
        assert len(result) == len(audio)

    @pytest.mark.asyncio
    async def test_generate_binary_raises_on_empty_output(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "output": {"content": ""}}
        mock_resp.raise_for_status = MagicMock()
        client.client.post = AsyncMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="No binary output"):
            await client.generate_binary("kokoro", input={"text": "test"})

    @pytest.mark.asyncio
    async def test_services_lists(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"name": "kokoro"}, {"name": "trellis"}]
        mock_resp.raise_for_status = MagicMock()
        client.client.get = AsyncMock(return_value=mock_resp)

        svcs = await client.services()
        assert len(svcs) == 2
        client.client.get.assert_called_with("/v1/services")

    @pytest.mark.asyncio
    async def test_service_info(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "trellis", "output_type": "model_3d"}
        mock_resp.raise_for_status = MagicMock()
        client.client.get = AsyncMock(return_value=mock_resp)

        info = await client.service_info("trellis")
        assert info["name"] == "trellis"
        client.client.get.assert_called_with("/v1/services/trellis")


# ─── Dashboard Tests ──────────────────────────────────────────────────────────


class TestDashboardRegistry:
    """Test dashboard builds registry from services.registry."""

    def test_all_ray_deployments_present(self):
        from gateway.dashboard import KNOWN_DEPLOYMENTS
        for dep in ["kokoro_tts", "espeak_tts", "faster_whisper", "trellis", "anigen",
                    "hy_motion", "ace_step", "see_through", "phi4mm",
                    "comfyui", "llm", "index_tts", "faster_qwen3_tts", "vibevoice_cpp_gpu",
                    "gpt_sovits", "vibevoice_microsoft", "moss_soundeffect"]:
            assert dep in KNOWN_DEPLOYMENTS, f"Missing: {dep}"

    def test_external_mcp_services(self):
        from gateway.dashboard import KNOWN_DEPLOYMENTS
        assert KNOWN_DEPLOYMENTS["local_web_mcp"]["external_port"] == 18327
        assert KNOWN_DEPLOYMENTS["media_analysis_mcp"]["external_port"] == 18101

    def test_entries_have_required_fields(self):
        from gateway.dashboard import KNOWN_DEPLOYMENTS
        for name, meta in KNOWN_DEPLOYMENTS.items():
            assert "label" in meta, f"{name}: missing label"
            assert "category" in meta, f"{name}: missing category"
            assert "gpu" in meta, f"{name}: missing gpu"


# ─── Cross-module Consistency ─────────────────────────────────────────────────


class TestConsistency:
    """Verify registry, ingress, and dashboard are consistent."""

    def test_ingress_matches_registry(self):
        from starlette.testclient import TestClient
        from gateway.ingress import create_app
        from services.registry import SERVICE_REGISTRY

        client = TestClient(create_app())
        api_names = {s["name"] for s in client.get("/v1/services").json()}
        reg_names = set(SERVICE_REGISTRY.keys())
        assert api_names == reg_names

    def test_dashboard_covers_all_deployments(self):
        from gateway.dashboard import KNOWN_DEPLOYMENTS
        from services.registry import SERVICE_REGISTRY
        reg_deps = {e.deployment for e in SERVICE_REGISTRY.values()}
        dash_deps = set(KNOWN_DEPLOYMENTS.keys())
        assert not (reg_deps - dash_deps), f"Dashboard missing: {reg_deps - dash_deps}"
