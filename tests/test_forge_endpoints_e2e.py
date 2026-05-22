"""E2E tests for Forge HTTP endpoints — tests the HTTP layer with real request/response.

Tests two HTTP surfaces:
  1. Forge __call__ logic — replicated as a standalone ASGI app wrapping ForgeCore
  2. Ingress API routes — the Starlette ingress app (mocks Ray handles)

Uses Starlette TestClient for real HTTP request/response without a server.
"""
from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.forge import ForgeCore, SERVICE_MAP, AVAILABLE_MB
from services.forge_base import ForgeService
from services.forge_persistence import Persistence


# ─── Mock Services ──────────────────────────────────────────────────────────

class MockGPUService(ForgeService):
    vram_mb = 4_096
    service_name = "mock_gpu"
    default_model = "mock-model-v1"

    def __init__(self):
        super().__init__()
        self.load_calls = []
        self.infer_calls = []

    def load(self, model_name: str, quant: str | None = None) -> None:
        self.load_calls.append((model_name, quant))
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None

    def actual_vram_mb(self) -> int:
        return self.vram_mb if self._loaded else 0

    def infer(self, payload: dict) -> dict:
        self.infer_calls.append(payload)
        return {
            "status": "success",
            "service": "mock_gpu",
            "model": self.model_name,
            "echo": payload,
        }


class MockBigService(ForgeService):
    vram_mb = 20_480
    service_name = "mock_big"
    default_model = "big-model"

    def __init__(self):
        super().__init__()
        self.load_calls = []

    def load(self, model_name: str, quant: str | None = None) -> None:
        self.load_calls.append((model_name, quant))
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None

    def actual_vram_mb(self) -> int:
        return self.vram_mb if self._loaded else 0

    def infer(self, payload: dict) -> dict:
        return {"status": "success", "service": "mock_big"}


class MockSelfManagedService(ForgeService):
    vram_mb = 0
    service_name = "mock_mmgp"
    default_model = "dynamic"

    def load(self, model_name: str, quant: str | None = None) -> None:
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        return {"status": "success", "service": "mock_mmgp"}


# ─── ASGI App ForgeCore HTTP Handler ───────────────────────────────────────
# Replicates the Forge.__call__ dispatch logic as a standalone ASGI app
# so we can test the HTTP layer without Ray's @serve.deployment wrapper.

def _make_forge_app(core: ForgeCore):
    """Build an ASGI app that replicates Forge.__call__ dispatch."""

    async def app(scope, receive, send):
        request = Request(scope, receive=receive)

        if request.method == "GET":
            response = JSONResponse(core.status_sync())
            await response(scope, receive, send)
            return

        try:
            body = await request.json()
        except Exception:
            response = JSONResponse(
                {"status": "error", "error": "invalid JSON body"},
                status_code=400,
            )
            await response(scope, receive, send)
            return

        try:
            action = body.get("action", "")
            if action == "release":
                svc = body.get("service")
                result = await core.release(svc)
                response = JSONResponse(result)
                await response(scope, receive, send)
                return

            if action == "status":
                response = JSONResponse(core.status_sync())
                await response(scope, receive, send)
                return

            if action == "preload":
                service = body.get("service")
                model = body.get("model")
                quant = body.get("quant")
                result = await core.preload(service, model, quant)
                response = JSONResponse(result)
                await response(scope, receive, send)
                return

            service = body.get("service")
            if not service or service not in core._service_map:
                response = JSONResponse(
                    {"status": "error",
                     "error": f"Specify 'service' as one of {sorted(core._service_map)}"},
                    status_code=400,
                )
                await response(scope, receive, send)
                return

            payload = {k: v for k, v in body.items() if k != "service"}
            model = payload.pop("model", None)
            quant = payload.pop("quant", None)
            result = await core.invoke(service, payload, model, quant)
            response = JSONResponse(result)
            await response(scope, receive, send)
        except Exception as e:
            response = JSONResponse(
                {"status": "error", "error": str(e)},
                status_code=500,
            )
            await response(scope, receive, send)

    return app


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def core() -> ForgeCore:
    c = ForgeCore(service_map={})
    c._register_service("mock_gpu", MockGPUService())
    c._register_service("mock_big", MockBigService())
    c._register_service("mock_mmgp", MockSelfManagedService())
    c._service_map["mock_gpu"] = ("__direct__", "MockGPUService")
    c._service_map["mock_big"] = ("__direct__", "MockBigService")
    c._service_map["mock_mmgp"] = ("__direct__", "MockSelfManagedService")
    return c


@pytest.fixture
def client(core: ForgeCore) -> TestClient:
    return TestClient(_make_forge_app(core))


# ═══════════════════════════════════════════════════════════════════════════
# Forge HTTP endpoint tests
# ═══════════════════════════════════════════════════════════════════════════

class TestForgeHTTPStatus:
    """GET / and POST with action=status"""

    def test_get_returns_status(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "loaded" in body
        assert "vram_free_mb" in body
        assert "vram_total_mb" in body
        assert body["vram_total_mb"] == AVAILABLE_MB

    def test_get_shows_no_loaded_services_initially(self, client: TestClient):
        resp = client.get("/")
        assert resp.json()["loaded"] == {}

    def test_post_action_status(self, client: TestClient):
        resp = client.post("/", json={"action": "status"})
        assert resp.status_code == 200
        body = resp.json()
        assert "loaded" in body
        assert body["vram_total_mb"] == AVAILABLE_MB

    def test_post_action_status_after_load(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        resp = client.post("/", json={"action": "status"})
        body = resp.json()
        assert "mock_gpu" in body["loaded"]

    def test_status_vram_fields(self, client: TestClient):
        resp = client.get("/")
        body = resp.json()
        assert body["vram_total_mb"] == AVAILABLE_MB
        assert body["vram_free_mb"] == AVAILABLE_MB
        assert body["vram_allocated_mb"] == 0


class TestForgeHTTPPreload:
    """POST / with action=preload"""

    def test_preload_loads_service(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "action": "preload",
            "service": "mock_gpu",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "loaded"
        assert body["service"] == "mock_gpu"
        assert body["vram_used_mb"] == 4_096
        assert body["vram_free_mb"] == AVAILABLE_MB - 4_096

    def test_preload_already_loaded(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        resp = client.post("/", json={"action": "preload", "service": "mock_gpu"})
        assert resp.json()["status"] == "already_loaded"

    def test_preload_with_model(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "action": "preload",
            "service": "mock_gpu",
            "model": "custom-model-v2",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        svc = core._services["mock_gpu"]
        assert svc.model_name == "custom-model-v2"

    def test_preload_with_quant(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "action": "preload",
            "service": "mock_gpu",
            "model": "quant-model",
            "quant": "q4_k_m",
        })
        assert resp.status_code == 200
        svc = core._services["mock_gpu"]
        assert svc.model_name == "quant-model"
        assert svc.load_calls[-1][1] == "q4_k_m"

    def test_preload_unknown_service(self, client: TestClient):
        resp = client.post("/", json={
            "action": "preload",
            "service": "nonexistent",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_preload_triggers_eviction(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_big"})
        resp = client.post("/", json={"action": "preload", "service": "mock_gpu"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"
        status = client.get("/").json()
        assert "mock_big" not in status["loaded"]
        assert "mock_gpu" in status["loaded"]

    def test_preload_without_service(self, client: TestClient):
        resp = client.post("/", json={"action": "preload"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


class TestForgeHTTPRelease:
    """POST / with action=release"""

    def test_release_all(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "preload", "service": "mock_mmgp"})

        resp = client.post("/", json={"action": "release"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "released"
        assert len(body["services"]) >= 1

        status = client.get("/").json()
        assert status["loaded"] == {}
        assert status["vram_free_mb"] == AVAILABLE_MB

    def test_release_specific_service(self, client: TestClient, core: ForgeCore):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "preload", "service": "mock_mmgp"})

        resp = client.post("/", json={"action": "release", "service": "mock_gpu"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "released"
        assert resp.json()["service"] == "mock_gpu"

        # mock_gpu freed VRAM
        assert core._vram_free_mb == AVAILABLE_MB
        # mock_gpu is unloaded
        assert not core._loaded.get("mock_gpu", False)

    def test_release_not_loaded(self, client: TestClient):
        resp = client.post("/", json={"action": "release", "service": "mock_gpu"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "released"

    def test_release_no_action_key(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        resp = client.post("/", json={"foo": "bar"})
        assert resp.status_code == 400

    def test_release_frees_vram(self, client: TestClient, core: ForgeCore):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        assert core._vram_allocations.get("mock_gpu") == 4_096

        client.post("/", json={"action": "release", "service": "mock_gpu"})
        assert core._vram_allocations.get("mock_gpu") is None
        assert core._vram_free_mb == AVAILABLE_MB

    def test_release_all_with_mixed_services(self, client: TestClient, core: ForgeCore):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "preload", "service": "mock_mmgp"})

        resp = client.post("/", json={"action": "release"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "released"
        assert not any(core._loaded.values())
        assert core._vram_free_mb == AVAILABLE_MB


class TestForgeHTTPInvoke:
    """POST / with service payload (no action key)"""

    def test_invoke_loads_on_demand(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "service": "mock_gpu",
            "prompt": "hello",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["service"] == "mock_gpu"
        assert core._loaded["mock_gpu"]

    def test_invoke_reuses_loaded_service(self, client: TestClient, core: ForgeCore):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        svc = core._services["mock_gpu"]
        load_count = len(svc.load_calls)

        resp = client.post("/", json={
            "service": "mock_gpu",
            "data": "test",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert len(svc.load_calls) == load_count

    def test_invoke_with_model(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "service": "mock_gpu",
            "model": "custom-llm",
            "prompt": "hello",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "custom-llm"

    def test_invoke_with_quant(self, client: TestClient):
        resp = client.post("/", json={
            "service": "mock_gpu",
            "model": "qwen",
            "quant": "q4_k_m",
            "prompt": "hello",
        })
        assert resp.status_code == 200

    def test_invoke_unknown_service(self, client: TestClient):
        resp = client.post("/", json={
            "service": "nonexistent",
            "prompt": "hello",
        })
        assert resp.status_code == 400

    def test_invoke_without_service_key(self, client: TestClient):
        resp = client.post("/", json={"prompt": "hello"})
        assert resp.status_code == 400

    def test_invoke_missing_body(self, client: TestClient):
        resp = client.post("/", content=b"", headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_invoke_malformed_json(self, client: TestClient):
        resp = client.post("/", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 400


class TestForgeHTTPVRAMLifecycle:
    """End-to-end VRAM tracking through HTTP"""

    def test_vram_invariant_after_load_unload(self, client: TestClient):
        initial = client.get("/").json()
        assert initial["vram_free_mb"] + initial["vram_allocated_mb"] == AVAILABLE_MB

        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        after_load = client.get("/").json()
        assert after_load["vram_free_mb"] == AVAILABLE_MB - 4_096
        assert after_load["vram_allocated_mb"] == 4_096
        assert after_load["vram_free_mb"] + after_load["vram_allocated_mb"] == AVAILABLE_MB

        client.post("/", json={"action": "release"})
        after_release = client.get("/").json()
        assert after_release["vram_free_mb"] == AVAILABLE_MB
        assert after_release["vram_allocated_mb"] == 0

    def test_self_managed_service_no_vram_tracking(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_mmgp"})
        status = client.get("/").json()
        assert status["vram_free_mb"] == AVAILABLE_MB
        assert status["vram_allocated_mb"] == 0
        # Self-managed (vram_mb=0) doesn't appear in status.loaded
        # because it has no vram allocation entry

    def test_gpu_and_self_managed_both_loaded(self, client: TestClient, core: ForgeCore):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "preload", "service": "mock_mmgp"})
        status = client.get("/").json()
        assert "mock_gpu" in status["loaded"]
        # mock_mmgp is internally loaded but not in vram_allocations
        assert core._loaded["mock_mmgp"]

    def test_eviction_frees_vram_for_new_service(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})

        client.post("/", json={"action": "preload", "service": "mock_big"})
        status = client.get("/").json()

        assert "mock_gpu" not in status["loaded"]
        assert "mock_big" in status["loaded"]
        assert status["vram_free_mb"] == AVAILABLE_MB - 20_480

    def test_concurrent_small_services(self, core: ForgeCore):
        core._register_service("mock_gpu2", MockGPUService())
        core._service_map["mock_gpu2"] = ("__direct__", "MockGPUService")
        client = TestClient(_make_forge_app(core))

        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "preload", "service": "mock_gpu2"})
        status = client.get("/").json()

        assert "mock_gpu" in status["loaded"]
        assert "mock_gpu2" in status["loaded"]
        assert status["vram_allocated_mb"] == 8_192


class TestForgeHTTPEdgeCases:
    """Edge cases and error handling through HTTP"""

    def test_no_loaded_services_after_initialize(self, client: TestClient):
        status = client.get("/").json()
        assert status["loaded"] == {}

    def test_release_all_when_nothing_loaded(self, client: TestClient):
        resp = client.post("/", json={"action": "release"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "released"
        assert resp.json()["services"] == []

    def test_preload_with_empty_service_string(self, client: TestClient):
        resp = client.post("/", json={"action": "preload", "service": ""})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_invoke_with_empty_service_string(self, client: TestClient):
        resp = client.post("/", json={"service": "", "prompt": "test"})
        assert resp.status_code == 400

    def test_get_after_load_unload_cycle(self, client: TestClient):
        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        client.post("/", json={"action": "release"})
        status = client.get("/").json()
        assert status["loaded"] == {}
        assert status["vram_free_mb"] == AVAILABLE_MB

    def test_multiple_preloads_same_service(self, client: TestClient):
        r1 = client.post("/", json={"action": "preload", "service": "mock_gpu"})
        r2 = client.post("/", json={"action": "preload", "service": "mock_gpu"})
        r3 = client.post("/", json={"action": "preload", "service": "mock_gpu"})
        assert r1.json()["status"] == "loaded"
        assert r2.json()["status"] == "already_loaded"
        assert r3.json()["status"] == "already_loaded"

    def test_invoke_then_release_then_invoke_again(self, client: TestClient):
        resp1 = client.post("/", json={"service": "mock_gpu", "data": "first"})
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "success"

        client.post("/", json={"action": "release"})

        resp2 = client.post("/", json={"service": "mock_gpu", "data": "second"})
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "success"

    def test_payload_passthrough(self, client: TestClient):
        payload = {"service": "mock_gpu", "key1": "val1", "key2": 42, "nested": {"a": [1, 2]}}
        resp = client.post("/", json=payload)
        assert resp.status_code == 200
        echo = resp.json()["echo"]
        assert echo["key1"] == "val1"
        assert echo["key2"] == 42
        assert echo["nested"] == {"a": [1, 2]}

    def test_invoke_with_extra_fields(self, client: TestClient):
        resp = client.post("/", json={
            "service": "mock_gpu",
            "prompt": "hello",
            "extra_field": "should_be_preserved",
            "another": 123,
        })
        assert resp.status_code == 200
        echo = resp.json()["echo"]
        assert echo["extra_field"] == "should_be_preserved"
        assert echo["another"] == 123

    def test_invoke_returns_model_name(self, client: TestClient, core: ForgeCore):
        resp = client.post("/", json={
            "service": "mock_gpu",
            "model": "my-model-name",
            "prompt": "test",
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "my-model-name"


class TestForgeHTTPPersistence:
    """Persistence levels affect eviction order."""

    def test_transient_evicted_before_persistent(self, core: ForgeCore):
        """Within same persistence level, largest vram consumer evicted first.
        With different persistence, transient before persistent."""
        from services.forge_persistence import Persistence

        # Create services with specific VRAM sizes.
        # actual_vram_mb returns the declared vram_mb when loaded (no CUDA available in tests).

        class SmallTransient(ForgeService):
            vram_mb = 2_000; service_name = "small_t"; default_model = "m"
            def load(self, model_name: str, quant: str | None = None): self._loaded = True; self.model_name = model_name
            def unload(self): self._loaded = False; self.model_name = None
            def infer(self, p): return {"status": "success"}
            def actual_vram_mb(self): return self.vram_mb if self._loaded else 0

        class MediumPersistent(ForgeService):
            vram_mb = 4_000; service_name = "medium_p"; default_model = "m"
            persistence = Persistence.PERSISTENT
            def load(self, model_name: str, quant: str | None = None): self._loaded = True; self.model_name = model_name
            def unload(self): self._loaded = False; self.model_name = None
            def infer(self, p): return {"status": "success"}
            def actual_vram_mb(self): return self.vram_mb if self._loaded else 0

        class LargeTransient(ForgeService):
            vram_mb = 6_000; service_name = "large_t"; default_model = "m"
            def load(self, model_name: str, quant: str | None = None): self._loaded = True; self.model_name = model_name
            def unload(self): self._loaded = False; self.model_name = None
            def infer(self, p): return {"status": "success"}
            def actual_vram_mb(self): return self.vram_mb if self._loaded else 0

        core._register_service("small_t", SmallTransient())
        core._register_service("medium_p", MediumPersistent())
        core._register_service("large_t", LargeTransient())
        core._service_map["small_t"] = ("__direct__", "")
        core._service_map["medium_p"] = ("__direct__", "")
        core._service_map["large_t"] = ("__direct__", "")

        client = TestClient(_make_forge_app(core))

        # Load all three: total = 2000 + 4000 + 6000 = 12000. Free = 22528 - 12000 = 10528
        resp_small = client.post("/", json={"action": "preload", "service": "small_t"})
        assert resp_small.status_code == 200, f"small_t preload failed: {resp_small.json()}"
        resp_med = client.post("/", json={"action": "preload", "service": "medium_p"})
        assert resp_med.status_code == 200, f"medium_p preload failed: {resp_med.json()}"
        resp_large = client.post("/", json={"action": "preload", "service": "large_t"})
        assert resp_large.status_code == 200, f"large_t preload failed: {resp_large.json()}"

        status = client.get("/").json()
        assert "small_t" in status["loaded"], f"small_t missing from {status}"
        assert "medium_p" in status["loaded"], f"medium_p missing from {status}"
        assert "large_t" in status["loaded"], f"large_t missing from {status}"

        # Request a new service needing 15000 MB. Free = 10528, need eviction.
        # Eviction order by (persistence value, -vram_mb):
        #   small_t: (0, -2000), large_t: (0, -6000), medium_p: (1, -4000)
        # Sorted: [large_t (0,6k), small_t (0,2k), medium_p (1,4k)]
        # After evicting large_t (6000): free = 10528 + 6000 = 16528 >= 15000 ✓
        class MediumNeedy(ForgeService):
            vram_mb = 15_000; service_name = "needy"; default_model = "m"
            def load(self, model_name: str, quant: str | None = None): self._loaded = True; self.model_name = model_name
            def unload(self): self._loaded = False; self.model_name = None
            def infer(self, p): return {"status": "success"}
            def actual_vram_mb(self): return self.vram_mb if self._loaded else 0

        core._register_service("needy", MediumNeedy())
        core._service_map["needy"] = ("__direct__", "")

        resp = client.post("/", json={"action": "preload", "service": "needy"})
        assert resp.status_code == 200

        status = client.get("/").json()
        # large_t (largest transient) should be evicted first
        assert "large_t" not in status["loaded"], "Largest transient should be evicted first"
        # small_t and medium_p should still be loaded
        assert "small_t" in status["loaded"]
        assert "medium_p" in status["loaded"]
        assert "needy" in status["loaded"]

    def test_pipeline_locked_not_evicted(self, core: ForgeCore):
        core._persistence_overrides["mock_gpu"] = Persistence.PIPELINE_LOCKED
        client = TestClient(_make_forge_app(core))

        client.post("/", json={"action": "preload", "service": "mock_gpu"})
        resp = client.post("/", json={"action": "preload", "service": "mock_big"})
        # Cannot free enough VRAM — pipeline-locked service can't be evicted
        assert resp.status_code == 500
        assert resp.json()["status"] == "error"
        assert "Cannot free enough VRAM" in resp.json()["error"]

        status = client.get("/").json()
        assert "mock_gpu" in status["loaded"]


# ═══════════════════════════════════════════════════════════════════════════
# Ingress API route tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIngressRoutes:
    """Ingress health and discovery endpoints."""

    @pytest.fixture
    def ingress_client(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        mock_handle = MagicMock()
        mock_handle.status.remote = AsyncMock(return_value={
            "loaded": {}, "vram_free_mb": 22528, "vram_total_mb": 22528,
            "vram_allocated_mb": 0, "gpu": {}, "gpu_nodes": {},
        })
        mock_handle.preload.remote = AsyncMock(return_value={
            "status": "loaded", "service": "test", "vram_used_mb": 4096,
            "vram_free_mb": 18432,
        })
        mock_handle.release.remote = AsyncMock(return_value={
            "status": "released", "services": [],
        })
        mock_handle.invoke.remote = AsyncMock(return_value={
            "status": "success", "output": "mock_output",
        })

        import gateway.ingress as ingress_mod
        monkeypatch.setattr(ingress_mod, "_get_forge", lambda: mock_handle)
        app = ingress_mod.create_app()
        return TestClient(app)

    def test_health(self, ingress_client: TestClient):
        resp = ingress_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_status(self, ingress_client: TestClient):
        resp = ingress_client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "vram" in body
        assert "resources" in body

    def test_list_services(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/services")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        svc_names = [s["name"] for s in body]
        assert "llm" in svc_names
        assert "wan2gp" in svc_names
        assert "comfyui" in svc_names

    def test_service_info(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/services/llm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "llm"
        assert "label" in body
        assert "category" in body

    def test_service_info_not_found(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/services/nonexistent")
        assert resp.status_code == 404

    def test_list_models(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) > 0

    def test_list_models_filtered(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/models?category=tts")
        assert resp.status_code == 200
        for model in resp.json()["data"]:
            assert model["category"] == "tts"

    def test_admin_load(self, ingress_client: TestClient):
        resp = ingress_client.post("/admin/load", json={"service": "llm"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"

    def test_admin_load_missing_service(self, ingress_client: TestClient):
        resp = ingress_client.post("/admin/load", json={})
        assert resp.status_code == 400

    def test_admin_unload(self, ingress_client: TestClient):
        resp = ingress_client.post("/admin/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "released"

    def test_run_catalog(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/run/catalog")
        assert resp.status_code == 200
        body = resp.json()
        assert "pipelines" in body
        assert "services" in body
        assert len(body["pipelines"]) > 0
        assert len(body["services"]) > 0

    def test_v1_models_has_expected_fields(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "owned_by" in model
        assert "category" in model

    def test_service_has_model_aliases(self, ingress_client: TestClient):
        resp = ingress_client.get("/v1/services")
        svc = next(s for s in resp.json() if s["name"] == "wan2gp")
        assert "model_aliases" in svc
        assert len(svc["model_aliases"]) > 0
