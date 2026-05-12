"""Tests for the Forge — VRAM-aware GPU manager."""
from __future__ import annotations

import pytest

from services.forge_base import ForgeService
from services.forge import ForgeCore


class MockService(ForgeService):
    """Test service that records calls."""
    vram_mb = 4_096
    service_name = "mock"
    default_model = "mock-model"

    def __init__(self):
        super().__init__()
        self.load_calls = []
        self.unload_calls = []
        self.infer_calls = []

    def load(self, model_name: str) -> None:
        self.load_calls.append(model_name)
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self.unload_calls.append(self.model_name)
        self._loaded = False
        self.model_name = None
        super().unload()

    def infer(self, payload: dict) -> dict:
        self.infer_calls.append(payload)
        return {"status": "success", "service": self.service_name, "payload": payload}


class BigService(ForgeService):
    """Test service that uses lots of VRAM."""
    vram_mb = 20_480
    service_name = "big"
    default_model = "big-model"

    def load(self, model_name: str) -> None:
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        return {"status": "success"}


class SelfManagedService(ForgeService):
    """Test service with vram_mb=0 (self-managed like Wan2GP)."""
    vram_mb = 0
    service_name = "self_managed"
    default_model = "dynamic"

    def load(self, model_name: str) -> None:
        self._loaded = True
        self.model_name = model_name

    def unload(self) -> None:
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        return {"status": "success"}


# ─── ForgeService base class ────────────────────────────────────────────────

class TestForgeService:
    def test_default_state(self):
        svc = MockService()
        assert not svc.is_loaded()
        assert svc.vram_mb == 4_096
        assert svc.service_name == "mock"

    def test_load_sets_state(self):
        svc = MockService()
        svc.load("test-model")
        assert svc.is_loaded()
        assert svc.model_name == "test-model"

    def test_unload_clears_state(self):
        svc = MockService()
        svc.load("test-model")
        svc.unload()
        assert not svc.is_loaded()
        assert svc.model_name is None

    def test_infer_returns_dict(self):
        svc = MockService()
        svc.load("test")
        result = svc.infer({"key": "value"})
        assert result["status"] == "success"
        assert result["payload"] == {"key": "value"}

    def test_not_implemented(self):
        svc = ForgeService()
        with pytest.raises(NotImplementedError):
            svc.load("test")
        with pytest.raises(NotImplementedError):
            svc.infer({})

    def test_actual_vram_without_cuda(self):
        svc = MockService()
        assert svc.actual_vram_mb() == 0


# ─── ForgeCore VRAM tracking ────────────────────────────────────────────────

class TestForgeVRAM:
    def _make_forge(self):
        forge = ForgeCore(service_map={})
        return forge

    def test_initial_vram(self):
        forge = self._make_forge()
        assert forge._vram_free_mb == 22_528

    def test_can_fit_small_service(self):
        forge = self._make_forge()
        forge._register_service("mock", MockService())
        assert forge._can_fit("mock")

    def test_can_fit_too_large(self):
        forge = self._make_forge()
        forge._vram_allocations["existing"] = 20_000
        forge._vram_free_mb = 22_528 - 20_000
        forge._register_service("big", BigService())
        assert not forge._can_fit("big")

    def test_eviction_frees_space(self):
        forge = self._make_forge()
        big = BigService()
        big._loaded = True
        forge._register_service("big", big)
        forge._loaded["big"] = True
        forge._vram_allocations["big"] = 20_480
        forge._vram_free_mb = 22_528 - 20_480

        forge._register_service("mock", MockService())
        evicted = forge._evict_for("mock")

        assert "big" in evicted
        assert not forge._loaded.get("big", False)
        assert forge._vram_free_mb == 22_528

    def test_self_managed_always_fits(self):
        forge = self._make_forge()
        forge._vram_allocations["big"] = 22_528
        forge._vram_free_mb = 0
        forge._register_service("sm", SelfManagedService())
        assert forge._can_fit("sm")

    def test_total_allocated(self):
        forge = self._make_forge()
        forge._vram_allocations = {"a": 1000, "b": 2000}
        assert forge._total_allocated() == 3000

    def test_do_load_tracks_vram(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)

        forge._do_load("mock", "test-model")

        assert svc.is_loaded()
        assert forge._loaded["mock"]
        assert forge._vram_allocations["mock"] == 4_096
        assert forge._vram_free_mb == 22_528 - 4_096

    def test_do_unload_frees_vram(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._do_load("mock", "test-model")

        forge._do_unload("mock")

        assert not svc.is_loaded()
        assert "mock" not in forge._vram_allocations
        assert forge._vram_free_mb == 22_528

    def test_status_returns_state(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._do_load("mock", "test-model")

        status = forge.status_sync()
        assert "mock" in status["loaded"]
        assert status["vram_free_mb"] == 22_528 - 4_096

    def test_vram_zero_sum_invariant(self):
        """Total allocated + free should always equal AVAILABLE_MB."""
        forge = self._make_forge()
        forge._register_service("mock", MockService())
        forge._register_service("big", BigService())

        forge._do_load("mock", "test")
        assert forge._total_allocated() + forge._vram_free_mb == 22_528

        forge._do_unload("mock")
        assert forge._total_allocated() + forge._vram_free_mb == 22_528


# ─── Forge async invoke ─────────────────────────────────────────────────────

class TestForgeInvoke:
    def _make_forge(self):
        return ForgeCore(service_map={})

    @pytest.mark.asyncio
    async def test_invoke_loads_and_infers(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._service_map["mock"] = ("", "")

        result = await forge.invoke("mock", {"key": "val"})

        assert result["status"] == "success"
        assert svc.is_loaded()
        assert len(svc.load_calls) == 1
        assert len(svc.infer_calls) == 1

    @pytest.mark.asyncio
    async def test_invoke_already_loaded_skips_load(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._service_map["mock"] = ("", "")

        await forge.invoke("mock", {"a": 1})
        await forge.invoke("mock", {"b": 2})

        assert len(svc.load_calls) == 1
        assert len(svc.infer_calls) == 2

    @pytest.mark.asyncio
    async def test_invoke_unknown_service(self):
        forge = self._make_forge()
        result = await forge.invoke("nonexistent", {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_release_unloads_all(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._service_map["mock"] = ("", "")

        await forge.invoke("mock", {})
        assert svc.is_loaded()

        result = await forge.release()
        assert not svc.is_loaded()
        assert result["status"] == "released"

    @pytest.mark.asyncio
    async def test_status_after_invoke(self):
        forge = self._make_forge()
        svc = MockService()
        forge._register_service("mock", svc)
        forge._service_map["mock"] = ("", "")

        await forge.invoke("mock", {})
        status = await forge.status()

        assert "mock" in status["loaded"]
        assert status["vram_free_mb"] == 22_528 - 4_096

    @pytest.mark.asyncio
    async def test_swap_evicts_old_service(self):
        forge = self._make_forge()
        mock_svc = MockService()
        big_svc = BigService()
        forge._register_service("mock", mock_svc)
        forge._register_service("big", big_svc)
        forge._service_map["mock"] = ("", "")
        forge._service_map["big"] = ("", "")

        # Load mock (4GB)
        await forge.invoke("mock", {"a": 1})
        assert mock_svc.is_loaded()

        # Load big (20GB) — should evict mock first
        await forge.invoke("big", {"b": 2})
        assert not mock_svc.is_loaded()
        assert big_svc.is_loaded()
        assert forge._vram_allocations.get("big") == 20_480

    @pytest.mark.asyncio
    async def test_self_managed_coexists(self):
        forge = self._make_forge()
        mock_svc = MockService()
        sm_svc = SelfManagedService()
        forge._register_service("mock", mock_svc)
        forge._register_service("sm", sm_svc)
        forge._service_map["mock"] = ("", "")
        forge._service_map["sm"] = ("", "")

        # Load mock (4GB)
        await forge.invoke("mock", {})
        assert mock_svc.is_loaded()

        # Load self-managed — should NOT evict mock (vram_mb=0 always fits)
        await forge.invoke("sm", {})
        assert mock_svc.is_loaded()
        assert sm_svc.is_loaded()


# ─── Subprocess mixin ───────────────────────────────────────────────────────

class TestForgeSubprocessMixin:
    def test_not_running_initially(self):
        from services.forge_subprocess import ForgeSubprocessMixin

        class TestSvc(ForgeSubprocessMixin, ForgeService):
            pass

        svc = TestSvc()
        assert not svc.is_running()

    def test_stop_when_not_started(self):
        from services.forge_subprocess import ForgeSubprocessMixin

        class TestSvc(ForgeSubprocessMixin, ForgeService):
            pass

        svc = TestSvc()
        svc.stop_subprocess()  # Should not raise
