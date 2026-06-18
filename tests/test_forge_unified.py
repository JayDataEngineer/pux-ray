"""Tests for the unified Forge interface — persistence, ForgeProxy, smart eviction."""
from __future__ import annotations

import pytest

from services.forge import ForgeCore, AVAILABLE_MB
from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.forge_proxy import ForgeProxy


class MockForgeService(ForgeService):
    """Mock service for testing VRAM tracking and persistence."""

    def __init__(self, vram_mb=1000, persistence=Persistence.TRANSIENT):
        super().__init__()
        self._vram_mb = vram_mb
        self._load_count = 0
        self._infer_count = 0
        self._last_payload = None
        self.persistence = persistence

    @property
    def vram_mb(self):
        return self._vram_mb

    def load(self, model_name=None, quant=None):
        self._loaded = True
        self._load_count += 1
        self.model_name = model_name

    def unload(self):
        self._loaded = False
        self.model_name = None

    def infer(self, payload):
        self._infer_count += 1
        self._last_payload = payload
        return {"status": "ok", "model": self.model_name}

    def actual_vram_mb(self):
        return self._vram_mb if self._loaded else 0


# ─── Persistence Enum ──────────────────────────────────────────────────────

def test_persistence_ordering():
    assert Persistence.TRANSIENT < Persistence.PERSISTENT < Persistence.PIPELINE_LOCKED


# ─── Smart Eviction ────────────────────────────────────────────────────────

def _make_core(services=None):
    core = ForgeCore(service_map={})
    if services:
        for name, svc in services.items():
            core._register_service(name, svc)
    return core


def test_evict_transient_before_persistent():
    """Transient services are evicted before persistent ones."""
    transient = MockForgeService(vram_mb=8000, persistence=Persistence.TRANSIENT)
    persistent = MockForgeService(vram_mb=8000, persistence=Persistence.PERSISTENT)

    core = _make_core({"transient_svc": transient, "persistent_svc": persistent})

    # Load both (free = AVAILABLE - 8000 - 8000 = AVAILABLE - 16000)
    core._do_load("transient_svc")
    core._do_load("persistent_svc")

    # Request a service that needs more than free but less than free + transient
    # free = 22528 - 16000 = 6528. Need 10000 → evicting transient (8000) gives 14528 > 10000.
    needy_svc = MockForgeService(vram_mb=10000)
    core._register_service("needy_svc", needy_svc)

    evicted = core._evict_for("needy_svc")

    # Only transient should be evicted
    assert "transient_svc" in evicted
    assert "persistent_svc" not in evicted


def test_never_evict_pipeline_locked():
    """Pipeline-locked services are never evicted."""
    locked = MockForgeService(vram_mb=10000, persistence=Persistence.PIPELINE_LOCKED)

    core = _make_core({"locked_svc": locked})
    core._do_load("locked_svc")

    big_svc = MockForgeService(vram_mb=AVAILABLE_MB)
    core._register_service("big_svc", big_svc)

    with pytest.raises(RuntimeError, match="Cannot free enough VRAM"):
        core._evict_for("big_svc")

    assert core._loaded["locked_svc"]


def test_persistence_override():
    """Runtime overrides take precedence over service defaults."""
    svc = MockForgeService(vram_mb=5000, persistence=Persistence.TRANSIENT)
    core = _make_core({"svc": svc})
    core._do_load("svc")

    # Override to pipeline-locked
    core._persistence_overrides["svc"] = Persistence.PIPELINE_LOCKED

    assert core._get_persistence("svc") == Persistence.PIPELINE_LOCKED

    # Should not be evictable
    big_svc = MockForgeService(vram_mb=AVAILABLE_MB)
    core._register_service("big_svc", big_svc)

    with pytest.raises(RuntimeError, match="Cannot free enough VRAM"):
        core._evict_for("big_svc")

    # Clean up override
    del core._persistence_overrides["svc"]
    assert core._get_persistence("svc") == Persistence.TRANSIENT


def test_evict_largest_first_within_same_persistence():
    """Within the same persistence level, largest VRAM consumer is evicted first."""
    small = MockForgeService(vram_mb=2000, persistence=Persistence.TRANSIENT)
    large = MockForgeService(vram_mb=8000, persistence=Persistence.TRANSIENT)

    core = _make_core({"small_svc": small, "large_svc": large})
    core._do_load("small_svc")
    core._do_load("large_svc")

    big_svc = MockForgeService(vram_mb=AVAILABLE_MB)
    core._register_service("big_svc", big_svc)

    evicted = core._evict_for("big_svc")

    # Large should be evicted first
    assert evicted[0] == "large_svc"


# ─── ForgeProxy ────────────────────────────────────────────────────────────

def test_proxy_first_load():
    """First load goes through Forge's full lifecycle."""
    adapter = MockForgeService(vram_mb=3000)
    adapter.persistence = Persistence.TRANSIENT

    core = _make_core({"native": adapter})
    proxy = ForgeProxy(core)

    proxy.load("z_image")

    assert proxy._native_loaded is True
    assert core._loaded["native"]
    assert adapter.model_name == "z_image"


def test_proxy_model_swap():
    """Subsequent loads swap the model and reconcile VRAM."""
    adapter = MockForgeService(vram_mb=3000)

    core = _make_core({"native": adapter})
    proxy = ForgeProxy(core)

    # First load
    proxy.load("z_image")
    assert core._vram_allocations["native"] == 3000

    # Simulate VRAM change on model swap
    adapter._vram_mb = 5000
    proxy.load("qwen-image-edit")

    assert adapter.model_name == "qwen-image-edit"
    assert core._vram_allocations["native"] == 5000


def test_proxy_infer():
    """Infer delegates to the adapter."""
    adapter = MockForgeService(vram_mb=3000)

    core = _make_core({"wan2gp": adapter})
    proxy = ForgeProxy(core)

    proxy.load("z_image")
    result = proxy.infer({"prompt": "test"})

    assert result["status"] == "ok"
    assert result["model"] == "z_image"
    assert adapter._infer_count == 1


def test_proxy_unload():
    """Unload removes the service from the Forge."""
    adapter = MockForgeService(vram_mb=3000)

    core = _make_core({"wan2gp": adapter})
    proxy = ForgeProxy(core)

    proxy.load("z_image")
    assert core._loaded["wan2gp"]

    proxy.unload()
    assert not core._loaded["wan2gp"]
    assert proxy._wan2gp_loaded is False
