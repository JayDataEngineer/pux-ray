"""Tests for GPUGovernor — VRAM constants and lease state machine.

No Ray cluster needed. Extracts the inner class via __ray_actor_class__.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


# ─── VRAM Configuration Tests ──────────────────────────────────────────────


class TestHeavyServicesConfig:

    def test_all_heavy_services_have_vram_estimates(self):
        from gateway.gpu_governor import HEAVY_SERVICES
        for name, vram in HEAVY_SERVICES.items():
            assert isinstance(vram, int), f"{name}: vram is not int"
            assert vram > 0, f"{name}: vram must be positive"

    def test_no_service_exceeds_total_vram(self):
        from gateway.gpu_governor import HEAVY_SERVICES, TOTAL_VRAM_MB
        for name, vram in HEAVY_SERVICES.items():
            assert vram <= TOTAL_VRAM_MB, f"{name}: {vram}MB > {TOTAL_VRAM_MB}MB total"

    def test_available_equals_total_minus_reserved(self):
        from gateway.gpu_governor import AVAILABLE_MB, TOTAL_VRAM_MB, RESERVED_MB
        assert AVAILABLE_MB == TOTAL_VRAM_MB - RESERVED_MB

    def test_governor_heavy_services_match_router(self):
        """GPU Governor and MasterRouter should agree on which services are heavy."""
        from gateway.gpu_governor import HEAVY_SERVICES as gov_services
        from services.creative.master_router import HEAVY_SERVICES as router_services
        for name in gov_services:
            assert name in router_services, f"{name} in governor but not master_router"


# ─── Governor State Machine Tests ──────────────────────────────────────────


def _make_governor():
    """Create a GPUGovernor instance without @ray.remote wrapper."""
    from gateway.gpu_governor import GPUGovernor
    inner_cls = GPUGovernor.__ray_actor_class__
    gov = inner_cls()
    return gov


class TestGPUGovernor:

    def test_initial_state_no_holder(self):
        gov = _make_governor()
        assert gov._holder is None

    def test_acquire_grants_lease(self):
        gov = _make_governor()
        result = asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        assert result["granted"] is True
        assert result["evicted"] is None
        assert gov._holder == "trellis"

    def test_acquire_same_service_idempotent(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        result = asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        assert result["evicted"] is None
        assert gov._holder == "trellis"

    def test_acquire_evicts_previous_holder(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))

        with patch("ray.serve") as mock_serve:
            mock_handle = AsyncMock()
            mock_serve.get_deployment_handle.return_value = mock_handle

            result = asyncio.get_event_loop().run_until_complete(gov.acquire("ace_step"))

        assert result["evicted"] == "trellis"
        assert gov._holder == "ace_step"

    def test_release_clears_holder(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        asyncio.get_event_loop().run_until_complete(gov.release("trellis"))
        assert gov._holder is None

    def test_release_no_op_if_not_holder(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        asyncio.get_event_loop().run_until_complete(gov.release("ace_step"))
        assert gov._holder == "trellis"

    def test_status_returns_holder_info(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("llm"))
        status = asyncio.get_event_loop().run_until_complete(gov.status())
        assert status["holder"] == "llm"
        assert status["holder_vram_mb"] > 0
        assert status["total_mb"] == 24576

    def test_park_releases_lease(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        asyncio.get_event_loop().run_until_complete(gov.park("trellis"))
        assert gov._holder is None

    def test_unpark_acquires_lease(self):
        gov = _make_governor()
        asyncio.get_event_loop().run_until_complete(gov.acquire("trellis"))
        asyncio.get_event_loop().run_until_complete(gov.park("trellis"))
        result = asyncio.get_event_loop().run_until_complete(gov.unpark("trellis", 8192))
        assert result["granted"] is True
        assert gov._holder == "trellis"
