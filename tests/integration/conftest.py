"""Integration test fixtures — VRAM-aware service lifecycle.

Provides `wan2gp_svc` fixture that:
  - Creates a fresh Wan2GPService per test
  - Unloads model + clears VRAM after each test (even on failure)
  - Verifies VRAM returns to baseline (detects leaks)

Also registers the `autoregressive` marker for skip-by-marker.
"""
from __future__ import annotations

import gc

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "autoregressive: >10min inference, skip by default")


def pytest_collection_modifyitems(config, items):
    """Auto-skip autoregressive tests unless explicitly selected with -m."""
    expr = config.getoption("-m", default="")
    if "autoregressive" not in expr:
        skip_auto = pytest.mark.skip(reason="autoregressive — run with -m autoregressive")
        for item in items:
            if "autoregressive" in item.keywords:
                item.add_marker(skip_auto)


def _vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        pass
    return 0


@pytest.fixture
def wan2gp_svc():
    """Yield a Wan2GPService with guaranteed VRAM cleanup after each test."""
    from services.wan2gp.deployment import Wan2GPService

    baseline = _vram_mb()
    svc = Wan2GPService()

    yield svc

    # ── Teardown: unload + clear VRAM ──
    try:
        svc.unload()
    except Exception:
        pass

    del svc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Verify VRAM returned close to baseline (allow 100MB tolerance)
    after = _vram_mb()
    if after > baseline + 100:
        import warnings
        warnings.warn(
            f"Potential VRAM leak: baseline={baseline}MB, after={after}MB "
            f"(delta={after - baseline}MB)",
            stacklevel=1,
        )
