"""Integration test fixtures — VRAM-aware service lifecycle.

Provides `wan2gp_svc` fixture that:
  - Creates a fresh Wan2GPService per test
  - Unloads model + clears VRAM after each test (even on failure)
  - Verifies VRAM returns to baseline (detects leaks)

Also registers the `autoregressive` marker for skip-by-marker.

Includes binary data fixtures (sample_wav_b64, sample_png_b64) so tests
can run standalone without the root conftest.py.
"""
from __future__ import annotations

import base64
import gc
import io
import math
import struct
import zlib

import pytest


# ─── Marker registration ──────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "autoregressive: >10min inference, skip by default")
    config.addinivalue_line("markers", "cpu: CPU-only model, no GPU needed")
    config.addinivalue_line("markers", "gpu: needs CUDA GPU with drivers")
    config.addinivalue_line("markers", "slow: takes >5 seconds")


def pytest_collection_modifyitems(config, items):
    """Auto-skip autoregressive tests unless explicitly selected with -m."""
    expr = config.getoption("-m", default="")
    if "autoregressive" not in expr:
        skip_auto = pytest.mark.skip(reason="autoregressive — run with -m autoregressive")
        for item in items:
            if "autoregressive" in item.keywords:
                item.add_marker(skip_auto)


# ─── Binary test data generators ──────────────────────────────────────────


def _make_png(width: int = 64, height: int = 64) -> bytes:
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            raw_rows.extend([100, 150, 200])
    idat = _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _make_wav(duration_s: float = 0.5, sample_rate: int = 16000,
              freq: int = 440) -> bytes:
    num_samples = int(sample_rate * duration_s)
    data = bytearray()
    for i in range(num_samples):
        sample = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
        data.extend(struct.pack("<h", max(-32768, min(32767, sample))))
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                          sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()


# ─── Binary data fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    return _make_wav()


@pytest.fixture(scope="session")
def sample_wav_bytes_22k() -> bytes:
    return _make_wav(duration_s=1.0, sample_rate=22050, freq=440)


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    return _make_png()


@pytest.fixture(scope="session")
def sample_wav_b64(sample_wav_bytes) -> str:
    return base64.b64encode(sample_wav_bytes).decode()


@pytest.fixture(scope="session")
def sample_wav_22k_b64(sample_wav_bytes_22k) -> str:
    return base64.b64encode(sample_wav_bytes_22k).decode()


@pytest.fixture(scope="session")
def sample_png_b64(sample_png_bytes) -> str:
    return base64.b64encode(sample_png_bytes).decode()


# ─── Service lifecycle fixture ────────────────────────────────────────────


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

    # Verify VRAM returned close to baseline (allow 200MB for mmgp overhead)
    after = _vram_mb()
    if after > baseline + 200:
        import warnings
        warnings.warn(
            f"Potential VRAM leak: baseline={baseline}MB, after={after}MB "
            f"(delta={after - baseline}MB)",
            stacklevel=1,
        )
