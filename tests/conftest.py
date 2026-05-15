"""Shared test fixtures for Tech Noir Ray unit tests.

Unit tests run WITHOUT a Ray cluster. All Ray interactions are mocked
at the session level so no test hangs on cluster connection.

Integration tests (in tests/integration/) use the ray_cluster fixture
and require a live GPU server.
"""
from __future__ import annotations

import base64
import io
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Marker Registration ────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests, no GPU/network needed")
    config.addinivalue_line("markers", "integration: needs live Ray cluster")
    config.addinivalue_line("markers", "gpu: needs CUDA GPU with drivers")
    config.addinivalue_line("markers", "slow: takes >5 seconds")
    config.addinivalue_line("markers", "handler: tests a specific family_handler")


# ─── Session-scoped Ray mocking ────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="session")
def _mock_ray_core():
    """Mock Ray globally so no unit test hangs on cluster connection."""
    mock_ray = MagicMock()
    mock_ray.get_actor.side_effect = ValueError("no cluster")
    mock_ray.available_resources.return_value = {"GPU": 1, "CPU": 8}
    mock_ray.get.return_value = {}
    mock_ray.is_initialized.return_value = False

    patches = [
        patch("ray.get_actor", mock_ray.get_actor),
        patch("ray.get", mock_ray.get),
        patch("ray.available_resources", mock_ray.available_resources),
        patch("ray.is_initialized", mock_ray.is_initialized),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# ─── Handler sys.path setup ────────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="session")
def _setup_custom_models_path():
    """Add custom_models/ and vendor/ to sys.path for handler imports."""
    project_root = Path(__file__).resolve().parent.parent
    custom_models = project_root / "services" / "wan2gp" / "custom_models"
    vendor_wan2gp = project_root / "vendor" / "wan2gp"

    for p in [str(custom_models), str(vendor_wan2gp)]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("WAN2GP_ROOT", str(vendor_wan2gp))
    yield


# ─── Binary test data generators ────────────────────────────────────────────


def _make_png(width: int = 64, height: int = 64) -> bytes:
    """Generate a minimal valid PNG image."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            raw_rows.extend([100, 150, 200])
    compressed = zlib.compress(bytes(raw_rows))
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _make_wav(duration_s: float = 0.5, sample_rate: int = 16000,
              freq: int = 440) -> bytes:
    """Generate a minimal valid WAV file with a sine tone."""
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


# ─── Binary data fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    return _make_wav()


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    return _make_png()


@pytest.fixture(scope="session")
def sample_wav_b64(sample_wav_bytes) -> str:
    return base64.b64encode(sample_wav_bytes).decode()


@pytest.fixture(scope="session")
def sample_png_b64(sample_png_bytes) -> str:
    return base64.b64encode(sample_png_bytes).decode()


# ─── Handler import fixture ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def custom_handler_paths() -> list[str]:
    """All custom family_handler import paths from CUSTOM_HANDLERS."""
    return [
        "trellis.trellis_handler",
        "anigen_handler.anigen_handler",
        "see_through.see_through_handler",
        "hy_motion.hy_motion_handler",
        "kokoro.kokoro_handler",
        "moss.moss_handler",
        "espeak.espeak_handler",
        "faster_whisper.faster_whisper_handler",
        "vibevoice_asr.vibevoice_asr_handler",
        "vibevoice_tts.vibevoice_tts_handler",
        "faster_qwen3_tts.faster_qwen3_tts_handler",
    ]


# ─── Integration fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ray_cluster():
    """Ensure Ray cluster is running for the test session."""
    try:
        result = subprocess.run(
            ["ray", "status"], capture_output=True, text=True, timeout=5,
        )
        if "node" not in result.stdout:
            subprocess.run(
                ["bash", "scripts/start_cluster.sh"],
                check=True, timeout=30,
            )
    except Exception:
        pass
    yield


@pytest.fixture
def free_vram_mb() -> int:
    """Get current free VRAM in MB via torch.cuda."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            return int(total - reserved)
    except Exception:
        pass
    return 0


def pytest_collection_modifyitems(items):
    """Auto-skip gpu-marked tests when CUDA is unavailable."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
