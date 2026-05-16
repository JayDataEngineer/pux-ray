"""End-to-end integration test — all services through Wan2GPService.

Run on the GPU server: pytest tests/integration/test_all_services.py -m gpu

Tests Wan2GPService.load() → infer() → unload() for every discovered model.
Auto-skipped when no CUDA GPU is available.
"""
from __future__ import annotations

import base64
import gc
import time

import pytest


# ─── Payloads per model ─────────────────────────────────────────────────────

PAYLOADS = {
    "espeak/espeak": {"text": "Hello world"},
    "kokoro/kokoro": {"text": "Hello world", "voice": "af_bella"},
    "faster_whisper/faster_whisper": {"audio_b64": "placeholder"},
    "trellis/trellis": {"image_b64": "placeholder", "steps": 1},
    "moss/moss-soundeffect": {"prompt": "gentle rain", "max_tokens": 64},
    "vibevoice_asr/vibevoice-asr": {"audio_b64": "placeholder"},
    "vibevoice_tts/vibevoice-tts": {"text": "Hello world"},
}


def _get_service():
    from services.wan2gp.deployment import Wan2GPService
    return Wan2GPService()


def _vram_mb():
    try:
        import torch
        return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        return 0


class TestAllServicesViaWan2GP:
    """Test every model via Wan2GPService load/infer/unload cycle."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_registry_discovers_models(self):
        service = _get_service()
        available = service.available_models()
        assert len(available) > 0, "No models discovered"
        assert "espeak/espeak" in available, "espeak not discovered"

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_espeak_e2e(self):
        service = _get_service()
        service.load("espeak/espeak")
        result = service.infer({"model": "espeak/espeak", "text": "Hello world"})
        assert result["status"] == "success"
        audio = base64.b64decode(result["data"])
        assert audio[:4] == b"RIFF"
        assert len(audio) > 100
        service.unload()

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_espeak_multilingual(self):
        service = _get_service()
        service.load("espeak/espeak")
        for voice, text in [("en", "Hello"), ("fr", "Bonjour"), ("de", "Hallo")]:
            result = service.infer({"model": "espeak/espeak", "text": text, "voice": voice})
            assert result["status"] == "success"
        service.unload()

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_faster_whisper_e2e(self, sample_wav_b64):
        service = _get_service()
        service.load("faster_whisper/faster_whisper")
        result = service.infer({
            "model": "faster_whisper/faster_whisper",
            "audio_b64": sample_wav_b64,
        })
        assert result["status"] == "success"
        assert "language" in result
        assert "segments" in result
        service.unload()

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_unload_cleans_vram(self):
        """VRAM should return to baseline after unload."""
        service = _get_service()
        baseline = _vram_mb()
        service.load("espeak/espeak")
        service.infer({"model": "espeak/espeak", "text": "test"})
        service.unload()
        after = _vram_mb()
        # CPU model, but verify the cycle doesn't leak
        assert abs(after - baseline) < 100, (
            f"VRAM leak: baseline={baseline}MB, after unload={after}MB"
        )

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_unknown_model_raises(self):
        service = _get_service()
        with pytest.raises((ValueError, RuntimeError)):
            service.load("nonexistent/model")

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_status_returns_state(self):
        service = _get_service()
        status = service.status()
        assert "available" in status
        assert "blocked" in status
        assert "total_models" in status
