"""End-to-end inference tests for all Wan2GP models.

Exercises load → infer → unload for every model with appropriate payloads.
Each test gets its own service instance with automatic VRAM cleanup.

Run on the GPU server (via kubectl exec or directly):
    pytest tests/integration/test_wan2gp_e2e.py -v

Skip already-passing tests:
    pytest tests/integration/test_wan2gp_e2e.py -v -k "not espeak"

Skip autoregressive models (>10min inference):
    pytest tests/integration/test_wan2gp_e2e.py -v -m "not autoregressive"

Run only fast CPU tests:
    pytest tests/integration/test_wan2gp_e2e.py -v -m cpu

Markers:
    cpu           — CPU-only, fast, no GPU needed
    gpu           — requires CUDA GPU
    slow          — >30s total (load + infer)
    autoregressive — >10min inference (moss voicegenerator/soundeffect)
"""
from __future__ import annotations

import base64
import gc

import pytest


def _svc():
    from services.wan2gp.deployment import Wan2GPService
    return Wan2GPService()


def _assert_success(result, model):
    assert result["status"] in ("success", "ok"), (
        f"{model}: expected success/ok, got {result.get('status')}: "
        f"{result.get('error', '')}"
    )


def _has_audio(result, min_bytes=100):
    data = result.get("data")
    if not data:
        return False
    raw = base64.b64decode(data) if isinstance(data, str) else data
    return len(raw) > min_bytes


# ─── CPU Models ────────────────────────────────────────────────────────────


class TestCPUModels:
    """Fast CPU-only models. No GPU needed."""

    @pytest.mark.cpu
    def test_espeak_e2e(self, wan2gp_svc):
        wan2gp_svc.load("espeak/espeak")
        result = wan2gp_svc.infer({"model": "espeak/espeak", "text": "Hello world"})
        _assert_success(result, "espeak")
        assert _has_audio(result)
        audio = base64.b64decode(result["data"])
        assert audio[:4] == b"RIFF"

    @pytest.mark.cpu
    def test_kokoro_e2e(self, wan2gp_svc):
        wan2gp_svc.load("kokoro/kokoro")
        result = wan2gp_svc.infer({
            "model": "kokoro/kokoro",
            "text": "Hello world",
            "voice": "af_bella",
        })
        _assert_success(result, "kokoro")
        assert _has_audio(result, min_bytes=1000)
        audio = base64.b64decode(result["data"])
        assert audio[:4] == b"RIFF"

    @pytest.mark.cpu
    def test_faster_whisper_e2e(self, wan2gp_svc, sample_wav_b64):
        wan2gp_svc.load("faster_whisper/faster_whisper")
        result = wan2gp_svc.infer({
            "model": "faster_whisper/faster_whisper",
            "audio_b64": sample_wav_b64,
        })
        _assert_success(result, "faster_whisper")


# ─── GPU Video ─────────────────────────────────────────────────────────────


class TestGPUVideoModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_wan_t2v_e2e(self, wan2gp_svc):
        wan2gp_svc.load("wan/t2v")
        result = wan2gp_svc.infer({
            "model": "wan/t2v",
            "prompt": "A cat walking in a garden",
            "frames": 3,
            "width": 256,
            "height": 256,
        })
        _assert_success(result, "wan/t2v")
        assert "data" in result
        data = base64.b64decode(result["data"])
        assert len(data) > 1000
        assert result.get("media_type") in ("video/mp4", "image/png")


# ─── GPU TTS ───────────────────────────────────────────────────────────────


class TestGPUTTSModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_index_tts2_e2e(self, wan2gp_svc, sample_wav_22k_b64):
        wan2gp_svc.load("tts/index_tts2")
        result = wan2gp_svc.infer({
            "model": "tts/index_tts2",
            "text": "Hello world",
            "audio_b64": sample_wav_22k_b64,
        })
        _assert_success(result, "index_tts2")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_vibevoice_tts_e2e(self, wan2gp_svc):
        wan2gp_svc.load("vibevoice_tts/vibevoice-tts")
        result = wan2gp_svc.infer({
            "model": "vibevoice_tts/vibevoice-tts",
            "text": "Hello world",
        })
        _assert_success(result, "vibevoice_tts")


# ─── GPU ASR ───────────────────────────────────────────────────────────────


class TestGPUASRModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_vibevoice_asr_e2e(self, wan2gp_svc, sample_wav_b64):
        wan2gp_svc.load("vibevoice_asr/vibevoice-asr")
        result = wan2gp_svc.infer({
            "model": "vibevoice_asr/vibevoice-asr",
            "audio_b64": sample_wav_b64,
        })
        _assert_success(result, "vibevoice_asr")


# ─── GPU 3D ────────────────────────────────────────────────────────────────


class TestGPU3DModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_trellis_e2e(self, wan2gp_svc, sample_png_b64):
        wan2gp_svc.load("trellis/trellis")
        result = wan2gp_svc.infer({
            "model": "trellis/trellis",
            "image_b64": sample_png_b64,
            "steps": 1,
        })
        _assert_success(result, "trellis")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_anigen_e2e(self, wan2gp_svc, sample_png_b64):
        wan2gp_svc.load("anigen/anigen")
        result = wan2gp_svc.infer({
            "model": "anigen/anigen",
            "image_b64": sample_png_b64,
        })
        _assert_success(result, "anigen")


# ─── GPU Image ─────────────────────────────────────────────────────────────


class TestGPUImageModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_see_through_e2e(self, wan2gp_svc, sample_png_b64):
        wan2gp_svc.load("see_through/see-through")
        result = wan2gp_svc.infer({
            "model": "see_through/see-through",
            "image_b64": sample_png_b64,
        })
        _assert_success(result, "see_through")


# ─── GPU Motion ────────────────────────────────────────────────────────────


class TestGPUMotionModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_hy_motion_lite_e2e(self, wan2gp_svc):
        wan2gp_svc.load("hy_motion/hy-motion-1.0-lite")
        result = wan2gp_svc.infer({
            "model": "hy_motion/hy-motion-1.0-lite",
            "prompt": "A person waving hello",
        })
        _assert_success(result, "hy_motion_lite")


# ─── GPU Autoregressive (>10min inference) ──────────────────────────────────


class TestGPUAutoregressiveModels:
    """These models generate 4096 tokens at ~2.5s/token = ~170min.
    Only run with:  pytest -m autoregressive
    """

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.autoregressive
    def test_moss_voicegenerator_e2e(self, wan2gp_svc):
        wan2gp_svc.load("moss/moss-voicegenerator")
        result = wan2gp_svc.infer({
            "model": "moss/moss-voicegenerator",
            "instruction": "Say hello in a friendly voice",
        })
        _assert_success(result, "moss-voicegenerator")

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.autoregressive
    def test_moss_soundeffect_e2e(self, wan2gp_svc):
        wan2gp_svc.load("moss/moss-soundeffect")
        result = wan2gp_svc.infer({
            "model": "moss/moss-soundeffect",
            "prompt": "gentle rain",
        })
        _assert_success(result, "moss-soundeffect")

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.autoregressive
    def test_moss_tts_e2e(self, wan2gp_svc, sample_wav_22k_b64):
        wan2gp_svc.load("moss/moss-tts")
        result = wan2gp_svc.infer({
            "model": "moss/moss-tts",
            "text": "Hello world",
            "audio_b64": sample_wav_22k_b64,
        })
        _assert_success(result, "moss-tts")
