"""End-to-end inference tests for all Wan2GP models.

Exercises load → infer → unload for every model with appropriate payloads.
Run on the GPU server:  pytest tests/integration/test_wan2gp_e2e.py -m gpu -v
"""
from __future__ import annotations

import base64
import gc
import time

import pytest


def _svc():
    from services.wan2gp.deployment import Wan2GPService
    return Wan2GPService()


def _cleanup(svc):
    svc.unload()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_success(result, model):
    assert result["status"] in ("success", "ok"), (
        f"{model}: expected success/ok, got {result.get('status')}: {result.get('error', '')}"
    )


def _decode_b64(s):
    return base64.b64decode(s) if isinstance(s, str) else s


# ─── CPU Models (fast, no GPU) ────────────────────────────────────────────


class TestCPUModels:

    @pytest.mark.integration
    def test_espeak_e2e(self):
        svc = _svc()
        svc.load("espeak/espeak")
        result = svc.infer({"model": "espeak/espeak", "text": "Hello world"})
        _assert_success(result, "espeak")
        audio = _decode_b64(result["data"])
        assert audio[:4] == b"RIFF", f"espeak: expected WAV, got {audio[:4]}"
        assert len(audio) > 100
        _cleanup(svc)

    @pytest.mark.integration
    def test_kokoro_e2e(self):
        svc = _svc()
        svc.load("kokoro/kokoro")
        result = svc.infer({
            "model": "kokoro/kokoro",
            "text": "Hello world",
            "voice": "af_bella",
        })
        _assert_success(result, "kokoro")
        audio = _decode_b64(result["data"])
        assert audio[:4] == b"RIFF"
        assert len(audio) > 1000
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_faster_whisper_e2e(self, sample_wav_b64):
        svc = _svc()
        svc.load("faster_whisper/faster_whisper")
        result = svc.infer({
            "model": "faster_whisper/faster_whisper",
            "audio_b64": sample_wav_b64,
        })
        _assert_success(result, "faster_whisper")
        assert "segments" in result, "faster_whisper: missing segments"
        _cleanup(svc)


# ─── GPU Models (slow, sequential to avoid VRAM contention) ────────────────


class TestGPUVideoModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_wan_t2v_e2e(self):
        svc = _svc()
        svc.load("wan/t2v")
        result = svc.infer({
            "model": "wan/t2v",
            "prompt": "A cat walking in a garden",
            "frames": 3,
            "width": 256,
            "height": 256,
        })
        _assert_success(result, "wan/t2v")
        assert "data" in result, "wan/t2v: missing output data"
        data = _decode_b64(result["data"])
        assert len(data) > 1000, f"wan/t2v: output too small ({len(data)} bytes)"
        assert result.get("media_type") in ("video/mp4", "image/png")
        _cleanup(svc)


class TestGPUAudioModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_moss_soundeffect_e2e(self):
        svc = _svc()
        svc.load("moss/moss-soundeffect")
        result = svc.infer({
            "model": "moss/moss-soundeffect",
            "prompt": "gentle rain falling on leaves",
        })
        _assert_success(result, "moss-soundeffect")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_moss_tts_e2e(self, sample_wav_22k_b64):
        svc = _svc()
        svc.load("moss/moss-tts")
        result = svc.infer({
            "model": "moss/moss-tts",
            "text": "Hello world",
            "audio_b64": sample_wav_22k_b64,
        })
        _assert_success(result, "moss-tts")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_moss_voicegenerator_e2e(self):
        svc = _svc()
        svc.load("moss/moss-voicegenerator")
        result = svc.infer({
            "model": "moss/moss-voicegenerator",
            "instruction": "Say hello in a friendly voice",
        })
        _assert_success(result, "moss-voicegenerator")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_index_tts2_e2e(self, sample_wav_22k_b64):
        svc = _svc()
        svc.load("tts/index_tts2")
        result = svc.infer({
            "model": "tts/index_tts2",
            "text": "Hello world",
            "audio_b64": sample_wav_22k_b64,
        })
        _assert_success(result, "index_tts2")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_vibevoice_tts_e2e(self):
        svc = _svc()
        svc.load("vibevoice_tts/vibevoice-tts")
        result = svc.infer({
            "model": "vibevoice_tts/vibevoice-tts",
            "text": "Hello world",
        })
        _assert_success(result, "vibevoice_tts")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_vibevoice_asr_e2e(self, sample_wav_b64):
        svc = _svc()
        svc.load("vibevoice_asr/vibevoice-asr")
        result = svc.infer({
            "model": "vibevoice_asr/vibevoice-asr",
            "audio_b64": sample_wav_b64,
        })
        _assert_success(result, "vibevoice_asr")
        _cleanup(svc)


class TestGPUImageModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_see_through_e2e(self, sample_png_b64):
        svc = _svc()
        svc.load("see_through/see-through")
        result = svc.infer({
            "model": "see_through/see-through",
            "image_b64": sample_png_b64,
        })
        _assert_success(result, "see_through")
        _cleanup(svc)


class TestGPU3DModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_trellis_e2e(self, sample_png_b64):
        svc = _svc()
        svc.load("trellis/trellis")
        result = svc.infer({
            "model": "trellis/trellis",
            "image_b64": sample_png_b64,
            "steps": 1,
        })
        _assert_success(result, "trellis")
        _cleanup(svc)

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_anigen_e2e(self, sample_png_b64):
        svc = _svc()
        svc.load("anigen/anigen")
        result = svc.infer({
            "model": "anigen/anigen",
            "image_b64": sample_png_b64,
        })
        _assert_success(result, "anigen")
        _cleanup(svc)


class TestGPUMotionModels:

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_hy_motion_lite_e2e(self):
        svc = _svc()
        svc.load("hy_motion/hy-motion-1.0-lite")
        result = svc.infer({
            "model": "hy_motion/hy-motion-1.0-lite",
            "prompt": "A person waving hello",
        })
        _assert_success(result, "hy_motion_lite")
        _cleanup(svc)
