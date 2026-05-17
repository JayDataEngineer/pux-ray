"""End-to-end tests — all models load → generate → validate output.

Run on the GPU server (or inside Docker with GPU):
  pytest tests/test_e2e_all.py -m gpu -v
  pytest tests/test_e2e_all.py -m gpu --timeout=600   # longer timeout for model loading

Auto-skipped when no CUDA GPU is available (see conftest.py).
"""
from __future__ import annotations

import base64
import gc
import io
import struct
import time

import pytest


def _vram_mb():
    try:
        import torch
        return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        return 0


def _get_service():
    from services.wan2gp.deployment import Wan2GPService
    return Wan2GPService()


def _is_valid_wav(data: bytes) -> bool:
    """Check if bytes are a valid WAV file."""
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _wav_has_audio(data: bytes) -> bool:
    """Check if WAV file has non-silent audio content."""
    if len(data) < 44:
        return False
    # Skip WAV header (44 bytes), check first 16000 samples for non-zero values
    sample_end = min(len(data), 44 + 32000)
    samples = struct.unpack(f'<{min(16000, (sample_end - 44) // 2)}h', data[44:sample_end])
    max_val = max(abs(s) for s in samples)
    return max_val > 10


# ── eSpeak TTS (CPU) ────────────────────────────────────────────────────


class TestEspeakE2E:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_espeak_load_and_generate(self):
        svc = _get_service()
        svc.load("espeak/espeak")

        result = svc.infer({"model": "espeak/espeak", "text": "Hello world"})
        assert result["status"] == "success", f"Expected success, got: {result}"

        audio = base64.b64decode(result["data"])
        assert _is_valid_wav(audio), "Output is not a valid WAV"
        assert len(audio) > 100, "WAV too small"
        assert _wav_has_audio(audio), "WAV appears to be silence"

        svc.unload()

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_espeak_multilingual(self):
        svc = _get_service()
        svc.load("espeak/espeak")

        for voice, text in [("en", "Hello"), ("fr", "Bonjour"), ("de", "Hallo")]:
            result = svc.infer({"model": "espeak/espeak", "text": text, "voice": voice})
            assert result["status"] == "success"
            audio = base64.b64decode(result["data"])
            assert _is_valid_wav(audio)

        svc.unload()


# ── Kokoro TTS (CPU) ────────────────────────────────────────────────────


class TestKokoroE2E:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_kokoro_load_and_generate(self):
        svc = _get_service()
        svc.load("kokoro/kokoro")

        result = svc.infer({"model": "kokoro/kokoro", "text": "Hello world", "voice": "af_bella"})
        assert result["status"] == "success", f"Expected success, got: {result}"

        audio = base64.b64decode(result["data"])
        assert _is_valid_wav(audio), "Output is not a valid WAV"
        assert len(audio) > 100, "WAV too small"
        assert _wav_has_audio(audio), "WAV appears to be silence"

        svc.unload()


# ── Faster-Whisper ASR (CPU) ────────────────────────────────────────────


class TestFasterWhisperE2E:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_faster_whisper_transcribes_wav(self, sample_wav_b64):
        svc = _get_service()
        svc.load("faster_whisper/faster_whisper")

        result = svc.infer({
            "model": "faster_whisper/faster_whisper",
            "audio_b64": sample_wav_b64,
        })
        assert result["status"] == "success", f"Expected success, got: {result}"
        assert "text" in result or "segments" in result, "Missing transcription output"

        svc.unload()


# ── MOSS SoundEffect (GPU) ──────────────────────────────────────────────


class TestMOSSSoundeffectE2E:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_moss_load_and_generate(self):
        svc = _get_service()
        t0 = time.time()
        svc.load("moss/moss-soundeffect")
        load_time = time.time() - t0

        result = svc.infer({"model": "moss/moss-soundeffect", "prompt": "gentle rain"})
        gen_time = time.time() - t0 - load_time

        assert result["status"] == "success", f"Expected success, got: {result}"

        audio = base64.b64decode(result["data"])
        assert _is_valid_wav(audio), "Output is not a valid WAV"
        assert len(audio) > 1000, f"WAV too small ({len(audio)} bytes)"
        assert _wav_has_audio(audio), "WAV appears to be silence"

        # VRAM check: MOSS should stay under 20GB
        vram = _vram_mb()
        assert vram < 20000, f"VRAM usage too high: {vram}MB"

        svc.unload()


# ── TRELLIS 3D (GPU) ───────────────────────────────────────────────────


class TestTRELLISE2E:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_trellis_load_and_generate(self, sample_png_b64):
        svc = _get_service()
        t0 = time.time()
        svc.load("trellis/trellis")
        load_time = time.time() - t0

        result = svc.infer({
            "model": "trellis/trellis",
            "image_b64": sample_png_b64,
            "steps": 1,
        })
        gen_time = time.time() - t0 - load_time

        assert result["status"] == "success", f"Expected success, got: {result}"
        # TRELLIS outputs 3D model data (GLB or SPZ bytes)
        assert "data" in result, "Missing data in output"

        output = base64.b64decode(result["data"]) if result.get("data") else b""
        assert len(output) > 100, f"Output too small ({len(output)} bytes)"

        svc.unload()


# ── ACE-Step Music (GPU) ────────────────────────────────────────────────


class TestACEStepE2E:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_ace_step_load_and_generate(self):
        svc = _get_service()
        t0 = time.time()
        svc.load("ace_step/ace_step")
        load_time = time.time() - t0

        result = svc.infer({
            "model": "ace_step/ace_step",
            "prompt": "gentle piano ambient",
            "duration_seconds": 5,
        })
        gen_time = time.time() - t0 - load_time

        assert result["status"] == "success", f"Expected success, got: {result}"

        audio = base64.b64decode(result["data"])
        assert _is_valid_wav(audio), "Output is not a valid WAV"
        assert len(audio) > 1000, f"WAV too small ({len(audio)} bytes)"

        svc.unload()


# ── Pipeline Execution E2E ──────────────────────────────────────────────


class TestPipelineE2E:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_single_step_pipeline(self):
        """Pipeline executor runs a single service step."""
        from gateway.pipeline import PipelineSpec, execute_pipeline

        spec = PipelineSpec.from_dict({"steps": [
            {"name": "sfx", "service": "moss_soundeffect", "params": {"prompt": "rain"}},
        ]})

        async def dispatch(service, params):
            if service == "moss_soundeffect":
                return {"output": {"content": "fake_audio"}}
            raise ValueError(f"Unknown service: {service}")

        import asyncio
        events = asyncio.get_event_loop().run_until_complete(
            execute_pipeline(spec, dispatch)
        )

        event_types = [e["event"] for e in events]
        assert "pipeline_started" in event_types
        assert "step_completed" in event_types
        assert "pipeline_completed" in event_types

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_chained_pipeline(self):
        """Pipeline executor chains outputs between steps."""
        from gateway.pipeline import PipelineSpec, execute_pipeline

        spec = PipelineSpec.from_dict({"steps": [
            {"name": "gen", "service": "ace_step", "params": {"prompt": "piano"}},
            {"name": "recognize", "service": "faster_whisper",
             "depends_on": ["gen"], "params": {"audio_b64": "{gen.output.content}"}},
        ]})

        call_log = []

        async def dispatch(service, params):
            call_log.append({"service": service, "params": dict(params)})
            return {"output": {"content": f"{service}_data"}}

        import asyncio
        events = asyncio.get_event_loop().run_until_complete(
            execute_pipeline(spec, dispatch)
        )

        # Second call should have resolved reference
        assert call_log[1]["params"]["audio_b64"] == "ace_step_data"

        # Check final results
        completed = [e for e in events if e["event"] == "pipeline_completed"]
        assert len(completed) == 1
        assert "gen" in completed[0]["results"]
        assert "recognize" in completed[0]["results"]


# ── VRAM Lifecycle ──────────────────────────────────────────────────────


class TestVRAMLifecycle:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_unload_cleans_vram(self):
        """VRAM should return to near baseline after unload."""
        svc = _get_service()
        baseline = _vram_mb()

        svc.load("espeak/espeak")
        svc.infer({"model": "espeak/espeak", "text": "test"})
        svc.unload()

        after = _vram_mb()
        # espeak is CPU-only, so VRAM delta should be minimal
        assert abs(after - baseline) < 100, (
            f"VRAM leak: baseline={baseline}MB, after unload={after}MB"
        )

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.integration
    def test_gpu_model_load_unload_cycle(self):
        """GPU model loads, generates, unloads cleanly."""
        svc = _get_service()
        baseline = _vram_mb()

        svc.load("moss/moss-soundeffect")
        peak = _vram_mb()
        assert peak > baseline, "GPU model didn't allocate VRAM"

        svc.infer({"model": "moss/moss-soundeffect", "prompt": "beep"})
        svc.unload()

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        after = _vram_mb()
        # Allow some fragmentation overhead
        assert abs(after - baseline) < 500, (
            f"VRAM not freed: baseline={baseline}MB, after unload={after}MB, peak={peak}MB"
        )


# ── Service Discovery ──────────────────────────────────────────────────


class TestServiceDiscovery:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_discovers_expected_models(self):
        svc = _get_service()
        available = svc.available_models()

        # These should always be discoverable (CPU models or models with weights)
        expected = ["espeak/espeak", "kokoro/kokoro", "faster_whisper/faster_whisper"]
        for model in expected:
            assert model in available, f"Missing expected model: {model}"

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_status_returns_gpu_info(self):
        svc = _get_service()
        status = svc.status()

        assert "available" in status
        assert "blocked" in status
        assert "total_models" in status
        assert "gpu" in status

        if status["gpu"]:
            assert "total_mb" in status["gpu"]
            assert status["gpu"]["total_mb"] > 0

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_unknown_model_raises(self):
        svc = _get_service()
        with pytest.raises((ValueError, RuntimeError)):
            svc.load("nonexistent/model")


# ── Ingress API (unit-level, no cluster needed) ─────────────────────────


class TestIngressPipelineRoute:
    """Test the pipeline route exists and validates input."""

    def test_pipeline_spec_rejects_empty(self):
        from gateway.pipeline import PipelineSpec
        with pytest.raises(ValueError, match="at least one step"):
            PipelineSpec.from_dict({"steps": []})

    def test_pipeline_spec_rejects_bad_step(self):
        from gateway.pipeline import PipelineSpec
        with pytest.raises(ValueError, match="service"):
            PipelineSpec.from_dict({"steps": [{"name": "x"}]})

    def test_pipeline_spec_accepts_valid(self):
        from gateway.pipeline import PipelineSpec
        spec = PipelineSpec.from_dict({"steps": [
            {"name": "gen", "service": "kokoro", "params": {"text": "hi"}},
        ]})
        assert len(spec.steps) == 1
