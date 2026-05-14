"""End-to-end tests for the unified model pipeline with REAL outputs.

Tests models that can run on the host machine (no Docker-only deps).
Each test loads a model, sends real input, and verifies the output is valid.

Usage: .venv/bin/python tests/test_e2e_real.py
"""
import base64
import io
import math
import struct
import sys
import time

sys.path.insert(0, "/home/user/Documents/programs/ray")

from services.wan2gp.deployment import Wan2GPService, discover_models


def make_wav(sample_rate: int = 16000, duration: float = 1.0, freq: int = 440) -> bytes:
    """Generate a minimal WAV file with a sine tone."""
    num_samples = int(sample_rate * duration)
    data = bytearray()
    for i in range(num_samples):
        sample = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
        data.extend(struct.pack("<h", max(-32768, min(32767, sample))))

    buf = bytearray()
    buf.extend(b"RIFF")
    buf.extend(struct.pack("<I", 36 + len(data)))
    buf.extend(b"WAVE")
    buf.extend(b"fmt ")
    buf.extend(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.extend(b"data")
    buf.extend(struct.pack("<I", len(data)))
    buf.extend(bytes(data))
    return bytes(buf)


def test_espeak():
    """eSpeak: real WAV audio from text."""
    svc = Wan2GPService()
    t0 = time.time()
    svc.load("espeak/espeak")
    load_s = time.time() - t0

    t1 = time.time()
    result = svc.infer({"model": "espeak/espeak", "text": "Hello world, testing the unified pipeline"})
    infer_s = time.time() - t1

    assert result["status"] == "success", f"espeak failed: {result}"
    audio = base64.b64decode(result["data"])
    assert audio[:4] == b"RIFF", "Not a WAV file"
    assert len(audio) > 1000, f"Audio too short: {len(audio)} bytes"

    svc.unload()
    print(f"  PASS: espeak ({load_s:.1f}s load, {infer_s:.1f}s infer, {len(audio)} bytes audio)")


def test_espeak_multilingual():
    """eSpeak: test different voices."""
    svc = Wan2GPService()
    svc.load("espeak/espeak")

    for voice, text in [("en", "Hello"), ("fr", "Bonjour"), ("de", "Hallo")]:
        result = svc.infer({"model": "espeak/espeak", "text": text, "voice": voice})
        assert result["status"] == "success", f"espeak {voice} failed"
        audio = base64.b64decode(result["data"])
        assert audio[:4] == b"RIFF"

    svc.unload()
    print(f"  PASS: espeak multilingual (en, fr, de)")


def test_faster_whisper():
    """Faster-Whisper: real ASR with model loading."""
    svc = Wan2GPService()
    t0 = time.time()
    svc.load("faster_whisper/faster_whisper")
    load_s = time.time() - t0

    wav_bytes = make_wav()
    audio_b64 = base64.b64encode(wav_bytes).decode()

    t1 = time.time()
    result = svc.infer({"model": "faster_whisper/faster_whisper", "audio_b64": audio_b64})
    infer_s = time.time() - t1

    assert result["status"] == "success", f"whisper failed: {result}"
    assert "language" in result, "Missing language field"
    assert "segments" in result, "Missing segments field"
    # Sine tone -> no speech, but should still detect language
    assert result["language_probability"] > 0.5, f"Low language confidence: {result['language_probability']}"

    svc.unload()
    print(f"  PASS: faster_whisper ({load_s:.1f}s load, {infer_s:.1f}s infer, lang={result['language']}@{result['language_probability']:.2f})")


def test_wan2gp_registry():
    """Verify the dynamic model registry discovers all expected handlers."""
    discovered = discover_models()

    # Check that all 12 custom family handlers are discovered
    expected_prefixes = [
        "espeak", "kokoro", "faster_whisper", "faster_qwen3_tts",
        "anigen", "trellis", "hy_motion", "moss", "see_through",
        "vibevoice_asr", "vibevoice_tts", "vnccs",
    ]

    available = [k for k, v in discovered.items() if not v.get("blocked")]
    print(f"  Registry: {len(discovered)} total models, {len(available)} available")

    for prefix in expected_prefixes:
        found = any(k.endswith(prefix) or prefix in k for k in discovered)
        assert found, f"Custom handler '{prefix}' not discovered"

    print(f"  PASS: All 12 custom handlers discovered")


def test_wan2gp_direct_espeak():
    """Direct Wan2GP: invoke espeak through the standalone service."""
    svc = Wan2GPService()
    result = svc.infer({"model": "espeak/espeak", "text": "Testing direct invocation"})

    assert result["status"] == "success", f"Direct invoke failed: {result}"
    audio = base64.b64decode(result["data"])
    assert audio[:4] == b"RIFF"
    print(f"  PASS: wan2gp → espeak ({len(audio)} bytes)")


def main():
    tests = [
        ("espeak", test_espeak),
        ("espeak_multilingual", test_espeak_multilingual),
        ("faster_whisper", test_faster_whisper),
        ("registry", test_wan2gp_registry),
        ("wan2gp_direct", test_wan2gp_direct_espeak),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"E2E: {name}")
        print(f"{'='*60}")
        try:
            test_fn()
            passed += 1
        except ImportError as e:
            print(f"  SKIP (missing dep): {e}")
            skipped += 1
        except FileNotFoundError as e:
            print(f"  SKIP (model not found): {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"E2E SUMMARY: {passed} passed  {failed} failed  {skipped} skipped")
    print(f"{'='*60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
