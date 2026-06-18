"""Test ALL models through the Forge."""

import ray
import asyncio
import json
import struct
import math
import base64
import io
import sys
import time

sys.path.insert(0, "/app")

def make_sine_wav(duration=1.0, sample_rate=22050, freq=440):
    n = int(sample_rate * duration)
    samples = [int(32767 * 0.3 * math.sin(2 * math.pi * freq * t / sample_rate)) for t in range(n)]
    buf = b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt "
    buf += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    buf += b"data" + struct.pack("<I", n * 2)
    buf += struct.pack("<" + "h" * n, *samples)
    return base64.b64encode(buf).decode()

def make_test_image(w=512, h=512):
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (w, h), (64, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

audio_b64 = make_sine_wav()
image_b64 = make_test_image()

def result_str(r):
    if isinstance(r, dict):
        s = r.get("status", "?")
        m = r.get("media_type", "")
        e = r.get("error", "")
        dl = len(r.get("data", ""))
        if s == "ok":
            return f"✅ {m} ({dl}B)"
        else:
            return f"❌ {e[:120]}"
    return f"❌ {str(r)[:120]}"

async def test(handle, label, payload, timeout_sec=600):
    sys.stdout.write(f"\n  {label:40s} ")
    sys.stdout.flush()
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            handle.invoke.remote("wan2gp", payload, model=payload.get("model", "")),
            timeout=timeout_sec,
        )
        elapsed = time.time() - t0
        sys.stdout.write(f"[{elapsed:5.1f}s] {result_str(result)}\n")
        sys.stdout.flush()
        return result
    except asyncio.TimeoutError:
        sys.stdout.write(f"[TIMEOUT>{timeout_sec}s]\n")
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        sys.stdout.write(f"[CRASH] {str(e)[:80]}\n")
        sys.stdout.flush()
        return {"status": "error", "error": str(e)}

async def main():
    ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
    from ray.serve import get_deployment_handle
    handle = get_deployment_handle("forge", "forge")

    print(f"\n{'='*70}")
    print("  TESTING ALL MODELS THROUGH FORGE")
    print(f"{'='*70}")
    print(f"  {time.ctime()}")
    print(f"{'='*70}")

    results = {}

    tests = [
        # ── IMMEDIATE: CPU-only models (kokoro) ──
        ("kokoro", {
            "model": "kokoro",
            "text": "The quick brown fox jumps over the lazy dog.",
            "voice": "af_heart",
        }, 60),

        # ── FAST: native Wan2GP image (Klien) ──
        ("flux2_klein_4b", {
            "model": "flux2_klein_4b",
            "prompt": "a serene mountain landscape at sunset, highly detailed",
            "steps": 4,
            "seed": 42,
            "embedded_guidance_scale": 1.0,
        }, 120),

        # ── FAST: native Wan2GP image (flux2) ──
        ("flux2_dev", {
            "model": "flux2_dev",
            "prompt": "a cat wearing a spacesuit, digital art",
            "steps": 4,
            "seed": 42,
        }, 120),

        # faster-qwen3-tts tests removed — engine dropped
        # (MOSS VoiceGenerator + sherpa-onnx Kokoro replace it)

        # ── CUSTOM HANDLER: ACE-Step music gen ──
        ("ace_step_v1_5", {
            "model": "ace_step",
            "prompt": "A calm piano melody with gentle strings",
            "duration_seconds": 8.0,
            "steps": 30,
            "seed": 42,
        }, 300),

        # ── CUSTOM HANDLER: index_tts2 (already tested, quick verify) ──
        ("index_tts2", {
            "model": "index_tts2",
            "text": "Hello, this is a test.",
            "audio_b64": audio_b64,
        }, 180),

        # ── CUSTOM HANDLER: See-Through image editing ──
        ("see-through", {
            "model": "see-through",
            "prompt": "a cat sitting on a chair",
            "image_b64": image_b64,
        }, 120),

        # ── NATIVE: wan t2v_1.3B (small video) ──
        ("t2v_1.3B", {
            "model": "t2v_1.3B",
            "prompt": "a cat walking on a beach",
            "steps": 20,
            "seed": 42,
        }, 300),

        # ── NATIVE: t2v (14B, larger) ──
        ("t2v_14B", {
            "model": "t2v",
            "prompt": "a cat walking on a beach",
            "seed": 42,
            "frame_num": 41,
        }, 300),

        # ── CUSTOM HANDLER: Trellis 3D ──
        ("trellis", {
            "model": "trellis",
            "image_b64": image_b64,
            "prompt": "A colorful 3D object",
        }, 300),

        # ── CUSTOM HANDLER: AniGen 3D char ──
        ("anigen", {
            "model": "anigen",
            "prompt": "A male warrior character, fantasy style",
        }, 300),

        # ── CUSTOM HANDLER: Pixal3D ──
        ("pixal3d", {
            "model": "pixal3d",
            "image_b64": image_b64,
        }, 300),

        # ── CUSTOM HANDLER: HyMotion ──
        ("hy-motion-1.0-lite", {
            "model": "hy-motion-1.0-lite",
            "prompt": "A person waving their hand",
        }, 300),

        # ── NATIVE: flux (full size) ──
        ("flux", {
            "model": "flux",
            "prompt": "a beautiful sunset over mountains",
            "steps": 30,
            "seed": 42,
        }, 300),
    ]

    for label, payload, timeout in tests:
        r = await test(handle, label, payload, timeout)
        results[label] = r

    print(f"\n\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    ok = 0
    fail = 0
    for label, r in results.items():
        icon = "✅" if r.get("status") == "ok" else "❌"
        if r.get("status") == "ok":
            ok += 1
        else:
            fail += 1
        print(f"  {icon} {label:40s} {r.get('media_type','')} / {r.get('error','ok')[:60]}")
    print(f"\n  Pass: {ok}  Fail: {fail}")
    print(f"{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
