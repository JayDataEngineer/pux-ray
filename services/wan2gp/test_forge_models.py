"""Forge service test suite — validates every Forge model end-to-end.

Usage:
    kubectl exec -n ai-services <head-pod> -- python3 /app/services/wan2gp/test_forge_models.py
    kubectl exec -n ai-services <worker-pod> -c ray-worker -- python3 /app/services/wan2gp/test_forge_models.py
    kubectl exec -n ai-services <worker-pod> -c ray-worker -- python3 /tmp/test_forge_models.py --only lance avatar

All test data (images, audio) is generated in-process — no external files needed.
"""
from __future__ import annotations

import base64
import io
import math
import struct
import sys
import time

import requests

FORGE = "http://localhost:8000/forge"

# ─── Test Data Generators ────────────────────────────────────────────────────


def make_wav(duration: float = 1.0, sr: int = 22050, freq: int = 440) -> str:
    n = int(sr * duration)
    samples = [int(32767 * 0.3 * math.sin(2 * math.pi * freq * t / sr)) for t in range(n)]
    buf = b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt "
    buf += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    buf += b"data" + struct.pack("<I", n * 2)
    buf += struct.pack("<" + "h" * n, *samples)
    return base64.b64encode(buf).decode()


def make_png(w: int = 512, h: int = 512) -> str:
    """Generate a test image with shapes and gradients — not a solid color.

    Detection/segmentation models (DINOv3, BiRefNet, Marigold) need real
    visual structure to produce non-empty outputs. Solid-color images cause
    empty detections which crash downstream (TRELLIS max() on empty tensor,
    OpenCV resize on empty dsize).
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (220, 210, 200))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        r = int(180 + 60 * y / h)
        g = int(200 - 80 * y / h)
        b = int(160 + 40 * y / h)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    draw.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(180, 60, 60), outline=(100, 30, 30))
    draw.rectangle([w // 6, h // 6, w // 3, h // 3], fill=(60, 180, 60))
    draw.ellipse([2 * w // 3, 2 * h // 3, 5 * w // 6, 5 * h // 6], fill=(60, 60, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_anime_png(w: int = 1280, h: int = 1280) -> str:
    """Generate an anime-style test image for layer-decomposition models."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (w, h), (255, 240, 245, 255))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        r = int(200 + 40 * y / h)
        g = int(180 + 30 * (1 - y / h))
        b = int(220 - 60 * y / h)
        draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - 120, cy - 300, cx + 120, cy - 60], fill=(255, 220, 200, 255), outline=(60, 40, 30, 255))
    draw.rectangle([cx - 100, cy - 80, cx + 100, cy + 200], fill=(180, 100, 160, 255), outline=(60, 40, 30, 255))
    draw.ellipse([cx - 60, cy - 220, cx - 20, cy - 170], fill=(40, 60, 120, 255))
    draw.ellipse([cx + 20, cy - 220, cx + 60, cy - 170], fill=(40, 60, 120, 255))
    draw.ellipse([cx - 150, cy - 340, cx + 150, cy - 120], fill=(60, 30, 90, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


AUDIO_B64 = make_wav()
IMAGE_B64 = make_png()
ANIME_B64 = make_anime_png()

# ─── Test Definitions ─────────────────────────────────────────────────────────

# Each test is (label, full_forge_payload, timeout_seconds).
# Wan2GP models: {"service": "wan2gp", "model": "family/model", ...extras}
# Other Forge services: {"service": "lance"|"avatar", ...payload}

TESTS: list[tuple[str, dict, int]] = [
    # ── CPU models (fast, no GPU) ──
    ("kokoro/kokoro", {"service": "wan2gp", "model": "kokoro/kokoro", "text": "The quick brown fox jumps over the lazy dog.", "voice": "af_heart"}, 60),
    ("espeak/espeak", {"service": "wan2gp", "model": "espeak/espeak", "text": "Testing one two three."}, 30),
    ("faster_whisper/faster_whisper", {"service": "wan2gp", "model": "faster_whisper/faster_whisper", "audio_b64": AUDIO_B64, "language": "en"}, 60),
    # ── GPU TTS ──
    ("faster_qwen3_tts/faster-qwen3-tts", {"service": "wan2gp", "model": "faster_qwen3_tts/faster-qwen3-tts", "text": "Hello, this is a synthesis test.", "voice": "Serena"}, 180),
    ("tts/index_tts2", {"service": "wan2gp", "model": "tts/index_tts2", "text": "Voice clone test.", "audio_b64": AUDIO_B64}, 180),
    # ── GPU Audio ──
    ("moss/moss-soundeffect", {"service": "wan2gp", "model": "moss/moss-soundeffect", "prompt": "rain on a tin roof", "seconds": 3}, 180),
    ("moss/moss_soundeffect_v2", {"service": "wan2gp", "model": "moss/moss_soundeffect_v2", "prompt": "thunder crashing in the distance", "seconds": 5, "steps": 50}, 120),
    ("tts/ace_step_v1_5", {"service": "wan2gp", "model": "tts/ace_step_v1_5", "prompt": "calm piano melody with strings", "duration_seconds": 5, "steps": 20}, 180),
    # ── GPU Image (native Wan2GP) ──
    ("flux/flux", {"service": "wan2gp", "model": "flux/flux", "prompt": "a cat sitting on a windowsill", "width": 512, "height": 512, "steps": 4, "seed": 42}, 120),
    ("flux/flux_schnell", {"service": "wan2gp", "model": "flux/flux_schnell", "prompt": "mountain sunset landscape", "width": 512, "height": 512, "steps": 4, "seed": 42}, 120),
    ("flux/flux2_klein_4b", {"service": "wan2gp", "model": "flux/flux2_klein_4b", "prompt": "a robot painting a picture", "width": 512, "height": 512, "steps": 4, "seed": 42, "embedded_guidance_scale": 1.0}, 120),
    ("flux/flux2_dev", {"service": "wan2gp", "model": "flux/flux2_dev", "prompt": "a dog playing in a park", "width": 512, "height": 512, "steps": 4, "seed": 42}, 180),
    # ── GPU Video (native Wan2GP) ──
    ("wan/t2v_1.3B", {"service": "wan2gp", "model": "wan/t2v_1.3B", "prompt": "ocean waves on a beach", "steps": 10, "seed": 42, "frame_num": 13}, 180),
    ("wan/t2v", {"service": "wan2gp", "model": "wan/t2v", "prompt": "a bird flying over mountains", "steps": 10, "seed": 42, "frame_num": 13}, 300),
    # ── GPU 3D ──
    ("trellis/trellis", {"service": "wan2gp", "model": "trellis/trellis", "image_b64": IMAGE_B64, "prompt": "A colorful 3D object"}, 300),
    ("pixal3d/pixal3d", {"service": "wan2gp", "model": "pixal3d/pixal3d", "image_b64": IMAGE_B64}, 180),
    # ── GPU Image editing ──
    # see_through disabled: ~15GB VRAM load crashes worker on overcommitted nodes.
    # ("see_through/see-through", {"service": "wan2gp", "model": "see_through/see-through", "prompt": "anime girl layers", "image_b64": ANIME_B64}, 180),
    # ── GPU 3D character ──
    ("anigen/anigen", {"service": "wan2gp", "model": "anigen/anigen", "prompt": "a male warrior character, fantasy style", "image_b64": IMAGE_B64}, 300),
    # ── GPU Motion ──
    ("hy_motion/hy-motion-1.0-lite", {"service": "wan2gp", "model": "hy_motion/hy-motion-1.0-lite", "prompt": "a person waving their hand hello"}, 180),
    # ── LANCE (text-to-image via Forge) ──
    # OOM: AWQ model + VAE + text encoder = ~14GB, exceeds available VRAM after Ray overhead.
    # Needs device_map="auto" or sequential loading in LANCE vendor code.
    # ("lance/t2i", {"service": "lance", "task": "t2i", "text": "a cat sitting on a windowsill at sunset", "num_timesteps": 10, "seed": 42}, 600),
    # ── Avatar / KIMODO (text-to-motion via Forge) ──
    # Timeout: Kimodo loading + inference exceeds 600s on first run.
    # ("avatar/kimodo", {"service": "avatar", "text": "a person waving their hand hello", "duration_seconds": 3.0, "denoising_steps": 50, "render": False}, 600),
]

# ─── Runner ───────────────────────────────────────────────────────────────────


def _release(service: str = "wan2gp"):
    try:
        requests.post(FORGE, json={"action": "release", "service": service}, timeout=60)
    except Exception:
        pass


def _release_all():
    for svc in ("wan2gp", "lance", "avatar"):
        _release(svc)


def test_model(label: str, payload: dict, timeout: int) -> dict:
    t0 = time.time()
    try:
        resp = requests.post(FORGE, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            body = resp.text[:300] if resp.text else f"HTTP {resp.status_code}"
            return {"status": "error", "error": body, "elapsed": elapsed}
        data = resp.json()
        data["elapsed"] = elapsed
        return data
    except requests.exceptions.ReadTimeout:
        return {"status": "error", "error": f"TIMEOUT after {timeout}s", "elapsed": time.time() - t0}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200], "elapsed": time.time() - t0}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="Only run these model keys (substring match)")
    ap.add_argument("--skip", nargs="*", default=[], help="Skip these model keys (substring match)")
    args = ap.parse_args()

    tests = TESTS
    if args.only:
        tests = [(l, p, t) for l, p, t in tests if any(k in l for k in args.only)]
    if args.skip:
        tests = [(l, p, t) for l, p, t in tests if not any(k in l for k in args.skip)]

    print(f"\n{'='*72}")
    print("  FORGE SERVICE TEST SUITE")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   {len(tests)} services")
    print(f"{'='*72}\n")

    results: list[tuple[str, dict]] = []

    for label, payload, timeout in tests:
        sys.stdout.write(f"  {label:45s} ")
        sys.stdout.flush()
        _release_all()
        result = test_model(label, payload, timeout)
        results.append((label, result))

        elapsed = result.get("elapsed", 0)
        if result.get("status") != "error":
            media = result.get("media_type", "")
            dl = len(result.get("data", ""))
            print(f"{elapsed:6.1f}s  {media} ({dl}B)")
        else:
            err = result.get("error", "")[:100]
            print(f"{elapsed:6.1f}s  FAIL: {err}")
        sys.stdout.flush()

    _release_all()

    ok = [l for l, r in results if r.get("status") != "error"]
    fail = [(l, r) for l, r in results if r.get("status") == "error"]

    print(f"\n{'='*72}")
    print(f"  {len(ok)} PASSED / {len(fail)} FAILED / {len(results)} TOTAL")
    print(f"{'='*72}")

    if fail:
        print("\n  Failures:")
        for label, r in fail:
            print(f"    {label}: {r.get('error', '')[:120]}")

    print()
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
