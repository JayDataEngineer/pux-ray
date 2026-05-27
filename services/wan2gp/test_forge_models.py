"""Forge model test suite — validates every Wan2GP model end-to-end.

Usage:
    kubectl exec -n ai-services <head-pod> -- python3 /app/services/wan2gp/test_forge_models.py
    kubectl exec -n ai-services <worker-pod> -c ray-worker -- python3 /app/services/wan2gp/test_forge_models.py

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
RELEASE = {"action": "release", "service": "wan2gp"}

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
    # Background gradient bands
    for y in range(h):
        r = int(180 + 60 * y / h)
        g = int(200 - 80 * y / h)
        b = int(160 + 40 * y / h)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    # Foreground shapes (circles, rectangles — objects to detect)
    draw.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(180, 60, 60), outline=(100, 30, 30))
    draw.rectangle([w // 6, h // 6, w // 3, h // 3], fill=(60, 180, 60))
    draw.ellipse([2 * w // 3, 2 * h // 3, 5 * w // 6, 5 * h // 6], fill=(60, 60, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_anime_png(w: int = 1280, h: int = 1280) -> str:
    """Generate an anime-style test image for layer-decomposition models.

    See-through needs recognizable foreground/background structure with
    edges and color variation to decompose into layers.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (w, h), (255, 240, 245, 255))
    draw = ImageDraw.Draw(img)
    # Gradient background
    for y in range(h):
        r = int(200 + 40 * y / h)
        g = int(180 + 30 * (1 - y / h))
        b = int(220 - 60 * y / h)
        draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
    # Foreground figure (simple character silhouette)
    cx, cy = w // 2, h // 2
    # Head
    draw.ellipse([cx - 120, cy - 300, cx + 120, cy - 60], fill=(255, 220, 200, 255), outline=(60, 40, 30, 255))
    # Body
    draw.rectangle([cx - 100, cy - 80, cx + 100, cy + 200], fill=(180, 100, 160, 255), outline=(60, 40, 30, 255))
    # Eyes
    draw.ellipse([cx - 60, cy - 220, cx - 20, cy - 170], fill=(40, 60, 120, 255))
    draw.ellipse([cx + 20, cy - 220, cx + 60, cy - 170], fill=(40, 60, 120, 255))
    # Hair
    draw.ellipse([cx - 150, cy - 340, cx + 150, cy - 120], fill=(60, 30, 90, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


AUDIO_B64 = make_wav()
IMAGE_B64 = make_png()
ANIME_B64 = make_anime_png()

# ─── Model Test Definitions ──────────────────────────────────────────────────

# (model_key, payload_extras, timeout_seconds)
# model_key is the exact Forge registry key.
# payload_extras are merged into {"service": "wan2gp", "model": model_key}.

TESTS: list[tuple[str, dict, int]] = [
    # ── CPU models (fast, no GPU) ──
    ("kokoro/kokoro", {"text": "The quick brown fox jumps over the lazy dog.", "voice": "af_heart"}, 60),
    ("espeak/espeak", {"text": "Testing one two three."}, 30),
    ("faster_whisper/faster_whisper", {"audio_b64": AUDIO_B64, "language": "en"}, 60),
    # ── GPU TTS ──
    ("faster_qwen3_tts/faster-qwen3-tts", {"text": "Hello, this is a synthesis test.", "voice": "Serena"}, 180),
    ("tts/index_tts2", {"text": "Voice clone test.", "audio_b64": AUDIO_B64}, 180),
    # ── GPU Audio ──
    ("moss/moss-soundeffect", {"prompt": "rain on a tin roof", "seconds": 3}, 180),
    ("moss/moss_soundeffect_v2", {"prompt": "thunder crashing in the distance", "seconds": 5, "steps": 50}, 120),
    ("tts/ace_step_v1_5", {"prompt": "calm piano melody with strings", "duration_seconds": 5, "steps": 20}, 180),
    # ── GPU Image (native Wan2GP) ──
    ("flux/flux", {"prompt": "a cat sitting on a windowsill", "width": 512, "height": 512, "steps": 4, "seed": 42}, 120),
    ("flux/flux_schnell", {"prompt": "mountain sunset landscape", "width": 512, "height": 512, "steps": 4, "seed": 42}, 120),
    ("flux/flux2_klein_4b", {"prompt": "a robot painting a picture", "width": 512, "height": 512, "steps": 4, "seed": 42, "embedded_guidance_scale": 1.0}, 120),
    ("flux/flux2_dev", {"prompt": "a dog playing in a park", "width": 512, "height": 512, "steps": 4, "seed": 42}, 180),
    # ── GPU Video (native Wan2GP) ──
    ("wan/t2v_1.3B", {"prompt": "ocean waves on a beach", "steps": 10, "seed": 42, "frame_num": 13}, 180),
    ("wan/t2v", {"prompt": "a bird flying over mountains", "steps": 10, "seed": 42, "frame_num": 13}, 300),
    # ── GPU 3D ──
    ("trellis/trellis", {"image_b64": IMAGE_B64, "prompt": "A colorful 3D object"}, 300),
    ("pixal3d/pixal3d", {"image_b64": IMAGE_B64}, 180),  # known: needs NATTEN built in image
    # ── GPU Image editing ──
    # see_through disabled: ~15GB VRAM load crashes worker on overcommitted nodes.
    # Re-enable when node has headroom or see_through gets mmgp decomposition.
    # ("see_through/see-through", {"prompt": "anime girl layers", "image_b64": ANIME_B64}, 180),
    # ── GPU 3D character ──
    ("anigen/anigen", {"prompt": "a male warrior character, fantasy style", "image_b64": IMAGE_B64}, 300),
    # ── GPU Motion ──
    ("hy_motion/hy-motion-1.0-lite", {"prompt": "a person waving their hand hello"}, 180),
]

# ─── Runner ───────────────────────────────────────────────────────────────────


def _release():
    try:
        requests.post(FORGE, json=RELEASE, timeout=60)
    except Exception:
        pass


def test_model(model: str, extras: dict, timeout: int) -> dict:
    payload = {"service": "wan2gp", "model": model, "seed": 42, **extras}
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
        tests = [(m, e, t) for m, e, t in tests if any(k in m for k in args.only)]
    if args.skip:
        tests = [(m, e, t) for m, e, t in tests if not any(k in m for k in args.skip)]

    print(f"\n{'='*72}")
    print("  FORGE MODEL TEST SUITE")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   {len(tests)} models")
    print(f"{'='*72}\n")

    results: list[tuple[str, dict]] = []

    for model, extras, timeout in tests:
        sys.stdout.write(f"  {model:45s} ")
        sys.stdout.flush()
        _release()
        result = test_model(model, extras, timeout)
        results.append((model, result))

        elapsed = result.get("elapsed", 0)
        if result.get("status") != "error":
            media = result.get("media_type", "")
            dl = len(result.get("data", ""))
            print(f"{elapsed:6.1f}s  {media} ({dl}B)")
        else:
            err = result.get("error", "")[:100]
            print(f"{elapsed:6.1f}s  FAIL: {err}")
        sys.stdout.flush()

    _release()

    ok = [m for m, r in results if r.get("status") != "error"]
    fail = [(m, r) for m, r in results if r.get("status") == "error"]

    print(f"\n{'='*72}")
    print(f"  {len(ok)} PASSED / {len(fail)} FAILED / {len(results)} TOTAL")
    print(f"{'='*72}")

    if fail:
        print("\n  Failures:")
        for model, r in fail:
            print(f"    {model}: {r.get('error', '')[:120]}")

    print()
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
