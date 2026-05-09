#!/usr/bin/env python3
"""Integration test for Tech Noir services — tier-aware testing."""
import base64, io, json, sys, time, argparse
import httpx

def make_png():
    from PIL import Image
    img = Image.new("RGB", (128, 128), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def make_wav():
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.sin(2 * np.pi * 440 * np.arange(0, 1.0, 1/sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, t, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

IMG = make_png()
WAV = make_wav()

PASS = 0
FAIL = 0

def test(name, url, payload, timeout=300):
    global PASS, FAIL
    print(f"  {name:25s}", end=" ", flush=True)
    try:
        if payload is None:
            # Health check: GET request, check for 200
            r = httpx.get(url, timeout=timeout)
            if r.status_code == 200:
                print("PASS")
                PASS += 1
            else:
                print(f"FAIL (HTTP {r.status_code})")
                FAIL += 1
        else:
            r = httpx.post(url, json=payload, timeout=timeout)
            d = r.json()
            st = d.get("status", "NO_STATUS")
            if st == "success":
                print("PASS")
                PASS += 1
            else:
                err = d.get("error", "")[:120]
                print(f"FAIL ({st}) {err}")
                FAIL += 1
    except Exception as e:
        print(f"FAIL ({e})")
        FAIL += 1

def main():
    parser = argparse.ArgumentParser(description="Tech Noir Integration Tests")
    parser.add_argument("--base", default="http://localhost:18080", help="Base URL")
    parser.add_argument("--tier1", action="store_true", default=True, help="Test Tier 1 services")
    parser.add_argument("--tier2", action="store_true", default=False, help="Test Tier 2 services")
    parser.add_argument("--all", action="store_true", default=False, help="Test all services")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test (seconds)")
    args = parser.parse_args()

    BASE = args.base.rstrip("/")
    T = args.timeout

    print("=" * 50)
    print(" Tech Noir Integration Test (Tiered)")
    print("=" * 50)

    if args.tier1 or args.all:
        print("\n--- Tier 1: First-Class Citizens ---")
        test("kokoro", f"{BASE}/tts/kokoro/", {"action": "generate", "input": {"text": "Hello"}}, 30)
        test("espeak", f"{BASE}/tts/espeak/", {"action": "generate", "input": {"text": "Hello"}}, 30)
        test("faster_whisper", f"{BASE}/asr/whisper/", {"action": "generate", "input": {"audio_b64": WAV}}, 30)
        test("faster_qwen3_tts", f"{BASE}/tts/faster-qwen3-tts/", {"action": "generate", "input": {"text": "Hello", "voice": "Aiden"}}, T)
        test("index_tts", f"{BASE}/tts/index-tts/", {"action": "generate", "input": {"text": "Hello"}}, T)
        test("trellis", f"{BASE}/3d/trellis/", {"action": "generate", "input": {"image_b64": IMG}, "config": {"low_resource": True}}, T)
        test("ace_step", f"{BASE}/music/ace-step/", {"action": "generate", "input": {"prompt": "ambient pad"}}, T)
        test("vibevoice_cpp_gpu_tts", f"{BASE}/tts/vibevoice-cpp-gpu/", {"action": "generate", "input": {"text": "Hello from vibevoice cpp."}}, T)
        test("vibevoice_cpp_gpu_asr", f"{BASE}/tts/vibevoice-cpp-gpu/", {"action": "generate", "input": {"audio_b64": WAV}}, T)

        print("\n--- Tier 1: GPU-Exclusive (needs own GPU) ---")
        # ComfyUI has its own API; just verify the server is up
        test("comfyui_health", f"{BASE}/image/comfyui/", None, 120)

    if args.tier2 or args.all:
        print("\n--- Tier 2: Second-Class Citizens ---")

    if args.all:
        print("\n--- Tier 3: Experimental ---")
        test("qwen_tts_legacy", f"{BASE}/tts/qwen-tts/", {"action": "generate", "input": {"text": "Hello"}}, T)
        test("gpt_sovits", f"{BASE}/tts/gpt-sovits/", {"action": "generate", "input": {"text": "Hello"}}, T)
        test("moss_sfx", f"{BASE}/audio/moss-sfx/", {"action": "generate", "input": {"prompt": "thunder"}}, T)

    print(f"\n{'=' * 50}")
    print(f" Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 50}")

    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
