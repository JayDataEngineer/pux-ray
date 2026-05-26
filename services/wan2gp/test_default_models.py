"""Test ALL default Wan2GP models through the Forge HTTP API."""
import requests
import json
import time
import base64
import struct
import math
import io

FORGE_URL = "http://10.0.0.68:8000/forge"

def forge_invoke(model, timeout=600, **params):
    body = {"service": "wan2gp", "model": model, **params}
    t0 = time.time()
    r = requests.post(FORGE_URL, json=body, timeout=timeout)
    elapsed = time.time() - t0
    result = r.json()
    return result, elapsed

def result_str(r):
    if isinstance(r, dict):
        s = r.get("status", "?")
        m = r.get("media_type", "")
        e = r.get("error", "")
        dl = len(r.get("data", ""))
        if s == "ok":
            return f"OK  {m:15s} ({dl//1024}KB)"
        else:
            return f"ERR {e[:100]}"
    return f"ERR {str(r)[:100]}"

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

ARCH_TO_FAMILY = {
    "t2v": "wan", "t2v_1.3B": "wan", "t2v_2_2": "wan",
    "i2v": "wan", "i2v_2_2": "wan",
    "flux": "flux", "flux2_dev": "flux", "flux2_klein_4b": "flux",
    "flux_schnell": "flux", "flux_chroma": "flux",
    "hunyuan": "hunyuan", "hunyuan_i2v": "hunyuan",
    "ltx2_19B": "ltx2", "ltx2_22B": "ltx2",
    "ltxv_13B": "ltxv",
    "ace_step_v1_5": "tts",
    "index_tts2": "tts",
    "trellis": "trellis",
    "z_image": "z_image",
    "qwen_image_20B": "qwen",
    "hidream_o1": "hidream",
    "longcat_video": "longcat", "longcat_avatar": "longcat",
    "magi_human": "magi_human",
    "k5_pro_t2v": "kandinsky5", "k5_lite_t2v": "kandinsky5",
}

# Test matrix: (label, model_name, params, timeout_sec)
TESTS = [
    # ── Wan T2V family ──
    ("Wan T2V 1.3B", "t2v_1.3B", {"prompt": "a cat walking on a beach", "steps": 4, "seed": 42, "frame_num": 5}, 300),
    ("Wan T2V 14B", "t2v", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 5}, 300),

    # ── Wan I2V family (needs image) ──
    ("Wan I2V 14B", "i2v", {"prompt": "a cat walking", "image_b64": image_b64, "steps": 4, "seed": 42, "frame_num": 5}, 300),

    # ── Flux image gens ──
    ("Flux 2 Klein 4B", "flux2_klein_4b", {"prompt": "a serene mountain landscape", "steps": 4, "seed": 42, "embedded_guidance_scale": 1.0}, 180),
    ("Flux 1 Dev", "flux", {"prompt": "a beautiful sunset", "steps": 4, "seed": 42}, 300),
    ("Flux 1 Schnell", "flux_schnell", {"prompt": "a beautiful sunset", "steps": 4, "seed": 42}, 300),
    ("Flux 2 Dev", "flux2_dev", {"prompt": "a cat wearing a spacesuit", "steps": 4, "seed": 42}, 300),
    ("Flux Chroma", "flux_chroma", {"prompt": "a colorful abstract painting", "steps": 4, "seed": 42}, 300),

    # ── Hunyuan Video ──
    ("Hunyuan T2V", "hunyuan", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),

    # ── LTX Video ──
    ("LTX-2 22B", "ltx2_22B", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),
    ("LTX-2 19B", "ltx2_19B", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),
    ("LTXV 13B", "ltxv_13B", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),

    # ── Z-Image ──
    ("Z-Image", "z_image", {"prompt": "a beautiful landscape", "steps": 4, "seed": 42}, 300),

    # ── Qwen Image ──
    ("Qwen Image 20B", "qwen_image_20B", {"prompt": "a beautiful landscape", "steps": 4, "seed": 42}, 300),

    # ── HiDream ──
    ("HiDream O1", "hidream_o1", {"prompt": "a beautiful landscape", "steps": 4, "seed": 42}, 300),

    # ── Vace ──
    ("Vace 1.3B", "vace_1.3B", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 5}, 300),

    # ── Kandinsky 5 ──
    ("K5 Pro T2V", "k5_pro_t2v", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),

    # ── Magi Human ──
    ("Magi Human", "magi_human", {"prompt": "a person walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),

    # ── LongCat ──
    ("LongCat Video", "longcat_video", {"prompt": "a cat walking", "steps": 4, "seed": 42, "frame_num": 9}, 300),
]

print(f"{'='*80}")
print(f"  TESTING DEFAULT WAN2GP MODELS")
print(f"  Forge: {FORGE_URL}")
print(f"{'='*80}")

results = {}
for label, model, params, timeout in TESTS:
    print(f"\n  {label:40s} [{model:20s}] ", end="", flush=True)
    try:
        result, elapsed = forge_invoke(model, timeout=timeout, **params)
        print(f"[{elapsed:5.1f}s] {result_str(result)}")
        results[label] = result
    except requests.Timeout:
        print(f"[TIMEOUT>{timeout}s]")
        results[label] = {"status": "error", "error": "timeout"}
    except Exception as e:
        print(f"[CRASH] {str(e)[:80]}")
        results[label] = {"status": "error", "error": str(e)}

print(f"\n{'='*80}")
print(f"  RESULTS SUMMARY")
print(f"{'='*80}")
ok = 0
fail = 0
for label, r in results.items():
    s = r.get("status", "")
    icon = "✅" if s == "ok" else "❌"
    if s == "ok":
        ok += 1
    else:
        fail += 1
    print(f"  {icon} {label:40s} {r.get('media_type',''):15s} {r.get('error','ok')[:80]}")
print(f"\n  Pass: {ok}  Fail: {fail}")
print(f"{'='*80}")

# Save results
with open("/tmp/test_default_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to /tmp/test_default_results.json")
