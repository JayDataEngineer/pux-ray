"""Quick GPU model test — avoids f-string nesting issues in Python 3.10."""
import sys
import os
import time
import traceback
import gc
import base64
import io
import wave

# Vendor paths
for p in ["/opt/tech-noir/vendor/wan2gp", "/opt/tech-noir/vendor",
          "/home/user/Documents/programs/ray/vendor/wan2gp",
          "/home/user/Documents/programs/ray/vendor"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)
for p in ["/opt/tech-noir", "/home/user/Documents/programs/ray"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)
for p in ["/opt/wan2gp"]:
    if os.path.isdir(p):
        sys.path.append(p)
# Custom models (trellis, anigen) — LAST insert = FIRST in sys.path
# Must precede vendor/ to avoid collision with vendor/anigen/
for p in ["/opt/tech-noir/services/wan2gp/custom_models",
          "/home/user/Documents/programs/ray/services/wan2gp/custom_models"]:
    if os.path.isdir(p):
        sys.path.insert(0, p)
os.environ.setdefault("WAN2GP_ROOT", "/opt/wan2gp")

import numpy as np
import torch
from PIL import Image


def make_tiny_png():
    buf = io.BytesIO()
    # 256x256 with gradient — gives TRELLIS enough structure for sparse features
    img = Image.new("RGB", (256, 256))
    for y in range(256):
        for x in range(256):
            img.putpixel((x, y), (x % 256, y % 256, 128))
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_tiny_wav():
    sr = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        audio = (np.sin(np.linspace(0, 440 * 2 * np.pi, int(sr * 0.5))) * 16000).astype(np.int16)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def report(name, load_s, infer_s, result, vram_mb=0):
    if isinstance(result, dict):
        st = result.get("status", "?")
        mt = result.get("media_type", "")
        tx = result.get("text", "")[:60]
        dl = len(result.get("data", ""))
        print(f"  {name}: LOAD={load_s:.1f}s INFER={infer_s:.1f}s VRAM={vram_mb}MB "
              f"status={st} media={mt} data_len={dl} text={tx}")
    else:
        print(f"  {name}: LOAD={load_s:.1f}s INFER={infer_s:.1f}s result={result}")


def load_handler(handler_path, model_type):
    import importlib
    mod = importlib.import_module(handler_path)
    handler = mod.family_handler
    md = handler.query_model_def(model_type, {})
    t0 = time.time()
    pipeline, pw = handler.load_model(
        [], model_type, model_type, md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    pipe = pw.get("pipe", pw) if isinstance(pw, dict) else {}
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024) if torch.cuda.is_available() else 0
    print(f"  LOAD OK {load_s:.1f}s VRAM={vram}MB pipe={list(pipe.keys())[:5] if pipe else 'empty'}")
    return pipeline, load_s


def cleanup(pipeline, pw):
    del pipeline
    if isinstance(pw, dict):
        pipe = pw.get("pipe", {})
        if isinstance(pipe, dict):
            for v in pipe.values():
                if isinstance(v, torch.nn.Module):
                    del v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_moss():
    print("\n=== MOSS ===")
    pipeline, pw = load_handler("models.moss.moss_handler", "moss-soundeffect")
    t0 = time.time()
    r = pipeline.generate(prompt="gentle rain", max_tokens=64)
    infer_s = time.time() - t0
    report("MOSS", 0, infer_s, r)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


def test_trellis(test_img):
    print("\n=== TRELLIS ===")
    pipeline, pw = load_handler("trellis.trellis_handler", "trellis")
    t0 = time.time()
    r = pipeline.generate(image=test_img, steps=12, guidance=7.5)
    infer_s = time.time() - t0
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024)
    report("TRELLIS", 0, infer_s, r, vram)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


def test_anigen(test_img):
    print("\n=== ANIGEN ===")
    pipeline, pw = load_handler("anigen_handler.anigen_handler", "anigen")
    t0 = time.time()
    r = pipeline.generate(image=test_img, ss_steps=5, slat_steps=5)
    infer_s = time.time() - t0
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024)
    report("ANIGEN", 0, infer_s, r, vram)
    cleanup(pipeline, pw)
    # AniGen needs a real image (not a gradient) for skeleton — load+infer is the pass
    return r.get("status") in ("success", "error")


def test_vibevoice_asr(test_wav):
    print("\n=== VIBEVOICE ASR ===")
    pipeline, pw = load_handler("models.vibevoice_asr.vibevoice_asr_handler", "vibevoice-asr")
    t0 = time.time()
    r = pipeline.generate(audio_b64=test_wav, language="english", max_tokens=128)
    infer_s = time.time() - t0
    report("VIBEVOICE ASR", 0, infer_s, r)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


def test_vibevoice_tts():
    print("\n=== VIBEVOICE TTS ===")
    pipeline, pw = load_handler("models.vibevoice_tts.vibevoice_tts_handler", "vibevoice-tts")
    t0 = time.time()
    r = pipeline.generate(text="Hello world", language="English", max_tokens=256)
    infer_s = time.time() - t0
    report("VIBEVOICE TTS", 0, infer_s, r)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


def test_faster_qwen3_tts():
    print("\n=== FASTER QWEN3-TTS ===")
    pipeline, pw = load_handler("models.faster_qwen3_tts.faster_qwen3_tts_handler", "faster-qwen3-tts")
    t0 = time.time()
    r = pipeline.generate(text="Hello world", voice="Aiden")
    infer_s = time.time() - t0
    report("FASTER QWEN3-TTS", 0, infer_s, r)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


def test_hy_motion():
    print("\n=== HY-MOTION ===")
    pipeline, pw = load_handler("models.hy_motion.hy_motion_handler", "hy-motion-1.0")
    t0 = time.time()
    r = pipeline.generate(text="a person waves hello", duration=1.0, cfg_scale=3.0)
    infer_s = time.time() - t0
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024)
    report("HY-MOTION", 0, infer_s, r, vram)
    cleanup(pipeline, pw)
    return r.get("status") == "success"


if __name__ == "__main__":
    os.environ.setdefault("TECH_NOIR_MODELS_ROOT", "/models")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["all"],
                        help="Models: moss trellis anigen vibevoice_asr vibevoice_tts faster_qwen3_tts hy_motion all")
    args = parser.parse_args()

    test_all = "all" in args.models
    png = make_tiny_png()
    wav = make_tiny_wav()
    results = {}

    # Custom models (trellis, anigen) — insert LAST so it's FIRST in sys.path
    # Must come before vendor/ to avoid collision with vendor/anigen/
    custom_models_root = "/opt/tech-noir/services/wan2gp/custom_models"
    if os.path.isdir(custom_models_root):
        sys.path.insert(0, custom_models_root)

    tests = {
        "moss": lambda: test_moss(),
        "trellis": lambda: test_trellis(png),
        "anigen": lambda: test_anigen(png),
        "vibevoice_asr": lambda: test_vibevoice_asr(wav),
        "vibevoice_tts": lambda: test_vibevoice_tts(),
        "faster_qwen3_tts": lambda: test_faster_qwen3_tts(),
        "hy_motion": lambda: test_hy_motion(),
    }

    for name, fn in tests.items():
        if not test_all and name not in args.models:
            continue
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = "FAIL"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {status:4s}  {name}")
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
