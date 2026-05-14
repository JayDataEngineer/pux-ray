"""Quick GPU model test — avoids f-string nesting issues in Python 3.10."""
import sys
import os
import time
import traceback
import gc
import base64
import io
import wave

sys.path.insert(0, "/home/user/Documents/programs/ray")
sys.path.insert(0, "/opt/wan2gp")
sys.path.insert(0, "/home/user/Documents/programs/ray/vendor/wan2gp")
os.environ.setdefault("WAN2GP_ROOT", "/opt/wan2gp")

import numpy as np
import torch
from PIL import Image


def make_tiny_png():
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (128, 64, 32, 255)).save(buf, format="PNG")
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


def test_moss():
    print("\n=== MOSS ===")
    from models.moss.moss_handler import family_handler
    md = family_handler.query_model_def("moss-soundeffect", {})
    t0 = time.time()
    pipeline, pw = family_handler.load_model(
        [], "moss-soundeffect", "moss-soundeffect", md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    t0 = time.time()
    r = pipeline.generate(prompt="gentle rain", max_tokens=64)
    infer_s = time.time() - t0
    report("MOSS", load_s, infer_s, r)
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return r.get("status") == "success"


def test_trellis(test_img):
    print("\n=== TRELLIS ===")
    from models.trellis.trellis_handler import family_handler
    md = family_handler.query_model_def("trellis", {})
    t0 = time.time()
    pipeline, pw = family_handler.load_model(
        [], "trellis", "trellis", md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    t0 = time.time()
    r = pipeline.generate(image=test_img, steps=12, guidance=7.5)
    infer_s = time.time() - t0
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024)
    report("TRELLIS", load_s, infer_s, r, vram)
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return r.get("status") == "success"


def test_anigen(test_img):
    print("\n=== ANIGEN ===")
    from models.anigen.anigen_handler import family_handler
    md = family_handler.query_model_def("anigen", {})
    t0 = time.time()
    pipeline, pw = family_handler.load_model(
        [], "anigen", "anigen", md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    t0 = time.time()
    r = pipeline.generate(image=test_img, ss_steps=5, slat_steps=5)
    infer_s = time.time() - t0
    vram = torch.cuda.memory_allocated(0) // (1024 * 1024)
    report("ANIGEN", load_s, infer_s, r, vram)
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return r.get("status") == "success"


def test_vibevoice_asr(test_wav):
    print("\n=== VIBEVOICE ASR ===")
    from models.vibevoice_asr.vibevoice_asr_handler import family_handler
    md = family_handler.query_model_def("vibevoice-asr", {})
    t0 = time.time()
    pipeline, pw = family_handler.load_model(
        [], "vibevoice-asr", "vibevoice-asr", md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    t0 = time.time()
    r = pipeline.generate(audio_b64=test_wav, language="english", max_tokens=128)
    infer_s = time.time() - t0
    report("VIBEVOICE ASR", load_s, infer_s, r)
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return r.get("status") == "success"


def test_vibevoice_tts():
    print("\n=== VIBEVOICE TTS ===")
    from models.vibevoice_tts.vibevoice_tts_handler import family_handler
    md = family_handler.query_model_def("vibevoice-tts", {})
    t0 = time.time()
    pipeline, pw = family_handler.load_model(
        [], "vibevoice-tts", "vibevoice-tts", md,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    load_s = time.time() - t0
    t0 = time.time()
    r = pipeline.generate(text="Hello world", language="English", max_tokens=256)
    infer_s = time.time() - t0
    report("VIBEVOICE TTS", load_s, infer_s, r)
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return r.get("status") == "success"


if __name__ == "__main__":
    os.environ.setdefault("TECH_NOIR_MODELS_ROOT", "/models")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["all"],
                        help="Models to test: moss trellis anigen vibevoice_asr vibevoice_tts all")
    args = parser.parse_args()

    test_all = "all" in args.models
    png = make_tiny_png()
    wav = make_tiny_wav()
    results = {}

    tests = {
        "moss": lambda: test_moss(),
        "trellis": lambda: test_trellis(png),
        "anigen": lambda: test_anigen(png),
        "vibevoice_asr": lambda: test_vibevoice_asr(wav),
        "vibevoice_tts": lambda: test_vibevoice_tts(),
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
