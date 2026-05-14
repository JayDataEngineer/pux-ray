"""Test each Wan2GP family_handler — import, load_model, generate.

For GPU models: tests load_model() + generate() with real models if available.
For CPU models: full end-to-end test.
Reports exact errors for each failure.
"""
import gc
import io
import os
import sys
import time
import traceback
import wave
import base64
import importlib
import logging

sys.path.insert(0, "/home/user/Documents/programs/ray/services/wan2gp/custom_models")
sys.path.insert(0, "/home/user/Documents/programs/ray")
sys.path.insert(0, "/opt/wan2gp")
sys.path.insert(0, "/home/user/Documents/programs/ray/vendor/wan2gp")
os.environ.setdefault("WAN2GP_ROOT", "/opt/wan2gp")

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("test_handlers")

import numpy as np
import torch


def make_tiny_wav_b64(duration_s=0.5, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        audio = (np.sin(np.linspace(0, 440 * 2 * np.pi, int(sr * duration_s))) * 16000).astype(np.int16)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def make_tiny_png_b64(size=64):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (size, size), (128, 64, 32, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


TINY_PNG_256 = make_tiny_png_b64(256)


TINY_WAV = make_tiny_wav_b64()
TINY_PNG = make_tiny_png_b64()


def _run_handler(handler_path, model_type, payload, *, timeout=120):
    """Import handler → load_model → generate → report."""
    print(f"\n{'='*60}")
    print(f"HANDLER: {handler_path}  model_type={model_type}")
    print(f"{'='*60}")
    t0 = time.time()

    # Step 1: Import handler
    try:
        mod = importlib.import_module(handler_path)
        handler = mod.family_handler
        print(f"  Import OK")
    except Exception as e:
        print(f"  IMPORT FAIL: {e}")
        traceback.print_exc()
        return False, f"import: {e}"

    # Step 2: Build model_def
    try:
        model_def = handler.query_model_def(model_type, {})
    except Exception as e:
        print(f"  MODEL_DEF FAIL: {e}")
        return False, f"model_def: {e}"

    # Step 3: Load
    try:
        t1 = time.time()
        pipeline, pipe_wrapper = handler.load_model(
            [], model_type, model_type, model_def,
            quantizeTransformer=False,
            text_encoder_quantization=None,
            dtype=None, VAE_dtype=None, profile=0,
        )
        load_s = time.time() - t1

        if isinstance(pipe_wrapper, dict):
            pipe = pipe_wrapper.get("pipe", pipe_wrapper)
            co_tenants = pipe_wrapper.get("coTenantsMap", {})
        else:
            pipe = {}
            co_tenants = {}

        vram = torch.cuda.memory_allocated(0) // (1024 * 1024) if torch.cuda.is_available() else 0
        print(f"  Load OK in {load_s:.1f}s  VRAM={vram}MB  pipe_keys={list(pipe.keys()) if isinstance(pipe, dict) else type(pipe)}")
    except ImportError as e:
        print(f"  LOAD SKIP (import): {e}")
        return None, f"load import: {e}"
    except FileNotFoundError as e:
        print(f"  LOAD SKIP (file): {e}")
        return None, f"load file: {e}"
    except Exception as e:
        print(f"  LOAD FAIL: {e}")
        traceback.print_exc()
        return False, f"load: {type(e).__name__}: {e}"

    # Step 4: Infer
    try:
        from services.wan2gp.deployment import _build_generate_kwargs
        kwargs = _build_generate_kwargs(payload, {})

        t2 = time.time()
        gen_result = pipeline.generate(**kwargs)
        infer_s = time.time() - t2

        status = gen_result.get("status", "unknown") if isinstance(gen_result, dict) else "ok"
        print(f"  Infer OK in {infer_s:.1f}s  status={status}")
        if isinstance(gen_result, dict):
            for k in ("text", "media_type", "error"):
                if k in gen_result and gen_result[k]:
                    print(f"    {k}={str(gen_result[k])[:80]}")
        print(f"  PASS")
        return True, f"ok ({load_s:.1f}s load, {infer_s:.1f}s infer, {vram}MB)"
    except Exception as e:
        print(f"  INFER FAIL: {e}")
        traceback.print_exc()
        return False, f"infer: {type(e).__name__}: {str(e)[:120]}"
    finally:
        del pipeline
        if isinstance(pipe, dict):
            for v in pipe.values():
                if isinstance(v, torch.nn.Module):
                    del v
        # Release mmgp-managed VRAM
        try:
            from mmgp import offload
            offload.unload_all()
            offload.release()
            offload.flush_torch_caches()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── Tests ──────────────────────────────────────────────────────────────────

TESTS = [
    ("models.espeak.espeak_handler", "espeak", {"text": "Hello world"}),
    ("models.kokoro.kokoro_handler", "kokoro", {"text": "Hello world", "voice": "af_bella"}),
    ("models.faster_whisper.faster_whisper_handler", "faster_whisper", {"audio_b64": TINY_WAV}),
    ("models.faster_qwen3_tts.faster_qwen3_tts_handler", "faster_qwen3_tts", {"text": "Hello world", "voice": "Aiden"}),
    ("anigen.anigen_handler", "anigen", {"image": TINY_PNG, "ss_steps": 1, "slat_steps": 1}),
    ("trellis.trellis_handler", "trellis", {"image": TINY_PNG, "steps": 1}),
    ("models.hy_motion.hy_motion_handler", "hy-motion-1.0", {"text": "a person waves", "duration": 1.0, "cfg_scale": 3.0}),
    ("models.moss.moss_handler", "moss-soundeffect", {"prompt": "gentle rain", "max_tokens": 64}),
    ("models.see_through.see_through_handler", "see_through", {"image": TINY_PNG_256, "resolution": 256, "steps": 2}),
    ("models.vibevoice_asr.vibevoice_asr_handler", "vibevoice-asr", {"audio_b64": TINY_WAV}),
    ("models.vibevoice_tts.vibevoice_tts_handler", "vibevoice-tts", {"text": "Hello world"}),
    ("models.vnccs.vnccs_handler", "qwen_image_edit_vnccs_20B", {"workflow": "char_sheet"}),
]


def main():
    os.environ.setdefault("TECH_NOIR_MODELS_ROOT", "/models")
    results = {}
    for handler_path, model_type, payload in TESTS:
        ok, msg = _run_handler(handler_path, model_type, payload)
        results[handler_path.split(".")[-2]] = (ok, msg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = failed = skipped = 0
    for name, (ok, msg) in results.items():
        if ok is True:
            tag = "PASS"
            passed += 1
        elif ok is False:
            tag = "FAIL"
            failed += 1
        else:
            tag = "SKIP"
            skipped += 1
        print(f"  {tag:4s}  {name:25s}  {msg}")

    print(f"\n  {passed} passed  {failed} failed  {skipped} skipped  ({len(results)} total)")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
