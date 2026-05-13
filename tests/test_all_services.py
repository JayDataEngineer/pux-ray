"""End-to-end test for ALL services through the Wan2GP unified system.

Tests every entry in the dynamic model registry via Wan2GPService.load() → infer() → unload().
Skips models that need Docker-only deps (marks clearly).
Reports pass/fail per model + summary.

Usage: python tests/test_all_services.py
"""
import sys
import time
import traceback

sys.path.insert(0, "/home/user/Documents/programs/ray")

from services.wan2gp.deployment import Wan2GPService


def vram():
    try:
        import torch
        return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        return 0


def test_model(service, model_key, payload):
    """Load → infer → unload for a single model. Returns (ok, msg)."""
    print(f"\n{'='*60}")
    print(f"TEST: {model_key}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        print(f"  Loading {model_key}...")
        service.load(model_key)
        load_s = time.time() - t0
        vram_after_load = vram()
        print(f"  Loaded in {load_s:.1f}s  VRAM: {vram_after_load}MB")

        print(f"  Inferring...")
        t1 = time.time()
        result = service.infer(payload)
        infer_s = time.time() - t1
        status = result.get("status", "unknown")
        media = result.get("media_type", result.get("text", "n/a"))

        if status in ("ok", "success"):
            print(f"  OK in {infer_s:.1f}s  status={status}  media={str(media)[:60]}")
        else:
            err = result.get("error", "unknown error")
            print(f"  FAIL: {err[:120]}")
            service.unload()
            return False, f"inference error: {err[:80]}"

        service.unload()
        total_s = time.time() - t0
        print(f"  Unloaded. Total: {total_s:.1f}s  VRAM after unload: {vram()}MB")
        return True, f"ok ({load_s:.1f}s load, {infer_s:.1f}s infer, {vram_after_load}MB VRAM)"

    except ImportError as e:
        service.unload()
        missing = str(e).split(":")[-1].strip()
        print(f"  SKIP (missing dep): {missing}")
        return None, f"missing dep: {missing}"
    except FileNotFoundError as e:
        service.unload()
        print(f"  SKIP (model not found): {e}")
        return None, f"model not found: {e}"
    except Exception as e:
        service.unload()
        tb = traceback.format_exc().split("\n")[-2]
        print(f"  FAIL: {e}")
        print(f"    {tb}")
        return False, f"{type(e).__name__}: {str(e)[:80]}"


# ─── Test payloads per model ──────────────────────────────────────────────

PAYLOADS = {
    # Vendor
    "wan/t2v-14B": {"prompt": "a cat walking", "steps": 1},
    "wan/i2v-14B": {"prompt": "a cat", "steps": 1},
    "hunyuan/t2v": {"prompt": "a dog running", "steps": 1},
    "flux/t2i": {"prompt": "a landscape", "steps": 1},
    "ace_step/v1_5": {"prompt": "jazz music", "duration": 5, "steps": 2},
    "index_tts/v2": {"text": "hello world"},
    # GPU model_engine
    "anigen": {"image_b64": "x", "ss_steps": 1, "slat_steps": 1},
    "trellis": {"image_b64": "x", "steps": 1},
    "hy_motion": {"text": "a person waves", "duration": 1.0, "cfg_scale": 3.0},
    "moss_soundeffect": {"prompt": "gentle rain", "max_tokens": 64},
    "see_through": {"image_b64": "x", "resolution": 256, "steps": 1},
    "faster_qwen3_tts": {"text": "Hello world", "voice": "Aiden"},
    "vibevoice_asr": {"audio_b64": "x"},
    "vibevoice_tts": {"text": "Hello world"},
    # CPU model_engine
    "kokoro": {"text": "Hello world", "voice": "af_bella"},
    "espeak": {"text": "Hello world"},
    "faster_whisper": {"audio_b64": "x"},
}


def main():
    service = Wan2GPService()
    results = {}

    keys = service.available_models()
    print(f"Testing {len(keys)} available models from dynamic registry")
    print(f"Models: {keys}")

    for key in keys:
        payload = PAYLOADS.get(key, {"prompt": "test"})
        ok, msg = test_model(service, key, payload)
        results[key] = (ok, msg)

        # Clean up between tests
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    skipped = 0

    for key in keys:
        ok, msg = results[key]
        if ok is True:
            tag = "PASS"
            passed += 1
        elif ok is False:
            tag = "FAIL"
            failed += 1
        else:
            tag = "SKIP"
            skipped += 1
        print(f"  {tag}: {key:30s} {msg}")

    print(f"\n  {passed} passed  {failed} failed  {skipped} skipped  ({len(keys)} total)")

    if failed > 0:
        print("\n  FAILURES need fixing before deploy.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
