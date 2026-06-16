#!/usr/bin/env python3
"""Convert Wan2.2-VACE-Fun-A14B BF16 weights → FP8 e4m3fn.

Memory-efficient: reads tensors one-at-a-time via safetensors mmap, casts to
FP8, accumulates in a new dict. Peak RSS ≈ FP8 output size (~17GB per expert).

Output layout:
  /mnt/data/models/video/wan-vace-fun-a14b/
    high_noise_model/diffusion_pytorch_model_fp8_e4m3fn.safetensors  (~17GB)
    low_noise_model/diffusion_pytorch_model_fp8_e4m3fn.safetensors   (~17GB)
    models_t5_umt5-xxl-enc-fp8.pth  (downloaded from Kijai — separate step)
    Wan2.1_VAE.pth  (kept BF16 — only 0.5GB, precision-critical)

Usage:
  python3 scripts/quantize_vace_fp8.py                    # quantize both experts
  python3 scripts/quantize_vace_fp8.py --only high        # just high_noise
  python3 scripts/quantize_vace_fp8.py --only low         # just low_noise
"""
import argparse, gc, os, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/mnt/data/models")
MODEL_DIR = Path(MODELS_ROOT) / "video" / "wan-vace-fun-a14b"

EXPERTS = {
    "high": ("high_noise_model", "diffusion_pytorch_model.safetensors",
             "diffusion_pytorch_model_fp8_e4m3fn.safetensors"),
    "low":  ("low_noise_model",  "diffusion_pytorch_model.safetensors",
             "diffusion_pytorch_model_fp8_e4m3fn.safetensors"),
}


def quantize_expert(name: str, subdir: str, src_name: str, dst_name: str):
    src_path = MODEL_DIR / subdir / src_name
    dst_path = MODEL_DIR / subdir / dst_name

    if not src_path.exists():
        print(f"  ❌ Source not found: {src_path}")
        return False

    if dst_path.exists():
        src_gb = src_path.stat().st_size / 1e9
        dst_gb = dst_path.stat().st_size / 1e9
        print(f"  SKIP {name}: already exists ({dst_gb:.1f}GB, src was {src_gb:.1f}GB)")
        return True

    src_gb = src_path.stat().st_size / 1e9
    print(f"  Quantizing {name}: {src_path.name} ({src_gb:.1f}GB BF16 → ~{src_gb/2:.1f}GB FP8)")

    t0 = time.perf_counter()
    fp8_dict = {}
    tensor_count = 0

    # Memory-mapped read — one tensor at a time, never loads full file
    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total = len(keys)
        for key in keys:
            tensor = f.get_tensor(key)
            # Cast to FP8 e4m3fn (halves memory)
            fp8_dict[key] = tensor.to(torch.float8_e4m3fn)
            del tensor
            tensor_count += 1
            if tensor_count % 200 == 0:
                rss_gb = _get_rss_gb()
                print(f"    [{tensor_count}/{total}] {rss_gb:.1f}GB RSS")

    elapsed = time.perf_counter() - t0
    print(f"    Cast complete: {tensor_count} tensors in {elapsed:.1f}s")

    # Save FP8 file
    print(f"    Saving → {dst_path.name}...")
    t1 = time.perf_counter()
    save_file(fp8_dict, str(dst_path), metadata={"format": "pt", "quantization": "fp8_e4m3fn"})
    save_elapsed = time.perf_counter() - t1

    dst_gb = dst_path.stat().st_size / 1e9
    del fp8_dict
    gc.collect()

    total_elapsed = time.perf_counter() - t0
    print(f"  ✅ {name}: {src_gb:.1f}GB → {dst_gb:.1f}GB in {total_elapsed:.1f}s "
          f"(save: {save_elapsed:.1f}s)")
    return True


def _get_rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Quantize Wan VACE-Fun-A14B to FP8 e4m3fn")
    parser.add_argument("--only", choices=["high", "low"], help="Only quantize one expert")
    args = parser.parse_args()

    print(f"Model dir: {MODEL_DIR}")
    print(f"FP8 output dtype: torch.float8_e4m3fn")
    print()

    targets = [args.only] if args.only else list(EXPERTS.keys())
    for name in targets:
        subdir, src, dst = EXPERTS[name]
        print(f"── Expert: {name} ──")
        quantize_expert(name, subdir, src, dst)
        print()

    # Summary
    print("=== Summary ===")
    for name in targets:
        subdir, src, dst = EXPERTS[name]
        dst_path = MODEL_DIR / subdir / dst
        if dst_path.exists():
            gb = dst_path.stat().st_size / 1e9
            print(f"  ✅ {name}: {dst_path} ({gb:.1f}GB)")
        else:
            print(f"  ❌ {name}: not created")

    total_fp8 = sum(
        (MODEL_DIR / EXPERTS[n][0] / EXPERTS[n][2]).stat().st_size / 1e9
        for n in targets if (MODEL_DIR / EXPERTS[n][0] / EXPERTS[n][2]).exists()
    )
    print(f"\n  Total FP8 weight: {total_fp8:.1f}GB (both experts)")
    print(f"  + T5 FP8: ~5.7GB (download from Kijai separately)")
    print(f"  + VAE BF16: 0.5GB (kept as-is)")
    print(f"  Grand total: ~{total_fp8 + 5.7 + 0.5:.1f}GB (fits in 59GB RAM ✅)")


if __name__ == "__main__":
    main()
