"""Convert model weight files from FP32 to FP16/BF16.

Scans directories or single files for .safetensors, .pth, .pt, .ckpt files,
identifies FP32 tensors, and converts them to the target dtype.

Usage:
    python3 scripts/convert_fp16.py /mnt/data/models/ --dry-run
    python3 scripts/convert_fp16.py /mnt/data/models/kokoro-v1_0.pth --apply
    python3 scripts/convert_fp16.py /mnt/data/models/ --apply --dtype bfloat16
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

WEIGHT_EXTENSIONS = {".safetensors", ".pth", ".pt", ".ckpt"}

# Models to skip entirely — already quantized or have dtype constraints
MODEL_SKIP = {
    "trellis": "ss_decoder requires fp32+fp16 mixed; bf16 breaks thresholding",
    "moss": "Already BF16",
    "pixal3d": "Already BF16",
}


def dtype_name(dt: torch.dtype) -> str:
    return {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}.get(dt, str(dt))


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path))


def save_safetensors(path: Path, tensors: dict[str, torch.Tensor]):
    from safetensors.torch import save_file
    save_file(tensors, str(path))


def load_pytorch(path: Path) -> dict[str, torch.Tensor]:
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if any(isinstance(v, torch.Tensor) for v in data.values()):
            return data
        # Nested state dict — flatten one level
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        return flat
    return {}


def save_pytorch(path: Path, tensors: dict[str, torch.Tensor]):
    torch.save(tensors, str(path))


def load_weights(path: Path) -> dict[str, torch.Tensor] | None:
    try:
        if path.suffix == ".safetensors":
            return load_safetensors(path)
        return load_pytorch(path)
    except Exception as e:
        print(f"  ERROR loading {path.name}: {e}")
        return None


def save_weights(path: Path, tensors: dict[str, torch.Tensor]):
    if path.suffix == ".safetensors":
        save_safetensors(path, tensors)
    else:
        save_pytorch(path, tensors)


def should_skip(path: Path) -> str | None:
    name_lower = path.name.lower()
    for skip_name, reason in MODEL_SKIP.items():
        if skip_name in name_lower or skip_name in str(path.parent).lower():
            return f"{skip_name}: {reason}"
    return None


def convert_file(path: Path, target_dtype: torch.dtype, apply: bool) -> dict:
    result = {"path": path, "skipped": False, "error": None, "fp32_count": 0,
              "original_size": 0, "new_size": 0, "saved_bytes": 0}

    result["original_size"] = path.stat().st_size

    skip_reason = should_skip(path)
    if skip_reason:
        result["skipped"] = True
        result["skip_reason"] = skip_reason
        return result

    tensors = load_weights(path)
    if tensors is None:
        result["error"] = "Failed to load"
        return result

    # Count dtypes
    dtype_counts = {}
    for v in tensors.values():
        if isinstance(v, torch.Tensor):
            dt = dtype_name(v.dtype)
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    fp32_keys = [k for k, v in tensors.items() if isinstance(v, torch.Tensor) and v.dtype == torch.float32]
    result["fp32_count"] = len(fp32_keys)
    result["dtype_breakdown"] = dtype_counts

    if not fp32_keys:
        result["skipped"] = True
        result["skip_reason"] = "No FP32 tensors"
        return result

    # Estimate savings: each FP32 param is 4 bytes, target is 2 bytes
    fp32_params = sum(tensors[k].numel() for k in fp32_keys)
    estimated_savings = fp32_params * 2  # 4 bytes - 2 bytes per param
    result["estimated_savings"] = estimated_savings

    if not apply:
        return result

    # Apply conversion
    for k in fp32_keys:
        tensors[k] = tensors[k].to(target_dtype)

    # Backup, write, verify
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(str(path), str(bak))

    try:
        save_weights(path, tensors)
        result["new_size"] = path.stat().st_size
        result["saved_bytes"] = result["original_size"] - result["new_size"]

        if result["new_size"] >= result["original_size"]:
            result["error"] = f"Converted file not smaller ({result['new_size']} >= {result['original_size']})"
            shutil.copy2(str(bak), str(path))
            bak.unlink()
            return result

        bak.unlink()
    except Exception as e:
        result["error"] = f"Write failed: {e}"
        if bak.exists():
            shutil.copy2(str(bak), str(path))
            bak.unlink()

    return result


def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b/1024:.1f}KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.2f}GB"


def find_weight_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix in WEIGHT_EXTENSIONS else []

    files = []
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            p = Path(root) / fn
            if p.suffix in WEIGHT_EXTENSIONS:
                files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Convert model weights FP32 -> FP16/BF16")
    parser.add_argument("path", type=Path, help="Directory or single weight file")
    parser.add_argument("--apply", action="store_true", help="Actually write converted files (default: dry-run)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                        help="Target dtype (default: float16)")
    args = parser.parse_args()

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    mode = "APPLY" if args.apply else "DRY-RUN"

    path = args.path.resolve()
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    files = find_weight_files(path)
    if not files:
        print(f"No weight files found in {path}")
        sys.exit(0)

    print(f"{'=' * 70}")
    print(f"  Weight Conversion: FP32 -> {dtype_name(target_dtype)} [{mode}]")
    print(f"  Scanning: {path}")
    print(f"  Files: {len(files)}")
    print(f"{'=' * 70}\n")

    total_original = 0
    total_saved = 0
    results = []

    for f in files:
        name = str(f.relative_to(path)) if path.is_dir() else f.name
        print(f"  {name:60s} ", end="", flush=True)
        t0 = time.time()
        r = convert_file(f, target_dtype, args.apply)
        dt = time.time() - t0

        if r["skipped"]:
            reason = r.get("skip_reason", "unknown")
            print(f"[SKIP] {reason} ({dt:.1f}s)")
        elif r["error"]:
            print(f"[ERR]  {r['error']} ({dt:.1f}s)")
        elif args.apply:
            saved = r["saved_bytes"]
            total_saved += saved
            total_original += r["original_size"]
            print(f"[DONE] {fmt_bytes(r['original_size'])} -> {fmt_bytes(r['new_size'])} "
                  f"(saved {fmt_bytes(saved)}) ({dt:.1f}s)")
        else:
            est = r.get("estimated_savings", 0)
            fp32 = r["fp32_count"]
            total_original += r["original_size"]
            total_saved += est
            breakdown = " ".join(f"{dtype}:{cnt}" for dtype, cnt in sorted(r["dtype_breakdown"].items()))
            print(f"[CONV] {fmt_bytes(r['original_size'])} ~{fmt_bytes(est)} saved "
                  f"({fp32} FP32 tensors, {breakdown}) ({dt:.1f}s)")

        results.append(r)

    # Summary
    converted = sum(1 for r in results if not r["skipped"] and not r["error"] and r["fp32_count"] > 0)
    skipped = sum(1 for r in results if r["skipped"])
    errors = sum(1 for r in results if r["error"])
    no_fp32 = sum(1 for r in results if r["skipped"] and r.get("skip_reason") == "No FP32 tensors")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY [{mode}]")
    print(f"{'=' * 70}")
    print(f"  Total files:     {len(results)}")
    print(f"  Convertible:     {converted}")
    print(f"  No FP32:         {no_fp32}")
    print(f"  Skipped:         {skipped - no_fp32}")
    print(f"  Errors:          {errors}")
    if total_original > 0:
        print(f"  Total size:      {fmt_bytes(total_original)}")
        print(f"  {'Estimated' if not args.apply else 'Actual'} savings: {fmt_bytes(total_saved)} "
              f"({total_saved/total_original*100:.1f}%)")
    print(f"{'=' * 70}")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
