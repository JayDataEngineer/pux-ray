#!/usr/bin/env python3
"""Prepare Qwen-Edit (non-2511) FP8 for vLLM-Omni.

The non-2511 Qwen-Edit model stores weights in ModelOpt FP8 format
(Float8_e4m3fn native). This script converts them to the compressed-tensors
FP8 weight-only format that vLLM-Omni's pipeline expects.

Strategy:
  1. Load non-2511 ModelOpt FP8 transformer weights
  2. Cast FP8→BF16 (native torch cast, no modelopt needed)
  3. Re-quantize to FP8 weight-only with per-tensor scales
  4. Use 2511 model's shared components (VAE, text_encoder, etc.)
  5. Add compressed-tensors quantization_config

Result: ~20GB DiT (FP8 weight-only) + CPU text encoder ≈ fits 24 GB

Usage:
  python3 scripts/prepare_qwen_edit_non2511_fp8.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# ─── Paths ──────────────────────────────────────────────────────────────────
MODELS_ROOT = Path(os.environ.get(
    "MODELS_ROOT",
    "/mnt/data/models" if os.path.exists("/mnt/data/models") else "/models",
))

# Non-2511 model parts
NON2511_TRANSFORMER = MODELS_ROOT / "native/qwen-edit-modelopt-fp8-transformer"
NON2511_GGUF_DIR = MODELS_ROOT / "native/qwen-edit-fp8-gguf"

# 2511 model (for shared components + reference config)
MODEL_2511 = MODELS_ROOT / "image-gen/qwen-image-edit/2511-fp8"

# Output directory
OUTPUT_DIR = MODELS_ROOT / "image-gen/qwen-edit-non2511-fp8"


def convert_transformer():
    """Convert ModelOpt FP8 weights to compressed-tensors FP8 weight-only."""
    if not NON2511_TRANSFORMER.exists():
        print(f"✗ Non-2511 transformer not found at {NON2511_TRANSFORMER}")
        sys.exit(1)
    if not MODEL_2511.exists():
        print(f"✗ 2511 model not found at {MODEL_2511}")
        print("  2511 model components are needed for VAE, text_encoder, etc.")
        print("  Run prepare_qwen_img_edit_fp8.py first to set up the 2511 model.")
        sys.exit(1)

    out_dir = OUTPUT_DIR / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Converting Qwen-Edit non-2511 ModelOpt FP8 → compressed-tensors FP8 weight-only...")
    print(f"  Source: {NON2511_TRANSFORMER}")
    print(f"  Target: {out_dir}")
    print()

    # Step 1: Load all weights, cast FP8→BF16
    t0 = time.perf_counter()
    all_tensors = {}
    for fname in sorted(os.listdir(NON2511_TRANSFORMER)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(NON2511_TRANSFORMER, fname)
        sd = load_file(fpath, device="cpu")
        for k in list(sd.keys()):
            if sd[k].dtype == torch.float8_e4m3fn:
                sd[k] = sd[k].to(torch.bfloat16)
        all_tensors.update(sd)
        del sd
        gc.collect()
    t1 = time.perf_counter()
    print(f"  Loaded & dequantized {len(all_tensors)} tensors in {t1-t0:.1f}s")

    # Separate weight tensors from bias/scale tensors
    scale_tensors = {k: v for k, v in all_tensors.items() if k.endswith("_scale") or k.endswith("input_scale")}
    bias_tensors = {k: v for k, v in all_tensors.items() if k.endswith(".bias")}
    weight_tensors = {k: v for k, v in all_tensors.items() 
                      if k.endswith(".weight") and k not in scale_tensors}

    print(f"  Weights: {len(weight_tensors)}, Biases: {len(bias_tensors)}, Scales: {len(scale_tensors)}")

    # Step 2: Create FP8 weight-only tensors with per-tensor scales
    # For each weight, quantize to FP8 and compute scale
    out_tensors = {}
    for k, w in weight_tensors.items():
        # Quantize: find max absolute value, compute scale
        absmax = w.abs().max()
        if absmax > 0:
            scale = absmax / 448.0  # FP8 max value for E4M3
            w_fp8 = (w / scale).to(torch.float8_e4m3fn)
        else:
            scale = torch.tensor(1.0, dtype=torch.bfloat16)
            w_fp8 = w.to(torch.float8_e4m3fn)
        
        out_tensors[k] = w_fp8
        # Add weight_scale (compressed-tensors format)
        scale_key = k.replace(".weight", ".weight_scale")
        out_tensors[scale_key] = scale.to(torch.bfloat16).reshape(1)
    
    # Add bias tensors as-is
    out_tensors.update(bias_tensors)

    # Step 3: Write as 3 shards (same count as original)
    out_keys = sorted(out_tensors.keys())
    shard_count = 3
    shard_size = (len(out_keys) + shard_count - 1) // shard_count
    weight_map = {}

    for i in range(shard_count):
        shard_keys = out_keys[i * shard_size:(i + 1) * shard_size]
        if not shard_keys:
            continue
        shard_name = f"diffusion_pytorch_model-{i + 1:05d}-of-{shard_count:05d}.safetensors"
        shard_dict = {k: out_tensors[k] for k in shard_keys}
        save_file(shard_dict, str(out_dir / shard_name), metadata=None)
        for k in shard_keys:
            weight_map[k] = shard_name
        print(f"  Wrote {shard_name}: {len(shard_dict)} tensors")
        del shard_dict
        gc.collect()

    # Write weight map index
    index = {"metadata": {}, "weight_map": weight_map}
    with open(out_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    del out_tensors, all_tensors
    gc.collect()

    # Step 4: Create config.json with compressed-tensors quantization_config
    # Use 2511 model's transformer config as reference
    ref_cfg_path = MODEL_2511 / "transformer" / "config.json"
    with open(ref_cfg_path) as f:
        cfg = json.load(f)

    # Add compressed-tensors config
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "dynamic": True,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "img_mod*", "txt_mod*", "norm*",
            "img_mlp*", "txt_mlp*",
        ],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    fp8_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1e9
    print(f"\n✓ Conversion complete: {fp8_gb:.1f} GB")
    print(f"  Config with compressed-tensors quantization added")


def assemble_model():
    """Assemble full model directory (shared components + converted transformer)."""
    print("\nAssembling model directory...")

    # Copy non-transformer components from 2511 model
    for item in os.listdir(MODEL_2511):
        if item in ("transformer", ".cache"):
            continue
        s = os.path.join(MODEL_2511, item)
        d = os.path.join(OUTPUT_DIR, item)
        if d.exists():
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True)
        else:
            shutil.copy2(s, d)
        print(f"  Copied {item}")

    # Ensure transformer exists
    if not (OUTPUT_DIR / "transformer").exists():
        print("  ✗ Transformer not found — did you run the conversion step?")
        sys.exit(1)

    total_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e9
    trans_gb = sum(f.stat().st_size for f in (OUTPUT_DIR / "transformer").glob("*.safetensors")) / 1e9
    print(f"\n✓ Model ready at {OUTPUT_DIR}")
    print(f"  Size: {total_gb:.1f} GB (transformer: {trans_gb:.1f} GB)")
    print(f"\n  VRAM budget on RTX 4090 (24GB):")
    print(f"    DiT (FP8 weight-only):  ~{trans_gb:.0f} GB")
    print(f"    VAE (with tiling):       ~0.3 GB")
    print(f"    Activations:              ~3 GB")
    print(f"    ────────────────────────────────────")
    print(f"    Total:                   ~{trans_gb + 3.3:.0f} GB  {'✓ fits!' if trans_gb + 3.3 < 24 else '✗ may OOM'}")


def list_status():
    """Print status of all model components."""
    print("Qwen-Edit non-2511 FP8 model status:")
    for name, path in [
        ("Non-2511 ModelOpt transformer", NON2511_TRANSFORMER),
        ("2511 model (shared components)", MODEL_2511),
        ("Output FP8 model directory", OUTPUT_DIR),
    ]:
        if path.exists():
            if path.is_dir():
                sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
                print(f"  {'✓' if sz > 0 else '✗'} {name}: {sz:.1f} GB")
            else:
                print(f"  ✗ {name}: not a directory")
        else:
            print(f"  ✗ {name}: NOT FOUND")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Qwen-Edit non-2511 FP8 for vLLM-Omni"
    )
    parser.add_argument("--convert", action="store_true", help="Only convert transformer")
    parser.add_argument("--assemble", action="store_true", help="Only assemble model")
    parser.add_argument("--status", action="store_true", help="Check status")
    args = parser.parse_args()

    if args.status:
        list_status()
        sys.exit(0)

    if args.convert:
        convert_transformer()
        assemble_model()
        sys.exit(0)

    if args.assemble:
        assemble_model()
        sys.exit(0)

    # Full pipeline
    convert_transformer()
    assemble_model()
    list_status()
