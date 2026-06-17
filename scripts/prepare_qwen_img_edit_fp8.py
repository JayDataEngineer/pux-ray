#!/usr/bin/env python3
"""Prepare Qwen-Image-Edit-2511 FP8 diffusers model for vLLM-Omni.

Strategy:
  The full BF16 model is ~55GB — too large. Instead, we:
  1) Download ONLY the non-transformer parts from HF (text_encoder, VAE, configs)
     skipping the 40GB BF16 transformer shards.
  2) Convert the existing ComfyUI FP8 checkpoint (20GB, Float8_e4m3fn with
     per-tensor scales) into diffusers format for the transformer.
  3) Add quantization_config (compressed-tensors) so vLLM-Omni recognizes
     this as an FP8 weight-only model.

  Result: ~20GB DiT (FP8) + ~14GB text_encoder (BF16) + VAE + configs ≈ ~35GB on disk
  GPU VRAM: DiT 20GB + VAE 0.3GB + activations ~3GB ≈ ~23GB — fits on 24GB!
  Text encoder is CPU-offloaded (saves ~14GB VRAM).

Usage:
  python3 scripts/prepare_qwen_img_edit_fp8.py               # Full prepare
  python3 scripts/prepare_qwen_img_edit_fp8.py --download    # Download base model only
  python3 scripts/prepare_qwen_img_edit_fp8.py --convert     # Convert transformer only
  python3 scripts/prepare_qwen_img_edit_fp8.py --status      # Check model files status
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download, hf_hub_download

# ─── Paths ──────────────────────────────────────────────────────────────────
MODELS_ROOT = Path(os.environ.get(
    "MODELS_ROOT",
    "/mnt/data/models" if os.path.exists("/mnt/data/models") else "/models",
))
# The ComfyUI FP8 checkpoint (already downloaded)
COMFY_CKPT = MODELS_ROOT / "image-gen/comfyui/checkpoints" / \
    "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors"
# The target diffusers FP8 model directory
FP8_MODEL_DIR = MODELS_ROOT / "image-gen/qwen-image-edit/2511-fp8"
# BF16 model cache (for text_encoder, VAE, configs)
BF16_CACHE = MODELS_ROOT / "image-gen/qwen-image-edit/2511-bf16-cache"

HF_REPO = "Qwen/Qwen-Image-Edit-2511"


def download_base():
    """Download non-transformer parts from HuggingFace."""
    if BF16_CACHE.exists() and (BF16_CACHE / "model_index.json").exists():
        print(f"✓ Base model already cached at {BF16_CACHE}")
        return

    print(f"Downloading base model (text_encoder, VAE, configs) from {HF_REPO}...")
    print(f"  This is ~15GB (text_encoder 4 shards @ ~14GB + VAE + configs)")
    print(f"  Target: {BF16_CACHE}")
    print(f"  Skipping transformer/* (will use ComfyUI FP8 checkpoint)")
    print()

    BF16_CACHE.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(BF16_CACHE),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["transformer/*", "*.md", "*.git*"],
    )

    total_gb = sum(f.stat().st_size for f in BF16_CACHE.rglob("*") if f.is_file()) / 1e9
    print(f"\n✓ Downloaded base model: {total_gb:.1f} GB at {BF16_CACHE}")


def convert_transformer():
    """Convert ComfyUI FP8 checkpoint to diffusers FP8 format.

    The ComfyUI checkpoint stores DiT weights as Float8_e4m3fn with
    per-tensor BF16 scale factors (*_scale_weight). We remap these to
    the diffusers naming: *.weight → *.weight_scale.
    """
    if not COMFY_CKPT.exists():
        print(f"✗ ComfyUI FP8 checkpoint not found at {COMFY_CKPT}")
        print(f"  Expected: {COMFY_CKPT}")
        sys.exit(1)

    if not BF16_CACHE.exists():
        print(f"✗ Base model cache not found at {BF16_CACHE}")
        print("  Run with --download first, or download the base model manually.")
        sys.exit(1)

    out_dir = FP8_MODEL_DIR / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting ComfyUI FP8 checkpoint to diffusers format...")
    print(f"  Source: {COMFY_CKPT}")
    print(f"  Target: {out_dir}")
    print()

    # Read all keys from ComfyUI checkpoint
    with safe_open(str(COMFY_CKPT), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

    # Categorize
    weight_keys = [k for k in all_keys if k.endswith('.weight') and 'scale_weight' not in k]
    bias_keys = [k for k in all_keys if k.endswith('.bias')]
    scale_keys = [k for k in all_keys if 'scale_weight' in k]
    other_keys = [k for k in all_keys if k not in weight_keys and k not in bias_keys and k not in scale_keys]

    print(f"  {len(weight_keys)} weight tensors")
    print(f"  {len(bias_keys)} bias tensors")
    print(f"  {len(scale_keys)} scale tensors")
    print(f"  {len(other_keys)} other tensors")

    # Load all tensors
    t0 = time.perf_counter()
    tensors = {}
    with safe_open(str(COMFY_CKPT), framework="pt", device="cpu") as f:
        for k in all_keys:
            tensors[k] = f.get_tensor(k)
    print(f"  Loaded all tensors in {time.perf_counter() - t0:.1f}s")

    # Build output dict with diffusers naming
    out_data = {}

    # Copy ALL weights (FP8 Linear + non-FP8 RMSNorm etc.) as-is
    for k in weight_keys:
        out_data[k] = tensors[k]

    # Copy all biases as-is
    for k in bias_keys:
        out_data[k] = tensors[k]

    # Rename scale keys: ComfyUI *.scale_weight → diffusers *.weight_scale
    for k in scale_keys:
        new_key = k.replace('.scale_weight', '.weight_scale')
        out_data[new_key] = tensors[k]

    # Handle other keys (skip "scaled_fp8" metadata marker)
    for k in other_keys:
        if k != "scaled_fp8":
            out_data[k] = tensors[k]

    # Write as 5 shards (matching original BF16 sharding count)
    out_keys = sorted(out_data.keys())
    shard_size = (len(out_keys) + 4) // 5
    weight_map = {}

    for i in range(5):
        shard_keys = out_keys[i * shard_size:(i + 1) * shard_size]
        if not shard_keys:
            continue
        shard_name = f"diffusion_pytorch_model-{i + 1:05d}-of-00005.safetensors"
        shard_dict = {k: out_data[k] for k in shard_keys}
        save_file(shard_dict, str(out_dir / shard_name))
        for k in shard_keys:
            weight_map[k] = shard_name
        print(f"  Wrote {shard_name}: {len(shard_dict)} tensors")
        del shard_dict
        gc.collect()

    del out_data, tensors
    gc.collect()

    # Write weight map index
    index = {"metadata": {}, "weight_map": weight_map}
    with open(out_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy transformer config.json from BF16 cache or download
    bf16_transformer_cfg = BF16_CACHE / "transformer" / "config.json"
    if bf16_transformer_cfg.exists():
        with open(bf16_transformer_cfg) as f:
            cfg = json.load(f)
    else:
        cfg_path = hf_hub_download(HF_REPO, "transformer/config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

    # Add quantization_config for compressed-tensors (FP8 weight-only)
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
    print(f"\n✓ Transformer FP8 conversion complete: {fp8_gb:.1f} GB")
    print(f"  Config with compressed-tensors quantization added")


def assemble_model():
    """Assemble the full FP8 diffusers model directory.

    Copies text_encoder, VAE, and configs from the BF16 cache into
    the FP8 directory (alongside the transformer already written by
    convert_transformer).
    """
    print("\nAssembling FP8 model directory...")

    # Copy everything from BF16 cache EXCEPT transformer and .cache
    for item in BF16_CACHE.iterdir():
        if item.name in ("transformer", ".cache"):
            continue  # Keep our FP8 transformer; skip HF cache dir
        dst = FP8_MODEL_DIR / item.name
        if dst.exists():
            continue  # Already in place (from a previous partial run)
        if item.is_dir():
            shutil.copytree(item, dst)
            print(f"  Copied {item.name}/")
        else:
            shutil.copy2(item, dst)
            print(f"  Copied {item.name}")

    # Verify the FP8 transformer exists
    transformer_src = FP8_MODEL_DIR / "transformer"
    if not transformer_src.exists():
        print(f"  ✗ Transformer not found at {transformer_src}")
        print("    Did you run the conversion step first?")
        sys.exit(1)

    total_gb = sum(f.stat().st_size for f in FP8_MODEL_DIR.rglob("*") if f.is_file()) / 1e9
    trans_gb = sum(f.stat().st_size for f in transformer_src.glob("*.safetensors")) / 1e9
    print(f"\n✓ FP8 model ready at {FP8_MODEL_DIR}")
    print(f"  Size: {total_gb:.1f} GB (transformer: {trans_gb:.1f} GB)")
    print(f"\n  VRAM budget on RTX 4090 (24GB):")
    print(f"    DiT (FP8 weight-only):  ~{trans_gb:.0f} GB  (all 60 blocks on GPU)")
    print(f"    VAE (with tiling):       ~0.3 GB")
    print(f"    Activations:              ~3 GB")
    print(f"    ────────────────────────────────────")
    print(f"    Total:                   ~{trans_gb + 3.3:.0f} GB  {'✓ fits!' if trans_gb + 3.3 < 24 else '✗ may OOM'}")
    print(f"\n  Text encoder on CPU (saves ~14GB VRAM):")
    print(f"    Loaded for prefill, released after ⇒ 0 GB on GPU")


def list_status():
    """Print status of all model components."""
    print("Qwen-Image-Edit-2511 FP8 model status:")
    for name, path in [
        ("ComfyUI FP8 checkpoint", COMFY_CKPT),
        ("BF16 base cache", BF16_CACHE),
        ("FP8 model directory", FP8_MODEL_DIR),
    ]:
        if path.exists():
            if path.is_dir():
                sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
                print(f"  ✓ {name}: {sz:.1f} GB  ({path})")
            else:
                sz = path.stat().st_size / 1e9
                print(f"  ✓ {name}: {sz:.1f} GB  ({path})")
        else:
            print(f"  ✗ {name}: NOT FOUND  ({path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Qwen-Image-Edit-2511 FP8 diffusers model"
    )
    parser.add_argument("--download", action="store_true",
                        help="Only download base model from HF")
    parser.add_argument("--convert", action="store_true",
                        help="Only convert transformer from ComfyUI FP8")
    parser.add_argument("--status", action="store_true",
                        help="Check status of model components")
    args = parser.parse_args()

    if args.status:
        list_status()
        sys.exit(0)

    if args.download:
        download_base()
        sys.exit(0)

    if args.convert:
        if not BF16_CACHE.exists():
            print("❌ Base model not downloaded. Run without --convert first.")
            sys.exit(1)
        convert_transformer()
        assemble_model()
        sys.exit(0)

    # Full pipeline
    if not BF16_CACHE.exists():
        download_base()
    convert_transformer()
    assemble_model()
    list_status()
