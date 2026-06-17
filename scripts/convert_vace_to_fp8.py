#!/usr/bin/env python3
"""Convert a BF16 Wan2.1 VACE diffusers checkpoint to direct-cast FP8.

Produces a model directory that loads via vLLM-Omni with `--quantization fp8`
and works correctly with the FP8 weight-only patch in
pipeline_wan2_2_vace_patch.py.

"Direct-cast" means we simply cast each transformer weight tensor to
torch.float8_e4m3fn with NO per-tensor scaling. The weight values are
preserved as-is (they live in the normal NN range ±0.5) — only the storage
dtype changes from BF16 (2 bytes) to FP8 (1 byte), cutting VRAM in half.

This works because pipeline_wan2_2_vace_patch.py overrides the FP8 apply
method to cast FP8 → BF16 directly (no scale multiplication), since the
stored FP8 values ARE the actual weights.

Usage:
  python convert_vace_to_fp8.py \\
      --src /path/to/wan2.1-vace-14b-diffusers \\
      --dst /path/to/wan2.1-vace-14b-fp8-diffusers

  # For the lightning variant, first merge the LightX2V LoRA into the BF16
  # checkpoint, then run this script on the merged model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}")
    dst.mkdir(parents=True)

    print(f"Converting BF16 → FP8 (direct cast)\n  src: {src}\n  dst: {dst}")

    # Copy everything except the transformer weights verbatim
    for item in src.iterdir():
        if item.name == "transformer":
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
            print(f"  copied dir:  {item.name}/")
        else:
            shutil.copy2(item, dst / item.name)
            print(f"  copied file: {item.name}")

    # Process transformer shard by shard
    src_tx = src / "transformer"
    dst_tx = dst / "transformer"
    dst_tx.mkdir()

    # Copy config files
    for cfg in src_tx.glob("*.json"):
        shutil.copy2(cfg, dst_tx / cfg.name)
        print(f"  copied config: transformer/{cfg.name}")

    # Patch config.json: mark as FP8-serialized, no ignored_layers needed
    config_path = dst_tx / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "fp8",
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "dynamic",
        }
        with config_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  patched:    transformer/config.json (quantization_config=fp8)")

    # Convert each shard
    shards = sorted(src_tx.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        # Single-file checkpoint
        shards = [src_tx / "diffusion_pytorch_model.safetensors"]

    for shard in shards:
        print(f"  converting: transformer/{shard.name}...", end=" ", flush=True)
        state = load_file(str(shard))
        converted: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            # Only cast floating-point tensors in the transformer body.
            # Skip norm scales, biases, and any integer/long tensors.
            if (
                tensor.is_floating_point()
                and tensor.dtype != torch.float8_e4m3fn
                and tensor.numel() > 1
            ):
                converted[key] = tensor.to(torch.float8_e4m3fn)
            else:
                converted[key] = tensor
        save_file(converted, str(dst_tx / shard.name))
        # Print size delta
        src_size = shard.stat().st_size
        dst_size = (dst_tx / shard.name).stat().st_size
        print(
            f"{src_size / 2**30:.2f}GB → {dst_size / 2**30:.2f}GB "
            f"({len(converted)} tensors)"
        )

    # Copy index file if present
    index = src_tx / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        shutil.copy2(index, dst_tx / index.name)

    print(f"\n✓ Done. Use with run_omni_14b.sh {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Source BF16 diffusers model directory")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Destination FP8 model directory (must not exist)")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
