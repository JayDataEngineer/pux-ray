#!/usr/bin/env python3
"""Build a Wan2.1 VACE 14B FP8 Lightning checkpoint.

STATUS (2026-06-17): PENDING — no compatible 4-step distillation source
exists for Wan2.1-VACE-14B in the diffusers format that vLLM-Omni expects.

Why this script exists:
  The LightX2V distill LoRA (hf://lightx2v/Wan2.2-Distill-Loras) targets
  Wan2.2-I2V-A14B, which is a DIFFERENT architecture from Wan2.1-VACE-14B:
    - Wan2.2-I2V has no VACE conditioning blocks (vace_blocks.0..7)
    - Wan2.2-I2V uses a single DiT stream; VACE adds a parallel
      conditioning stream that intersects at 8 layers
  Merging the Wan2.2 LoRA into Wan2.1-VACE produced washed-out output
  (std~12 vs ~72 for base) because the LoRA only perturbed the
  scale_shift_table (the only shared parameter shape), destroying the
  time-modulation dynamics.

  This script is kept as IaC so that WHEN a Wan2.1-VACE-specific
  distillation LoRA becomes available, the lightning checkpoint can be
  rebuilt reproducibly in one command.

Prerequisites when a compatible LoRA is available:
  1. BF16 base model at WAN_BF16_PATH (default downloads from HF)
  2. LoRA weights at LORA_PATH (set --lora-path)
  3. PEFT installed: pip install peft

Usage (when compatible LoRA is available):
  python build_lightning_model.py \\
      --bf16-path /path/to/wan2.1-vace-14b-diffusers \\
      --lora-path /path/to/wan2.1-vace-distill-lora.safetensors \\
      --lora-strength 1.0 \\
      --dst /mnt/data/models/video/wan2.1-vace-14b-fp8-lightning

  # Then launch with:
  ./scripts/run_omni_14b_lightning.sh
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def check_peft() -> None:
    try:
        import peft  # noqa: F401
    except ImportError:
        print(
            "ERROR: peft is required for LoRA merging. Install with:\n"
            "  pip install peft",
            file=sys.stderr,
        )
        sys.exit(2)


def merge_lora_into_transformer(
    transformer_dir: Path,
    lora_path: Path,
    strength: float,
) -> None:
    """Merge a LoRA into all transformer shards in-place.

    Loads each shard, applies the LoRA to matching keys, saves the merged
    weights back to the same shard.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from peft import LoraConfig, PeftModel

    # Build a minimal wrapper around the state dict for LoRA application.
    # PEFT's merge_and_unload expects a torch.nn.Module, so we apply LoRA
    # weight math directly: W_merged = W_base + strength * (B @ A)
    print(f"Merging LoRA {lora_path} at strength={strength} into {transformer_dir}")

    # Load LoRA weights directly
    lora_state = load_file(str(lora_path))
    # Group by target layer: { "blocks.0.attn1.to_q": {"lora_A": ..., "lora_B": ...} }
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_state.items():
        # Typical key: base_model.model.blocks.0.attn1.to_q.lora_A.weight
        if "lora_A" in k:
            base = k.replace(".lora_A.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(base, {})["A"] = v
        elif "lora_B" in k:
            base = k.replace(".lora_B.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(base, {})["B"] = v

    print(f"  LoRA targets {len(lora_pairs)} layers")

    # Process each shard
    shards = sorted(transformer_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        shards = [transformer_dir / "diffusion_pytorch_model.safetensors"]

    merged_count = 0
    for shard in shards:
        state = load_file(str(shard))
        changed = False
        for key in list(state.keys()):
            if key in lora_pairs and "weight" in key:
                A = lora_pairs[key]["A"].to(state[key].dtype)
                B = lora_pairs[key]["B"].to(state[key].dtype)
                # LoRA merge: W_new = W + (strength * B @ A)
                delta = (B @ A).reshape(state[key].shape)
                state[key] = state[key] + strength * delta
                merged_count += 1
                changed = True
        if changed:
            save_file(state, str(shard))
            print(f"  merged into {shard.name}")
    print(f"  Merged {merged_count} layers total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-path", type=Path, required=True,
        help="Source BF16 diffusers model directory (will be copied)",
    )
    parser.add_argument(
        "--lora-path", type=Path, required=True,
        help="Path to compatible Wan2.1-VACE distillation LoRA (.safetensors)",
    )
    parser.add_argument(
        "--lora-strength", type=float, default=1.0,
        help="LoRA merge strength (default 1.0)",
    )
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Destination directory for FP8 lightning model",
    )
    args = parser.parse_args()

    if args.dst.exists():
        raise SystemExit(f"Destination already exists: {args.dst}")

    check_peft()

    # Stage 1: copy BF16 base to a staging dir
    staging = args.dst.with_suffix(".staging")
    if staging.exists():
        shutil.rmtree(staging)
    print(f"\n[1/3] Copying BF16 base -> {staging}")
    shutil.copytree(args.bf16_path, staging)

    # Stage 2: merge LoRA into the staging transformer
    print(f"\n[2/3] Merging LoRA")
    merge_lora_into_transformer(
        staging / "transformer", args.lora_path, args.lora_strength
    )

    # Stage 3: convert to FP8 via the companion script
    print(f"\n[3/3] Converting BF16 -> direct-cast FP8")
    from convert_vace_to_fp8 import convert
    convert(staging, args.dst)
    shutil.rmtree(staging)

    print(f"\n✓ Lightning model built at {args.dst}")
    print(f"  Launch with: ./scripts/run_omni_14b_lightning.sh")


if __name__ == "__main__":
    main()
