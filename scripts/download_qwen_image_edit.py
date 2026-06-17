#!/usr/bin/env python3
"""Download Qwen-Image-Edit-2511 diffusers model from HuggingFace.

The model uses QwenImageEditPlusPipeline (20B MMDiT) and requires the full
diffusers layout. On RTX 4090 (24GB), serve with layerwise offload:

  vllm serve Qwen/Qwen-Image-Edit-2511 --omni \\
    --port 8092 \\
    --vae-use-slicing --vae-use-tiling \\
    --cache-backend cache_dit \\
    --enable-layerwise-offload

Usage:
  python3 scripts/download_qwen_image_edit.py             # Download full model
  python3 scripts/download_qwen_image_edit.py --list       # List available models
  python3 scripts/download_qwen_image_edit.py --model 2511 # Specific variant
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub not installed. Install with:\n"
        "  pip install huggingface_hub\n"
        "  # or: uv pip install huggingface_hub"
    )

MODELS_ROOT = os.environ.get(
    "MODELS_ROOT",
    "/mnt/data/models" if os.path.exists("/mnt/data/models") else "/models",
)

QWEN_IMAGE_EDIT_MODELS = {
    "base": (    # Qwen-Image-Edit (original, single-image)
        "Qwen/Qwen-Image-Edit",
        "image-gen/qwen-image-edit/base",
        40.0,
    ),
    "2511": (    # Qwen-Image-Edit-2511 (latest, multi-image, built-in LoRA)
        "Qwen/Qwen-Image-Edit-2511",
        "image-gen/qwen-image-edit/2511",
        40.0,
    ),
}


def download(variant: str) -> None:
    if variant not in QWEN_IMAGE_EDIT_MODELS:
        print(f"Unknown variant: {variant}")
        print(f"Available: {list(QWEN_IMAGE_EDIT_MODELS.keys())}")
        sys.exit(1)

    repo_id, rel_path, approx_gb = QWEN_IMAGE_EDIT_MODELS[variant]
    target = Path(MODELS_ROOT) / rel_path

    if target.exists() and any(target.iterdir()):
        print(f"✓ Model already exists at {target}")
        return

    print(f"Downloading {repo_id} (~{approx_gb} GB BF16)...")
    print(f"  Target: {target}")
    print(f"  This will take a while on slow connections.")
    print()

    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.md", "*.git*"],
    )

    # Calculate actual size
    total_bytes = sum(
        f.stat().st_size for f in target.rglob("*") if f.is_file()
    )
    total_gb = total_bytes / (1024**3)
    print(f"\n✓ Downloaded {repo_id} to {target}")
    print(f"  Size: {total_gb:.1f} GB")
    print(f"  Files: {len(list(target.rglob('*')))}")


def list_models() -> None:
    print("Available Qwen-Image-Edit models:")
    for name, (repo_id, rel_path, approx_gb) in sorted(QWEN_IMAGE_EDIT_MODELS.items()):
        target = Path(MODELS_ROOT) / rel_path
        status = "✓" if target.exists() and any(target.iterdir()) else " "
        print(f"  [{status}] {name:8s} → {repo_id:40s}  ~{approx_gb:.0f}GB  → {rel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen-Image-Edit diffusers model")
    parser.add_argument("--model", default="2511", help="Model variant (base, 2511)")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        list_models()
    else:
        download(args.model)
