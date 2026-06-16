#!/usr/bin/env python3
"""Download Wan VACE video models from ModelScope.

VACE requires the modular DiffSynth-Studio file layout (not diffusers format).
Models come from alibaba-pai and Wan-AI on ModelScope — HF mirrors exist but
lag behind. We pull from ModelScope to match DiffSynth's example pipelines.

Usage:
  python3 scripts/download_vace_models.py             # Download default (fun-a14b)
  python3 scripts/download_vace_models.py --only fun-a14b tokenizer
  python3 scripts/download_vace_models.py --list
"""
import argparse
import os
import sys
from pathlib import Path

# ModelScope is the canonical source for Wan VACE — HF repos lag.
try:
    from modelscope import snapshot_download
except ImportError:
    sys.exit(
        "modelscope not installed. Install with:\n"
        "  pip install modelscope\n"
        "  # or: uv pip install modelscope"
    )

# Model definitions: short-name → (modelscope_repo, local_path, approx_size_gb, file_patterns)
# local_path is the in-container mount; registry uses relative path under /models/.
VACE_MODELS = {
    # Primary production target — Wan2.2 VACE-Fun A14B modular
    "fun-a14b": (
        "alibaba-pai/Wan2.2-VACE-Fun-A14B",
        "/models/video/wan-vace-fun-a14b",
        34.0,
        [
            "high_noise_model/diffusion_pytorch_model*.safetensors",
            "low_noise_model/diffusion_pytorch_model*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "configs/*.json",
        ],
    ),
    # Monolithic 14B (alternative if MoE expert routing causes artifacts)
    "vace-14b": (
        "Wan-AI/Wan2.1-VACE-14B",
        "/models/video/wan-vace-14b",
        56.0,
        ["*.safetensors", "*.pth", "configs/*.json"],
    ),
    # Efficiency tier — fast prototyping on 8GB cards
    "vace-1.3b": (
        "Wan-AI/Wan2.1-VACE-1.3B",
        "/models/video/wan-vace-1.3b",
        5.0,
        ["*.safetensors", "*.pth", "configs/*.json"],
    ),
    # Tokenizer — shared umt5-xxl, pulled from Wan2.1-T2V-1.3B (matches DiffSynth example)
    "tokenizer": (
        "Wan-AI/Wan2.1-T2V-1.3B",
        "/models/video/wan-vace-tokenizer",
        0.05,
        ["google/umt5-xxl/*"],
    ),
}

# Default download set — fun-a14b bundle + tokenizer
DEFAULT_SET = ["fun-a14b", "tokenizer"]


def download(name: str, repo: str, path: str, size: float, patterns: list[str]):
    # Heuristic: already downloaded if any safetensors or .pth exists in tree
    existing = list(Path(path).rglob("*.safetensors")) + list(Path(path).rglob("*.pth"))
    if existing:
        print(f"  SKIP {name}: {len(existing)} weight file(s) already at {path}")
        return
    print(f"  Downloading {name} (~{size}GB): {repo} → {path}")
    print(f"    patterns: {patterns}")
    snapshot_download(
        repo,
        local_dir=path,
        allow_file_patterns=patterns,
    )
    print(f"  DONE {name}")


def main():
    parser = argparse.ArgumentParser(description="Download Wan VACE video models from ModelScope")
    parser.add_argument(
        "--only", nargs="*",
        help=f"Only download specific models. Available: {list(VACE_MODELS.keys())}",
    )
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        for name, (repo, path, size, _) in VACE_MODELS.items():
            existing = list(Path(path).rglob("*.safetensors")) + list(Path(path).rglob("*.pth"))
            mark = "✅" if existing else "❌"
            in_default = " (default)" if name in DEFAULT_SET else ""
            print(f"  {mark} {name:12s} {size:>6.1f}GB  {repo}{in_default}")
        return

    targets = args.only if args.only else DEFAULT_SET
    for name in targets:
        if name not in VACE_MODELS:
            print(f"  UNKNOWN: {name}. Available: {list(VACE_MODELS.keys())}")
            continue
        repo, path, size, patterns = VACE_MODELS[name]
        try:
            download(name, repo, path, size, patterns)
        except Exception as e:
            print(f"  FAILED {name}: {e}")

    print("\nDownload complete. VACE models:")
    for name, (repo, path, size, _) in VACE_MODELS.items():
        existing = list(Path(path).rglob("*.safetensors")) + list(Path(path).rglob("*.pth"))
        mark = "✅" if existing else "❌"
        print(f"  {mark} {name:12s} → {path}")


if __name__ == "__main__":
    main()
