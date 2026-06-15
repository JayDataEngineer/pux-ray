#!/usr/bin/env python3
"""Download all MOSS audio models from HuggingFace.

Usage:
  python3 scripts/download_moss_models.py          # Download all
  python3 scripts/download_moss_models.py --only v2  # Download only v2
"""
import argparse
import os
import sys
from huggingface_hub import snapshot_download

# Model definitions: name → (hf_repo, local_path, approx_size_gb)
MOSS_MODELS = {
    "v2": ("OpenMOSS-Team/MOSS-SoundEffect-v2.0", "/models/audio/moss-soundeffect-v2", 11.3),
    "nano": ("OpenMOSS-Team/MOSS-TTS-Nano-100M", "/models/audio/moss-tts-nano", 0.25),
    "sfx-v1": ("OpenMOSS-Team/MOSS-SoundEffect", "/models/audio/moss-soundeffect", 33.5),
    "tts": ("OpenMOSS-Team/MOSS-TTS", "/models/audio/moss-tts", 33.5),
    "ttsd": ("OpenMOSS-Team/MOSS-TTSD-v1.0", "/models/audio/moss-ttsd", 33.5),
    "voicegen": ("OpenMOSS-Team/MOSS-VoiceGenerator", "/models/audio/moss-voicegenerator", 7.0),
    "realtime": ("OpenMOSS-Team/MOSS-TTS-Realtime", "/models/audio/moss-tts-realtime", 4.7),
    "local-tx": ("OpenMOSS-Team/MOSS-TTS-Local-Transformer", "/models/audio/moss-tts-local-transformer", 6.2),
}


def download(name: str, repo: str, path: str, size: float):
    if os.path.exists(os.path.join(path, "model_index.json")) or \
       os.path.exists(os.path.join(path, "config.json")):
        print(f"  SKIP {name}: already at {path}")
        return
    print(f"  Downloading {name} ({size}GB): {repo} → {path}")
    snapshot_download(repo, local_dir=path)
    print(f"  DONE {name}")


def main():
    parser = argparse.ArgumentParser(description="Download MOSS audio models")
    parser.add_argument("--only", nargs="*", help="Only download specific models")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        for name, (repo, path, size) in MOSS_MODELS.items():
            exists = "✅" if os.path.exists(os.path.join(path, "model_index.json")) else "❌"
            print(f"  {exists} {name:12s} {size:>6.1f}GB  {repo}")
        return

    targets = args.only if args.only else list(MOSS_MODELS.keys())
    for name in targets:
        if name not in MOSS_MODELS:
            print(f"  UNKNOWN: {name}. Available: {list(MOSS_MODELS.keys())}")
            continue
        repo, path, size = MOSS_MODELS[name]
        try:
            download(name, repo, path, size)
        except Exception as e:
            print(f"  FAILED {name}: {e}")

    print("\nDownload complete. Available models:")
    for name, (repo, path, size) in MOSS_MODELS.items():
        exists = "✅" if os.path.exists(os.path.join(path, "model_index.json")) or \
                        os.path.exists(os.path.join(path, "config.json")) else "❌"
        print(f"  {exists} {name:12s} → {path}")


if __name__ == "__main__":
    main()
