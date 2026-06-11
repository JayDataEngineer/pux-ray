#!/bin/bash
# Download missing Qwen3-0.6B dependencies for Anima model
# This script ensures all required tokenizer files are present

set -e

PYTHON_SCRIPT="$1"
if [ -z "$PYTHON_SCRIPT" ]; then
    PYTHON_SCRIPT="/tmp/download_anima_deps.py"
fi

cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Download Qwen3-0.6B dependencies for Anima model."""
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

qwen_dir = Path("/home/user/Documents/programs/ray/opt/wan2gp/ckpts/Qwen3-0.6B")
qwen_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Qwen3-0.6B dependencies for Anima...")

# Download missing files
files_to_download = [
    "merges.txt",
    "generation_config.json"
]

for filename in files_to_download:
    target_file = qwen_dir / filename
    if target_file.exists():
        print(f"✓ {filename} already exists")
        continue
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="Qwen/Qwen3-0.6B",
            filename=filename,
            local_dir=qwen_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

# Create special_tokens_map.json if needed
special_tokens_path = qwen_dir / "special_tokens_map.json"
if not special_tokens_path.exists():
    config_path = qwen_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        special_tokens_map = {
            "vocab_size": config.get("vocab_size", 151936),
            "special_tokens": {
                "eos_token": "<|endoftext|>",
                "bos_token": "<|beginningoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|unknown|>"
            }
        }
        
        with open(special_tokens_path, 'w') as f:
            json.dump(special_tokens_map, f, indent=2)
        print("✓ Created special_tokens_map.json")

print("Qwen3-0.6B dependencies ready!")
PYTHON_EOF

python3 "$PYTHON_SCRIPT"
