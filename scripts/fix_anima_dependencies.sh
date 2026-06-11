#!/bin/bash
# Fix all missing anima model dependencies

set -e

PYTHON_SCRIPT="$1"
if [ -z "$PYTHON_SCRIPT" ]; then
    PYTHON_SCRIPT="/tmp/fix_anima_deps.py"
fi

cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Fix all missing dependencies for Anima model loading."""
import os
import sys
import json
import shutil
from pathlib import Path

wan2gp_root = Path("/home/user/Documents/programs/ray/opt/wan2gp")
ckpts_dir = wan2gp_root / "ckpts"

print("Fixing Anima model dependencies...")

# 1. Set correct checkpoints path for Wan2GP
sys.path.insert(0, str(wan2gp_root))
os.environ['WAN2GP_ROOT'] = str(wan2gp_root)

from shared.utils.files_locator import set_checkpoints_paths
set_checkpoints_paths([str(ckpts_dir)])

# 2. Create symlink for text encoder in main ckpts if not exists
te_src = ckpts_dir / "Qwen3-0.6B" / "qwen_3_06b_base.safetensors"
te_dst = ckpts_dir / "qwen_3_06b_base.safetensors"

if not te_dst.exists():
    te_dst.symlink_to(te_src)
    print("✓ Created text encoder symlink")
else:
    print("✓ Text encoder symlink already exists")

# 3. Download correct Qwen-Image VAE if needed
from huggingface_hub import hf_hub_download

qwen_vae_src = ckpts_dir / "split_files" / "vae" / "qwen_image_vae.safetensors"
qwen_vae_dst = ckpts_dir / "qwen_image_vae.safetensors"

if not qwen_vae_dst.exists():
    try:
        downloaded_path = hf_hub_download(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            filename="split_files/vae/qwen_image_vae.safetensors",
            local_dir=str(ckpts_dir),
            local_dir_use_symlinks=False
        )
        qwen_vae_src = Path(downloaded_path)
        qwen_vae_dst.symlink_to(qwen_vae_src)
        print(f"✓ Downloaded Qwen-Image VAE: {qwen_vae_dst}")
    except Exception as e:
        print(f"❌ Failed to download Qwen-Image VAE: {e}")
else:
    print("✓ Qwen-Image VAE symlink already exists")

# 4. Clean up incorrect ZImage VAE files
zimage_files = [
    "ZImageTurbo_VAE_bf16.safetensors",
    "ZImageTurbo_VAE_bf16_config.json",
    "ZImageTurbo_scheduler_config.json",
]

for f in zimage_files:
    path = ckpts_dir / f
    if path.exists():
        path.unlink()
        print(f"✓ Removed incorrect ZImage file: {f}")

print("\n✅ Anima model dependencies fixed!")
print("✅ Anima model should now load successfully")
PYTHON_EOF

python3 "$PYTHON_SCRIPT"
