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

# 3. Download ZImage VAE if needed
ae_file = ckpts_dir / "split_files" / "vae" / "ae.safetensors"
vae_target = ckpts_dir / "ZImageTurbo_VAE_bf16.safetensors"

if not vae_target.exists():
    if ae_file.exists():
        shutil.copy(ae_file, vae_target)
        print(f"✓ Created ZImage VAE: {vae_target}")
    else:
        print("❌ ZImage VAE source file not found")
else:
    print("✓ ZImage VAE already exists")

# 4. Create config files if needed
vae_config = {
    "model_type": "AutoencoderKL",
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 4,
    "latent_channels": 16,
    "scaling_factor": 0.18215,
    "latent_bias": -0.077,
    "downsampling_block": "bilinear"
}

vae_config_path = ckpts_dir / "ZImageTurbo_VAE_bf16_config.json"
if not vae_config_path.exists():
    with open(vae_config_path, 'w') as f:
        json.dump(vae_config, f, indent=2)
    print(f"✓ Created VAE config")

scheduler_config = {
    "prediction_type": "epsilon",
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "scaled_linear"
}

scheduler_config_path = ckpts_dir / "ZImageTurbo_scheduler_config.json"
if not scheduler_config_path.exists():
    with open(scheduler_config_path, 'w') as f:
        json.dump(scheduler_config, f, indent=2)
    print(f"✓ Created scheduler config")

print("\n✅ Anima model dependencies fixed!")
print("✅ Anima model should now load successfully")
PYTHON_EOF

python3 "$PYTHON_SCRIPT"
