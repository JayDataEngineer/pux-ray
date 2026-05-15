"""Build-time patch for TRELLIS attention backends.

Adds SageAttention varlen and SDPA as sparse attention backends,
matching what's available in the Wan2GP Docker image (sageattention,
torch SDPA — no standalone flash_attn or xformers packages).

Copied from vendor/trellis2 with minimal additions.
"""
import site
import shutil
from pathlib import Path

SITE = Path(site.getsitepackages()[0])
TRELLIS = SITE / "trellis2"

PATCHES = Path(__file__).parent

files = {
    "modules/sparse/config.py": "trellis_sparse_config.py",
    "modules/sparse/attention/full_attn.py": "trellis_sparse_full_attn.py",
}

for rel, src_name in files.items():
    dst = TRELLIS / rel
    src = PATCHES / src_name
    if not dst.exists():
        print(f"[SKIP] {dst} not found (trellis2 not installed?)")
        continue
    shutil.copy2(src, dst)
    print(f"[PATCH] {src_name} -> {dst}")

print("TRELLIS attention backend patch applied.")
