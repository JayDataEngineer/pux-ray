"""Patches Python source files in the trellis2 tree to make cumesh and flex_gemm
imports lazy. These compiled CUDA extensions are optional — the API uses
nvdiffrast for GLB export as fallback when GPU extensions are unavailable.
"""
from pathlib import Path

base = Path("/opt/trellis")


def lazy_import(path: Path, old: str, new: str):
    src = path.read_text()
    if old in src:
        src = src.replace(old, new)
        path.write_text(src)
        print(f"  patched: {path.relative_to(base)}")
    else:
        print(f"  skip (not found): {path.relative_to(base)}")


PATCHES = [
    # o-voxel postprocess.py
    (
        "o-voxel/o_voxel/postprocess.py",
        "from flex_gemm.ops.grid_sample import grid_sample_3d",
        "try:\n    from flex_gemm.ops.grid_sample import grid_sample_3d\nexcept ImportError:\n    grid_sample_3d = None",
    ),
    (
        "o-voxel/o_voxel/postprocess.py",
        "import cumesh",
        "try:\n    import cumesh\nexcept ImportError:\n    cumesh = None",
    ),
]

print("Applying lazy-import patches...")
for path_rel, old, new in PATCHES:
    p = base / path_rel
    if p.exists():
        lazy_import(p, old, new)
    else:
        print(f"  SKIP (file not found): {path_rel}")

print("Done.")
