"""Patch trellis2 source files to make cumesh and flex_gemm imports lazy.
Runs from within /opt/trellis.
"""
import os

os.chdir("/opt/trellis")

for path in [
    "trellis2/representations/mesh/base.py",
    "trellis2/representations/mesh/pbr.py",
]:
    if os.path.exists(path):
        src = open(path).read()
        src = src.replace(
            "import cumesh",
            "try:\n    import cumesh\nexcept ImportError:\n    cumesh = None",
        )
        src = src.replace(
            "from flex_gemm.ops.grid_sample import grid_sample_3d",
            "try:\n    from flex_gemm.ops.grid_sample import grid_sample_3d\nexcept ImportError:\n    grid_sample_3d = None",
        )
        with open(path, "w") as f:
            f.write(src)
        print(f"patched {path}")

print("trellis2 mesh patches done")
