"""Create stub flex_gemm and cumesh packages so API starts without GPU extensions.
The real flex_gemm has triton 3.x incompatibility and cumesh won't compile.
These stubs provide the minimal API surface needed for pipeline initialization.
"""
import os

site = "/usr/local/lib/python3.11/dist-packages"

# flex_gemm stub
os.makedirs(f"{site}/flex_gemm/ops", exist_ok=True)

files = {
    "flex_gemm/__init__.py": "",
    "flex_gemm/ops/__init__.py": "from .spconv import *\nfrom .grid_sample import *\n",
    "flex_gemm/ops/spconv.py": """
class Algorithm:
    EXPLICIT_GEMM = "explicit_gemm"
ALGORITHM = Algorithm.EXPLICIT_GEMM
def sparse_submanifold_conv3d(feats, indices, *a, **kw):
    return feats, indices
def sparse_conv3d(feats, indices, *a, **kw):
    return feats, indices
""",
    "flex_gemm/ops/grid_sample.py": """def grid_sample_3d(*a, **kw):
    raise NotImplementedError("grid_sample_3d stub")
""",
}

for path, content in files.items():
    full = os.path.join(site, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content.lstrip("\n"))

# cumesh stub
os.makedirs(f"{site}/cumesh", exist_ok=True)
open(f"{site}/cumesh/__init__.py", "w").write("")

print("Stub packages created")
