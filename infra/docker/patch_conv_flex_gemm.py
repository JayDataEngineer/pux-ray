"""Runtime patch for conv_flex_gemm.py — add needs_grad=False to _compute_neighbor_cache calls.

The flex_gemm SubMConv3dFunction._compute_neighbor_cache() now requires a 5th
`needs_grad: bool` argument. The trellis2 conv_flex_gemm code only passes 4 args.
"""
import os

os.chdir("/opt/trellis")
path = "trellis2/modules/sparse/conv/conv_flex_gemm.py"

if not os.path.exists(path):
    print(f"[INIT] {path} not found, skipping conv_flex_gemm patch")
    exit(0)

src = open(path).read()

# Both call sites have the same pattern ending with `self.dilation\n            )`
# We add `, False` before the closing paren, but only for _compute_neighbor_cache calls.
marker = "self.dilation,\n                False"
if marker in src:
    print("[INIT] conv_flex_gemm already patched (needs_grad=False)")
else:
    old = "self.dilation\n            )"
    new = "self.dilation,\n                False\n            )"
    count = src.count(old)
    if count == 0:
        print("[INIT] WARNING: conv_flex_gemm patch target not found")
    else:
        src = src.replace(old, new)
        open(path, "w").write(src)
        print(f"[INIT] Patched conv_flex_gemm needs_grad=False ({count} call sites)")
