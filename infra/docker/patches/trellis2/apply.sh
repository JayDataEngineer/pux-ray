#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# apply.sh — apply vendor/trellis2 patches needed for runtime
# ═════════════════════════════════════════════════════════════════════════════
# Patches:
#   1. conv_flex_gemm.py   — adds needs_grad=False to _compute_neighbor_cache calls
#                            (flex_gemm API requires the 5th arg; trellis2 only passes 4)
#   2. mesh/base.py        — guards cumesh operations so missing CuMesh doesn't crash
#                            (lets GLB export proceed without cumesh.CuMesh)
#
# Idempotent — safe to run multiple times. Patches are applied to vendor/trellis2/
# in-place; if vendor/ is read-only, this script will fail (caller must chmod +w).
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VENDOR="${1:-/opt/trellis/trellis2}"
echo "[trellis2-patches] Applying to ${VENDOR}"

# ─── Patch 1: conv_flex_gemm.py — needs_grad=False ───────────────────────────
FILE="${VENDOR}/modules/sparse/conv/conv_flex_gemm.py"
if [ ! -f "$FILE" ]; then
    echo "[trellis2-patches] SKIP: $FILE not found"
else
    python3 - << PYEOF
path = "${FILE}"
src = open(path).read()
# Detect already-patched state by the trailing comma we add
if "self.dilation,\n                False\n            )" in src:
    print("[trellis2-patches] conv_flex_gemm already patched")
else:
    old = "self.dilation\n            )"
    new = "self.dilation,\n                False\n            )"
    count = src.count(old)
    if count == 0:
        print("[trellis2-patches] WARN: conv_flex_gemm patch target not found")
    else:
        src = src.replace(old, new)
        open(path, "w").write(src)
        print(f"[trellis2-patches] conv_flex_gemm patched ({count} sites)")
PYEOF
fi

# ─── Patch 2: mesh/base.py — cumesh fallback guard ──────────────────────────
FILE="${VENDOR}/representations/mesh/base.py"
if [ ! -f "$FILE" ]; then
    echo "[trellis2-patches] SKIP: $FILE not found"
else
    python3 - << PYEOF
path = "${FILE}"
src = open(path).read()
if "if not hasattr(cumesh, 'CuMesh')" in src:
    print("[trellis2-patches] mesh/base.py already patched")
else:
    replacements = [
        (
            "    def fill_holes(self, max_hole_perimeter=3e-2):\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
            "    def fill_holes(self, max_hole_perimeter=3e-2):\n        if not hasattr(cumesh, 'CuMesh'):\n            return\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
        ),
        (
            "    def remove_faces(self, face_mask: torch.Tensor):\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
            "    def remove_faces(self, face_mask: torch.Tensor):\n        if not hasattr(cumesh, 'CuMesh'):\n            return\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
        ),
        (
            "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
            "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n        if not hasattr(cumesh, 'CuMesh'):\n            return\n        vertices = self.vertices.cuda()\n        faces = self.faces.cuda()\n        \n        mesh = cumesh.CuMesh()",
        ),
    ]
    count = 0
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            count += 1
    open(path, "w").write(src)
    print(f"[trellis2-patches] mesh/base.py patched ({count}/3 methods)")
PYEOF
fi

echo "[trellis2-patches] done"
