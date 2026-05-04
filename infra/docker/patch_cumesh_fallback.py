"""Runtime patch for mesh/base.py — add cumesh fallback for mesh operations.

When cumesh.CuMesh is not available, skip the operation (the mesh is still
valid, just unoptimized). This allows GLB export to proceed without cumesh.
"""
import os

os.chdir("/opt/trellis")
path = "trellis2/representations/mesh/base.py"

if not os.path.exists(path):
    print(f"[INIT] {path} not found, skipping cumesh fallback patch")
    exit(0)

src = open(path).read()

marker = "_cumesh_available"
if marker in src:
    print("[INIT] cumesh fallback already patched")
else:
    # Add cumesh availability check as a guard for each method
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

    if count > 0:
        open(path, "w").write(src)
        print(f"[INIT] Patched {count} mesh methods with cumesh fallback")
    else:
        print("[INIT] WARNING: no cumesh fallback targets found")
