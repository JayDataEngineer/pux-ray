"""Patch BiRefNet to detect when model_name is a valid local path on disk.
Idempotent — restores from git before patching to avoid double-patch issues.
"""
import os, subprocess
os.chdir("/opt/trellis")

path = "trellis2/pipelines/rembg/BiRefNet.py"

# Restore original from git HEAD
subprocess.run(["git", "checkout", "HEAD", "--", path], check=True, capture_output=True)

src = open(path).read()

old = """            target_path = model_name
            use_local = False"""

new = """        elif os.path.exists(os.path.join(model_name, 'config.json')):
            target_path = model_name
            use_local = True
        else:
            target_path = model_name
            use_local = False"""

src = src.replace(old, new)
open(path, "w").write(src)
print("Patched BiRefNet local-path detection")
