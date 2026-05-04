"""Patch state_manage.py for local_files_only=True to avoid HF gated-repo online check."""
import os
os.chdir("/opt/trellis")
path = "api_spz/core/state_manage.py"
src = open(path).read()
old = "Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')"
new = "Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B', local_files_only=True)"
src = src.replace(old, new)
open(path, "w").write(src)
print("patched state_manage.py with local_files_only=True")
