"""Runtime patch for BiRefNet — detects local model paths on disk."""
import os

os.chdir("/opt/trellis")
path = "trellis2/pipelines/rembg/BiRefNet.py"
src = open(path).read()

marker = "elif os.path.exists(os.path.join(model_name, 'config.json'))"
if marker in src:
    print("[INIT] BiRefNet already patched")
else:
    # Replace entire else block (including the else: line)
    old = (
        "        else:\n"
        "            # Fallback to whatever was passed in (likely briaai/RMBG-2.0 which will fail if not logged in)\n"
        "            target_path = model_name\n"
        "            use_local = False"
    )
    new = (
        "        elif os.path.exists(os.path.join(model_name, 'config.json')):\n"
        "            target_path = model_name\n"
        "            use_local = True\n"
        "        else:\n"
        "            # Fallback to whatever was passed in (likely briaai/RMBG-2.0 which will fail if not logged in)\n"
        "            target_path = model_name\n"
        "            use_local = False"
    )
    src = src.replace(old, new)
    open(path, "w").write(src)
    print("[INIT] Patched BiRefNet local-path detection")
