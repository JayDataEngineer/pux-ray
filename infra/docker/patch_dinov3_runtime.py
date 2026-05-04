"""Runtime patch for DinoV3FeatureExtractor — local paths + transformers 5.x compat."""
import os

os.chdir("/opt/trellis")
path = "trellis2/modules/image_feature_extractor.py"
src = open(path).read()

# Patch 1: local model path detection
marker1 = "elif os.path.exists(os.path.join(model_name, 'config.json'))"
if marker1 not in src:
    old = (
        "        else:\n"
        "            target_path = model_name\n"
        "            use_local = False"
    )
    new = (
        "        elif os.path.exists(os.path.join(model_name, 'config.json')):\n"
        "            target_path = model_name\n"
        "            use_local = True\n"
        "        else:\n"
        "            target_path = model_name\n"
        "            use_local = False"
    )
    src = src.replace(old, new)
    open(path, "w").write(src)
    src = open(path).read()  # re-read for patch 2
    print("[INIT] Patched DinoV3FeatureExtractor local-path detection")

# Patch 2: transformers 5.x compat — self.model.layer → self.model.model.layer
marker2 = "self.model.model.layer"
if marker2 not in src:
    src = src.replace(
        "enumerate(self.model.layer)",
        "enumerate(self.model.model.layer)",
    )
    open(path, "w").write(src)
    print("[INIT] Patched DinoV3FeatureExtractor layer ref (transformers 5.x compat)")

if marker1 in open(path).read() and marker2 in open(path).read():
    print("[INIT] DinoV3FeatureExtractor patches verified")
