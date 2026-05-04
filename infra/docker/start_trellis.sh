#!/bin/bash
# TRELLIS.2 startup script — applies runtime patches then starts the API.
cd /opt/trellis || exit 1

# Restore from git (idempotent)
git checkout HEAD -- trellis2/pipelines/rembg/BiRefNet.py 2>/dev/null || true
git checkout HEAD -- trellis2/modules/image_feature_extractor.py 2>/dev/null || true

# Patch to detect local model paths
python3 /opt/patch_birefnet_runtime.py || true
python3 /opt/patch_dinov3_runtime.py || true

exec python api_spz/main_api.py --host 0.0.0.0 --port 8000
