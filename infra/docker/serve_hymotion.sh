#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_hymotion.sh — HY-Motion 1.0 / 1.0-Lite text-to-3D human motion
# ═════════════════════════════════════════════════════════════════════════════
# Uses the gpu-all image's bundled /opt/hymotion/ source (T2MRuntime pipeline).
# The launcher pre-creates symlinks in /opt/hymotion/ckpts/ so local_infer.py
# resolves clip-vit-large-patch14 + Qwen3-8B + the chosen motion checkpoint
# from the host-mounted model root.
#
# Layout expected on host (provided by comfyui/HY-Motion download):
#   /mnt/data/models/image-gen/comfyui/HY-Motion/ckpts/clip-vit-large-patch14/
#   /mnt/data/models/image-gen/comfyui/HY-Motion/ckpts/Qwen3-8B/             (optional)
#   /mnt/data/models/image-gen/comfyui/HY-Motion/ckpts/tencent/HY-Motion-1.0[-Lite]/HY-Motion-1.0[-Lite]/{config.yml,latest.ckpt}
#
# Port: 8097 → 8000 (container internal)
# Output: NPZ always works; GLB if trimesh is available; FBX requires FBX SDK
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8097}"
CONTAINER_NAME="${2:-hymotion}"
MODEL_VARIANT="${HYMOTION_MODEL_VARIANT:-HY-Motion-1.0-Lite}"   # HY-Motion-1.0 | HY-Motion-1.0-Lite
COMFYUI_ROOT="${HYMOTION_COMFYUI_ROOT:-/mnt/data/models/image-gen/comfyui/HY-Motion}"

if [ ! -d "${COMFYUI_ROOT}/ckpts/clip-vit-large-patch14" ]; then
    echo "ERROR: CLIP ViT-L/14 not found at ${COMFYUI_ROOT}/ckpts/clip-vit-large-patch14"
    echo "Set HYMOTION_COMFYUI_ROOT to the comfyui HY-Motion root."
    exit 1
fi
if [ ! -f "${COMFYUI_ROOT}/ckpts/tencent/${MODEL_VARIANT}/${MODEL_VARIANT}/config.yml" ]; then
    echo "ERROR: ${MODEL_VARIANT} config.yml not found under ${COMFYUI_ROOT}/ckpts/tencent/${MODEL_VARIANT}/${MODEL_VARIANT}/"
    echo "Available variants:"
    ls "${COMFYUI_ROOT}/ckpts/tencent/" 2>/dev/null
    exit 1
fi

echo "Starting HY-Motion (${MODEL_VARIANT}) on port ${PORT}..."

# Host layout (built by comfyui/HY-Motion + motion/hy-motion-1.0 downloads):
#   ${COMFYUI_ROOT}/ckpts/clip-vit-large-patch14/
#   ${COMFYUI_ROOT}/ckpts/tencent/${MODEL_VARIANT}/${MODEL_VARIANT}/{config.yml,latest.ckpt}
#   /mnt/data/models/motion/hy-motion-1.0/ckpts/Qwen3-8B/   (text encoder, shared across variants)
QWEN3_ROOT="${QWEN3_ROOT:-/mnt/data/models/motion/hy-motion-1.0}"

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8000 \
  -v "${COMFYUI_ROOT}:/models/hy-motion:ro" \
  -v "${QWEN3_ROOT}/ckpts/Qwen3-8B:/models/qwen3-8b:ro" \
  -v "$(dirname "$0")/api_hymotion.py:/opt/api.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e "MODEL_VARIANT=${MODEL_VARIANT}" \
  --entrypoint bash \
  forge-reg.local:30500/tech-noir/gpu-all:latest \
  -c '
    set -e
    # 1) Ensure hymotion package is importable (add __init__.py if missing)
    find /opt/hymotion/hymotion -type d -name "__pycache__" -prune -o -type d -print 2>/dev/null | while read d; do
      [ -f "$d/__init__.py" ] || touch "$d/__init__.py"
    done
    # 2) Symlink host-mounted ckpts into /opt/hymotion/ckpts/ so local_infer.py finds them
    ln -sfn /models/hy-motion/ckpts/clip-vit-large-patch14 /opt/hymotion/ckpts/clip-vit-large-patch14
    ln -sfn /models/qwen3-8b /opt/hymotion/ckpts/Qwen3-8B
    # Make sure all Tencent motion checkpoint variants are linked too
    mkdir -p /opt/hymotion/ckpts/tencent
    for variant_dir in /models/hy-motion/ckpts/tencent/*/; do
        [ -d "$variant_dir" ] || continue
        name=$(basename "$variant_dir")
        ln -sfn "$variant_dir" "/opt/hymotion/ckpts/tencent/$name"
    done
    # 3) Export env for the API server
    export PYTHONPATH="/opt/hymotion:${PYTHONPATH:-}"
    export HYMOTION_MODEL_PATH="/opt/hymotion/ckpts/tencent/${MODEL_VARIANT}/${MODEL_VARIANT}"
    cd /opt/hymotion
    exec python3 /opt/api.py
  '

echo ""
echo "HY-Motion running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test (NPZ output):"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"a person waving hello\", \"duration\": 2.0, \"format\": \"npz\"}' \\"
echo "    -o motion.npz"
