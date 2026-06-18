#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_seethrough.sh — See-Through anime layer decomposition (Diffusers pool)
# ═════════════════════════════════════════════════════════════════════════════
# Serves the See-Through pipeline (LayerDiff 3D + Marigold depth) from the
# gpu-all image. Outputs layered PSD with up to 23 semantic layers.
#
# Required models (host-mounted at /mnt/data/models/image/see-through):
#   layerdiff3d/  ← layerdifforg/seethroughv0.0.2_layerdiff3d
#   marigold/     ← 24yearsold/seethroughv0.0.1_marigold
#   juggernautXL/ ← frankjoshua/juggernautXL_version6Rundiffusion (SDXL base
#                   used implicitly by KDiffusionSDXL)
#
# Mounts:
#   /opt/seethrough            ← bundled in gpu-all image (read-only is fine)
#   /mnt/.../see-through       ← host model root (RW so HF can write .cache/)
#   /opt/api_seethrough.py     ← FastAPI server
#
# RTX 4090 (24GB): bf16 inference @ 1280 resolution ~12-16 GB VRAM.
# Set SEETHROUGH_GROUP_OFFLOAD=1 to enable group offload (~10 GB at 1280).
#
# Port: 8100
# Output: layered PSD (image/vnd.adobe.photoshop)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8100}"
CONTAINER_NAME="${2:-seethrough}"
MODEL_ROOT="${SEETHROUGH_MODEL_ROOT:-/mnt/data/models/image/see-through}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${SEETHROUGH_IMAGE:-forge-reg.local:30500/tech-noir/gpu-all:latest}"

# Sanity checks
if [ ! -d "${MODEL_ROOT}/layerdiff3d" ]; then
    echo "ERROR: ${MODEL_ROOT}/layerdiff3d not found."
    echo "  hf download layerdifforg/seethroughv0.0.2_layerdiff3d --local-dir ${MODEL_ROOT}/layerdiff3d"
    exit 1
fi
if [ ! -d "${MODEL_ROOT}/marigold" ]; then
    echo "ERROR: ${MODEL_ROOT}/marigold not found."
    echo "  hf download 24yearsold/seethroughv0.0.1_marigold --local-dir ${MODEL_ROOT}/marigold"
    exit 1
fi
if [ ! -d "${MODEL_ROOT}/juggernautXL" ]; then
    echo "ERROR: ${MODEL_ROOT}/juggernautXL not found."
    echo "  hf download frankjoshua/juggernautXL_version6Rundiffusion --local-dir ${MODEL_ROOT}/juggernautXL"
    exit 1
fi

GROUP_OFFLOAD="${SEETHROUGH_GROUP_OFFLOAD:-0}"
RESOLUTION="${SEETHROUGH_RESOLUTION:-1280}"
RESOLUTION_DEPTH="${SEETHROUGH_RESOLUTION_DEPTH:-768}"

echo "Starting See-Through on port ${PORT} (res=${RESOLUTION}, depth_res=${RESOLUTION_DEPTH}, group_offload=${GROUP_OFFLOAD})..."
echo "  Image:   ${IMAGE}"
echo "  Models:  ${MODEL_ROOT}"

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8100 \
  -v "${MODEL_ROOT}:${MODEL_ROOT}" \
  -v "/mnt/data/models/cache/huggingface:/mnt/data/models/cache/huggingface" \
  -v "${SCRIPT_DIR}/api_seethrough.py:/opt/api_seethrough.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SEETHROUGH_PORT=8100 \
  -e "SEETHROUGH_MODEL_ROOT=${MODEL_ROOT}" \
  -e "SEETHROUGH_LAYERDIFF_PATH=${MODEL_ROOT}/layerdiff3d" \
  -e "SEETHROUGH_MARIGOLD_PATH=${MODEL_ROOT}/marigold" \
  -e "SEETHROUGH_RESOLUTION=${RESOLUTION}" \
  -e "SEETHROUGH_RESOLUTION_DEPTH=${RESOLUTION_DEPTH}" \
  -e "SEETHROUGH_GROUP_OFFLOAD=${GROUP_OFFLOAD}" \
  -e HF_HOME=/mnt/data/models/cache/huggingface \
  -e TRANSFORMERS_CACHE=/mnt/data/models/cache/huggingface \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'cd /opt/seethrough && exec python3 /opt/api_seethrough.py'

echo ""
echo "See-Through running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test (image → PSD):"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -F 'image=@/path/to/anime.png' \\"
echo "    -o out.psd"
echo ""
echo "Warm load first (loads both pipelines):"
echo "  curl -X POST http://localhost:${PORT}/load"
