#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_wan_i2v.sh — Wan2.1 I2V 14B (image+text-to-video) in the Omni-VLLM pool
# ═════════════════════════════════════════════════════════════════════════════
# Serves Wan2.1-I2V-14B-480P-Diffusers via a custom FastAPI server (Boogu pattern)
# running inside the vllm-omni image. Uses diffusers + enable_sequential_cpu_offload
# so the 30 GB BF16 model (DiT + CLIP image encoder + VAE) fits on a 24 GB RTX 4090.
#
# Required model (host-mounted at /mnt/data/models/video/wan2.1-i2v-14b):
#   hf download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir <MODEL_PATH>
#
# Usage:
#   ./serve_wan_i2v.sh                     # port 8002, sequential offload
#   ./serve_wan_i2v.sh 8002 sequential     # explicit args
#   WAN_OFFLOAD=model_cpu ./serve_wan_i2v.sh   # faster, needs ~22 GB VRAM
#
# API:
#   POST http://localhost:8002/generate  (multipart: image + prompt → video/mp4)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8002}"
OFFLOAD_STRATEGY="${2:-${WAN_OFFLOAD:-sequential}}"
CONTAINER_NAME="${3:-wan-i2v}"
MODEL_ROOT="${WAN_MODEL_ROOT:-/mnt/data/models/video/wan2.1-i2v-14b}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${WAN_IMAGE:-forge-reg.local:30500/tech-noir/vllm-omni:fork}"

if [ ! -d "${MODEL_ROOT}/transformer" ]; then
    echo "ERROR: ${MODEL_ROOT}/transformer not found."
    echo "  hf download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir ${MODEL_ROOT}"
    exit 1
fi
if [ ! -d "${MODEL_ROOT}/image_encoder" ]; then
    echo "ERROR: ${MODEL_ROOT}/image_encoder not found (required for I2V)."
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting Wan2.1 I2V 14B (diffusers, offload=${OFFLOAD_STRATEGY}) on port ${PORT}..."
echo "  Image:    ${IMAGE}"
echo "  Model:    ${MODEL_ROOT}"
echo "  Offload:  ${OFFLOAD_STRATEGY}"

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  --ipc=host \
  -p ${PORT}:8002 \
  -v "${MODEL_ROOT}:${MODEL_ROOT}:ro" \
  -v "/mnt/data/models/cache/huggingface:/mnt/data/models/cache/huggingface" \
  -v "${SCRIPT_DIR}/api_wan_i2v.py:/opt/api_wan_i2v.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e WAN_PORT=8002 \
  -e WAN_MODEL_PATH="${MODEL_ROOT}" \
  -e WAN_OFFLOAD="${OFFLOAD_STRATEGY}" \
  -e HF_HOME=/mnt/data/models/cache/huggingface \
  -e TRANSFORMERS_CACHE=/mnt/data/models/cache/huggingface \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'exec python3 /opt/api_wan_i2v.py'

echo ""
echo "Wan-I2V running on port ${PORT} (container: ${CONTAINER_NAME})"
echo ""
echo "Test (image + text → MP4):"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -F 'image=@/tmp/test.png' \\"
echo "    -F 'prompt=the character turns and waves' \\"
echo "    -F 'num_frames=33' \\"
echo "    -o /tmp/wan_i2v.mp4"
echo ""
echo "Warm load first:"
echo "  curl -X POST http://localhost:${PORT}/load"
