#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_wan_t2v.sh — Wan2.1 T2V 14B (text-to-video) in the Omni-VLLM pool
# ═════════════════════════════════════════════════════════════════════════════
# Serves Wan2.1-T2V-14B-Diffusers via a custom FastAPI server (Boogu pattern)
# running inside the vllm-omni image. Uses diffusers + enable_sequential_cpu_offload
# so the 28 GB BF16 model fits on a 24 GB RTX 4090.
#
# Background: vllm-omni's openai.api_server CLI cannot currently load 14B BF16
# diffusion models on a 24 GB card (--cpu-offload-gb alone OOMs at load time
# because the full model is resident before offload hooks fire). The diffusers
# path used here streams blocks to the GPU one at a time and works reliably.
#
# Required model (host-mounted at /mnt/data/models/video/wan2.1-t2v-14b):
#   hf download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir <MODEL_PATH>
#
# Usage:
#   ./serve_wan_t2v.sh                     # port 8001, sequential offload
#   ./serve_wan_t2v.sh 8001 sequential     # explicit args
#   WAN_OFFLOAD=model_cpu ./serve_wan_t2v.sh   # faster, needs ~22 GB VRAM
#
# API:
#   POST http://localhost:8001/generate  (JSON body, returns video/mp4)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8001}"
OFFLOAD_STRATEGY="${2:-${WAN_OFFLOAD:-sequential}}"
CONTAINER_NAME="${3:-wan-t2v}"
MODEL_ROOT="${WAN_MODEL_ROOT:-/mnt/data/models/video/wan2.1-t2v-14b}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${WAN_IMAGE:-forge-reg.local:30500/tech-noir/vllm-omni:fork}"

if [ ! -d "${MODEL_ROOT}/transformer" ]; then
    echo "ERROR: ${MODEL_ROOT}/transformer not found."
    echo "  hf download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir ${MODEL_ROOT}"
    exit 1
fi

# Reuse a bundled test image if /tmp has one (so smoke tests can hit I2V too).
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting Wan2.1 T2V 14B (diffusers, offload=${OFFLOAD_STRATEGY}) on port ${PORT}..."
echo "  Image:    ${IMAGE}"
echo "  Model:    ${MODEL_ROOT}"
echo "  Offload:  ${OFFLOAD_STRATEGY}"

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  --ipc=host \
  -p ${PORT}:8001 \
  -v "${MODEL_ROOT}:${MODEL_ROOT}:ro" \
  -v "/mnt/data/models/cache/huggingface:/mnt/data/models/cache/huggingface" \
  -v "${SCRIPT_DIR}/api_wan_t2v.py:/opt/api_wan_t2v.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e WAN_PORT=8001 \
  -e WAN_MODEL_PATH="${MODEL_ROOT}" \
  -e WAN_OFFLOAD="${OFFLOAD_STRATEGY}" \
  -e HF_HOME=/mnt/data/models/cache/huggingface \
  -e TRANSFORMERS_CACHE=/mnt/data/models/cache/huggingface \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'exec python3 /opt/api_wan_t2v.py'

echo ""
echo "Wan-T2V running on port ${PORT} (container: ${CONTAINER_NAME})"
echo ""
echo "Test (text → MP4):"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\":\"a cat playing piano\",\"num_frames\":33}' \\"
echo "    -o /tmp/wan_t2v.mp4"
echo ""
echo "Warm load first:"
echo "  curl -X POST http://localhost:${PORT}/load"
