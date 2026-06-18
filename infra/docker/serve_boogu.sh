#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_boogu.sh — Boogu-Image-0.1-Edit serving container
# ════════════════════════════════════════════════════════════════════
# DEPRECATED: use serve_boogu_omni.sh (Omni-VLLM route). This is the
# standalone fallback, kept only for when vllm-omni is unavailable.
# Boogu-Image-0.1-Edit: unified image generation & editing model.
# Qwen3-VL MLLM + custom DiT + FLUX.1 VAE.
# ~35.8 GB weights, ~22 GB VRAM with model CPU offload.
#
# Port: 8096 (Tier A — boogu pool)
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8096}"
CONTAINER_NAME="${2:-boogu-edit}"
MODEL_PATH="${BOOGU_MODEL_PATH:-/mnt/data/models/image-gen/Boogu-Image-0.1-Edit}"
OFFLOAD="${BOOGU_OFFLOAD:-model_cpu}"

# Check model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "Set BOOGU_MODEL_PATH or download the model first:"
    echo "  hf download Boogu/Boogu-Image-0.1-Edit --local-dir ${MODEL_PATH}"
    exit 1
fi

# Check for safetensors (ensure download completed)
if ! ls "${MODEL_PATH}"/mllm/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: No safetensors found in ${MODEL_PATH}/mllm/"
    echo "Model download may be incomplete or still in progress."
    exit 1
fi

echo "Starting Boogu-Image-0.1-Edit on port ${PORT} (offload=${OFFLOAD})..."

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8096 \
  -v "${MODEL_PATH}:/mnt/data/models/image-gen/Boogu-Image-0.1-Edit:ro" \
  -e BOOGU_MODEL_PATH=/mnt/data/models/image-gen/Boogu-Image-0.1-Edit \
  -e BOOGU_DEVICE=cuda:0 \
  -e BOOGU_OFFLOAD="${OFFLOAD}" \
  -e BOOGU_PORT=8096 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  forge-reg.local:30500/tech-noir/boogu:latest

echo ""
echo "Boogu-Image-0.1-Edit running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "T2I test:"
echo "  curl -X POST http://localhost:${PORT}/v1/images/generations \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"A serene mountain lake at sunset, cinematic quality\"}'"
echo ""
echo "TI2I (edit) test:"
echo "  IMG_B64=\$(base64 -w0 input.png)"
echo '  curl -X POST http://localhost:'${PORT}'/v1/images/generations \'
echo "    -H 'Content-Type: application/json' \\"
echo '    -d "{\"prompt\": \"Replace the sky with a starry night\", \"input_image_b64\": \"\$IMG_B64\"}"'
