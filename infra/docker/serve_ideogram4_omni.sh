#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_ideogram4_omni.sh — Ideogram 4 via vllm-omni container
# ════════════════════════════════════════════════════════════════════
# Runs the custom api_ideogram4.py FastAPI server inside the vllm-omni
# container. The container has torch, diffusers (>=0.39.0.dev0 from
# git+https://github.com/huggingface/diffusers.git@main), and
# bitsandbytes — everything needed for the NF4 weight injection.
#
# Ideogram 4 NF4: 9.3B single-stream DiT + Qwen3-VL-8B text encoder
# Peak VRAM ~16.8 GB at 1024×1024, fits RTX 4090.
#
# Port: 8093 (maps to omni-vllm pool)
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8093}"
CONTAINER_NAME="${2:-ideogram4-omni}"
MODEL_PATH="${IDEOGRAM4_MODEL_PATH:-/mnt/data/models/image-gen/ideogram4-nf4}"

# Check model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "Set IDEOGRAM4_MODEL_PATH or update the default."
    exit 1
fi

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8093 \
  -v "${MODEL_PATH}:/models/ideogram4-nf4:ro" \
  -v "$(dirname "$0")/api_ideogram4.py:/opt/api_ideogram4.py:ro" \
  -e IDEOGRAM4_MODEL_PATH=/models/ideogram4-nf4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 \
  forge-reg.local:30500/tech-noir/vllm-omni:fork \
  /opt/api_ideogram4.py

echo "Ideogram 4 (Omni-VLLM) on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test:"
echo "  curl -X POST http://localhost:${PORT}/v1/images/generations \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"A serene mountain lake at sunset, cinematic quality\"}'"
