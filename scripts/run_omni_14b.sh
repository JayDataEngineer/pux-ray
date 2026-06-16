#!/usr/bin/env bash
# ── Run Omni 14B VACE on RTX 4090 (24GB) ──
# Text encoder on CPU, transformer FP8 on GPU
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/video/wan2.1-vace-14b-fp8-scaled}"
PATCH_FILE="$(dirname "$0")/pipeline_wan2_2_vace_patch.py"

echo "Starting Omni VACE 14B..."
echo "Model: $MODEL_DIR"
echo "Patch: $PATCH_FILE"

docker run --rm -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/vace-fp8:ro \
  -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:ro \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --name omni-14b-vace \
  vllm/vllm-omni:latest \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/vace-fp8 \
    --host 0.0.0.0 --port 8000 \
    --enforce-eager \
    --cpu-offload-gb 6 \
    --quantization fp8 \
    --dtype auto

echo "Container started. Watch logs: docker logs -f omni-14b-vace"
echo "API: http://localhost:8000/v1/images/generations"
