#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_kimodo.sh — Kimodo-SOMA-RP text-to-3D human motion (NVIDIA, Apache-2.0)
# ═════════════════════════════════════════════════════════════════════════════
# Uses the `kimodo` package bundled in /opt/kimodo/ inside the gpu-all image.
# Text encoder (LLM2Vec-Meta-Llama-3-8B) runs on CPU; denoiser on GPU.
#
# Variants:
#   - Kimodo-SOMA-RP-v1.1   (default)  — SOMA skeleton
#   - Kimodo-SOMA-SEED-v1.1            — SOMA skeleton, SEED bench
#   - Kimodo-G1-RP-v1                  — Unitree G1 robot
#   - Kimodo-SMPLX-RP-v1               — SMPL-X skeleton
#
# Host layout expected:
#   /mnt/data/models/avatar/kimodo/${VARIANT}/                          — Kimodo checkpoint
#   /mnt/data/models/cache/huggingface/McGill-NLP/LLM2Vec-...           — text encoder
#
# HF_TOKEN env required (gated Llama-3-8B). Set via export HF_TOKEN=hf_...
#
# Port: 8098 → 8098 (container internal)
# Output: NPZ (posed_joints, betas, etc.) with X-Inference-Time-S header
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8098}"
CONTAINER_NAME="${2:-kimodo}"
VARIANT="${KIMODO_VARIANT:-Kimodo-SOMA-RP-v1.1}"
MODEL_ROOT="${KIMODO_MODEL_ROOT:-/mnt/data/models/avatar/kimodo}"
LLM2VEC_ROOT="${KIMODO_LLM2VEC_ROOT:-/mnt/data/models/cache/huggingface/McGill-NLP}"

if [ ! -d "${MODEL_ROOT}/${VARIANT}" ]; then
    echo "ERROR: ${MODEL_ROOT}/${VARIANT} not found"
    echo "Available:"; ls "${MODEL_ROOT}" 2>/dev/null
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env required (gated Meta-Llama-3-8B-Instruct text encoder)"
    exit 1
fi

echo "Starting Kimodo (${VARIANT}) on port ${PORT}..."

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8098 \
  -v "${MODEL_ROOT}:/mnt/data/models/avatar/kimodo:ro" \
  -v "/mnt/data/models/cache/huggingface:/mnt/data/models/cache/huggingface:ro" \
  -v "$(dirname "$0")/api_kimodo.py:/opt/api_kimodo.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e KIMODO_VARIANT="${VARIANT}" \
  -e KIMODO_MODEL_ROOT="/mnt/data/models/avatar/kimodo" \
  -e KIMODO_LLM2VEC_ROOT="/mnt/data/models/cache/huggingface/McGill-NLP" \
  -e TEXT_ENCODER_DEVICE=cpu \
  -e LOCAL_CACHE=True \
  -e TEXT_ENCODERS_DIR="/mnt/data/models/cache/huggingface" \
  -e HF_HOME="/mnt/data/models/cache/huggingface" \
  -e HF_HUB_CACHE="/mnt/data/models/cache/huggingface/hub" \
  -e TRANSFORMERS_CACHE="/mnt/data/models/cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  --entrypoint python3 \
  forge-reg.local:30500/tech-noir/gpu-all:latest \
  /opt/api_kimodo.py

echo ""
echo "Kimodo running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test:"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"a person walks forward\", \"num_frames\": 60, \"num_denoising_steps\": 25}' \\"
echo "    -o motion.npz"
