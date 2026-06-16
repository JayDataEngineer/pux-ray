#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_ideogram4.sh — Ideogram 4 Text-to-Image
# ════════════════════════════════════════════════════════════════════
# Ideogram 4: typography-aware text-to-image with strong composition.
# Presets: V4_DEFAULT_20, V4_QUALITY_48, V4_TURBO_12
#
# PREREQUISITE: Accept license at:
#   https://huggingface.co/ideogram-ai/ideogram-4-nf4
#   https://huggingface.co/ideogram-ai/ideogram-4-fp8
# Then set HF_TOKEN env var.
#
# Variants for RTX 4090:
#   NF4 (16 GB): text_encoder 5.5GB + transformer 5.2GB — fits easily
#   FP8 (28 GB): text_encoder 8.8GB + transformer 9.3GB — needs offload
#
# Port: 30013
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-30013}"
VARIANT="${VARIANT:-nf4}"  # nf4 or fp4
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var with access to ideogram repos}"

case "$VARIANT" in
  nf4)
    MODEL="ideogram-ai/ideogram-4-nf4"
    EXTRA="--quantization bitsandbytes"
    ;;
  fp8)
    MODEL="ideogram-ai/ideogram-4-fp8"
    EXTRA=""
    ;;
  *)
    echo "Unknown variant: $VARIANT (use nf4 or fp8)"
    exit 1
    ;;
esac

docker run -d --name sglang-ideogram --gpus all --restart=no \
  -p ${PORT}:8080 \
  -v /mnt/data/models:/models:rw \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/models/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:latest \
  sglang serve \
    --model-path ${MODEL} \
    --performance-mode auto \
    --server-warmup false \
    --host 0.0.0.0 --port 8080

echo "Ideogram 4 (${VARIANT}) on port ${PORT}"
echo "Presets: V4_DEFAULT_20, V4_QUALITY_48, V4_TURBO_12"
echo ""
echo "Test:"
echo "  curl -X POST http://localhost:${PORT}/v1/images/generations \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL}\", \"prompt\": \"A serene mountain landscape at dawn\", \"size\": \"1024x1024\", \"extra_body\": {\"preset\": \"V4_DEFAULT_20\"}}'"
