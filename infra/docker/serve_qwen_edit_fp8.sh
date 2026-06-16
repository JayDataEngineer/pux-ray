#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_qwen_edit_fp8.sh — Qwen-Image-Edit-2511 ModelOpt FP8
# ════════════════════════════════════════════════════════════════════
# Uses lmsys/qwen-image-edit-modelopt-fp8-sglang-transformer (official
# ModelOpt FP8, compatible with SGLang's layerwise offload).
#
# STATUS: OOM on RTX 4090 (24GB). The Qwen2.5-VL text encoder (16GB)
# stages on GPU during initialization. PyTorch caching allocator holds
# ~18GB residual, leaving insufficient VRAM for the 29GB FP8 transformer's
# ModelOpt dequantization step. This is the same structural limitation
# as LTX-2.3 on 24GB.
#
# EXPECTED TO WORK ON: 48GB+ GPUs (RTX 6000, A6000, etc.)
#
# Model: /mnt/data/models/native/qwen-image-edit-2511-fp8
# ════════════════════════════════════════════════════════════════════
set -euo pipefail
PORT="${1:-30011}"

docker run -d --name sglang-qwen-edit --gpus all --restart=no \
  -p ${PORT}:8080 \
  -v /mnt/data/models:/models:rw \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:latest \
  sglang serve \
    --model-path /models/native/qwen-image-edit-2511-fp8 \
    --transformer-path /models/native/qwen-edit-modelopt-fp8-transformer \
    --text-encoder-cpu-offload true \
    --pin-cpu-memory true \
    --server-warmup false \
    --host 0.0.0.0 --port 8080

echo "Qwen-Image-Edit ModelOpt FP8 on port ${PORT}"
echo "NOTE: May OOM on 24GB GPUs during loading"
