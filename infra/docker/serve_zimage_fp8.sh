#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_zimage_fp8.sh — LOCKED Z-Image-Turbo FP8 + Cache-DiT
# ════════════════════════════════════════════════════════════════════
# Performance: 1.61s/image on RTX 4090 (beats MI300X 1.84s datacenter)
# Throughput:  ~0.62 img/s (compute-bound on single 4090)
#
# Stack:
#   - Pre-quantized FP8 weights (convert_hf_to_fp8.py, block 128x128)
#   - FlashAttention backend (FA3/FA4)
#   - Cache-DiT with TaylorSeer O(2) calibration
#   - Performance mode: speed (no CPU offload, GPU-resident)
#
# NOTE: batching-max-size=1 because Z-Image at 1024x1024 fully saturates
# the 4090's tensor cores. Batch≥2 causes OOM on 24GB and does NOT improve
# throughput (GPU is compute-bound, not memory-bandwidth-bound).
# For batching, use multiple GPUs or a 48GB+ card.
#
# Model: /mnt/data/models/native/z-image-turbo-fp8 (secondary drive)
# Port:  8081
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

PORT="${1:-8081}"

docker run -d --name sglang-zimage-fp8 --gpus all --restart=no \
  -p ${PORT}:8080 \
  \
  -e SGLANG_CACHE_DIT_ENABLED=true \
  -e SGLANG_CACHE_DIT_FN=2 \
  -e SGLANG_CACHE_DIT_BN=1 \
  -e SGLANG_CACHE_DIT_WARMUP=2 \
  -e SGLANG_CACHE_DIT_RDT=0.5 \
  -e SGLANG_CACHE_DIT_MC=6 \
  -e SGLANG_CACHE_DIT_TAYLORSEER=true \
  -e SGLANG_CACHE_DIT_TS_ORDER=2 \
  \
  -v /mnt/data/models/native/z-image-turbo-fp8:/models/native/z-image-turbo-fp8:ro \
  \
  lmsysorg/sglang:latest \
  \
  sglang serve \
    --model-path /models/native/z-image-turbo-fp8 \
    --transformer-weights-path /models/native/z-image-turbo-fp8/transformer \
    --quantization fp8 \
    --attention-backend fa \
    --performance-mode speed \
    --host 0.0.0.0 --port 8080

echo "Z-Image-Turbo FP8 + Cache-DiT on port ${PORT}"
echo "Expected: ~1.61s/image, ~0.62 img/s"
