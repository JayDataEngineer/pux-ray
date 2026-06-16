#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_zimage_base_fp8.sh — LOCKED Z-Image-Base FP8 + Cache-DiT
# ════════════════════════════════════════════════════════════════════
# Performance: 7.78s/image (50 steps) on RTX 4090
#
# Z-Image-Base is the full (non-distilled) model. Uses 50 steps vs
# Turbo's 8. Higher quality, slower. Same aggressive Cache-DiT settings
# as Turbo — 50-step trajectory has MORE redundancy between steps,
# so aggressive caching is actually more effective here.
#
# Model: /mnt/data/models/native/z-image-base-fp8 (secondary drive)
# Port:  8081
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

PORT="${1:-8081}"

docker run -d --name sglang-zimage-base-fp8 --gpus all --restart=no \
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
  -v /mnt/data/models/native/z-image-base-fp8:/models/native/z-image-base-fp8:ro \
  \
  lmsysorg/sglang:latest \
  \
  sglang serve \
    --model-path /models/native/z-image-base-fp8 \
    --transformer-weights-path /models/native/z-image-base-fp8/transformer \
    --quantization fp8 \
    --attention-backend fa \
    --performance-mode speed \
    --host 0.0.0.0 --port 8080

echo "Z-Image-Base FP8 + Cache-DiT on port ${PORT}"
echo "Expected: ~7.78s/image (50 steps)"
