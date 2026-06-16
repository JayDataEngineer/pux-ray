#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_cosmos3_nano.sh — Cosmos3-Nano (nvidia/Cosmos3-Nano)
# ════════════════════════════════════════════════════════════════════
# NVIDIA Cosmos3-Nano — unified T2V, I2V, T2I model.
# 14B Cosmos3OmniTransformer with Qwen3VL vision encoder.
#
# STATUS: BF16 WORKS with auto CPU offload (transformer on CPU).
#         FP8 conversion done but SGLang's FP8 handler doesn't support
#         Cosmos3's fused LLM parameters (to_qkv, gate_up_proj).
#         Need newer SGLang or ModelOpt FP8 for GPU-resident FP8.
#
# Components (BF16):
#   - Transformer: 27 GB (Cosmos3OmniTransformer) → CPU offloaded
#   - VAE:         1.3 GB → GPU resident
#   - Vision enc:  1.1 GB → GPU resident
#   - Sound tok:   1.8 GB
#
# Model: /mnt/data/models/cosmos3-nano
# Port:  30012
# ════════════════════════════════════════════════════════════════════
set -euo pipefail
PORT="${1:-30012}"
MODEL="${MODEL:-/mnt/data/models/cosmos3-nano}"

docker run -d --name sglang-cosmos3 --gpus all --restart=no \
  -p ${PORT}:8080 \
  -v /mnt/data/models:/models:ro \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:latest \
  sglang serve \
    --model-path /models/cosmos3-nano \
    --server-warmup false \
    --host 0.0.0.0 --port 8080

echo "Cosmos3-Nano on port ${PORT}"
echo "NOTE: BF16 with auto CPU offload. FP8 needs SGLang update for fused LLM params."
