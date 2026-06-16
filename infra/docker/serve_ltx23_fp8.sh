#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# serve_ltx23_fp8.sh — LTX-2.3 ModelOpt FP8 Two-Stage Video Generation
# ════════════════════════════════════════════════════════════════════
# Serves LTX-2.3 (22B) with the BBuf ModelOpt FP8 transformer on RTX 4090.
#
# WHY ModelOpt FP8 (not our convert_hf_to_fp8.py):
#   The standard FP8 block-128 format from convert_hf_to_fp8.py runs
#   sgl_per_token_group_quant_8bit_v2 (CUDA-only) during weight loading.
#   On 24GB GPUs, the 22B transformer MUST be CPU-offloaded, so weights
#   are on CPU at init time → CUDA kernel on CPU = crash.
#
#   ModelOpt FP8 (quant_method=modelopt, quant_algo=FP8) uses SGLang's
#   dedicated ModelOpt loader that preserves FP8 tensor strides directly,
#   bypassing the CUDA-only initialization kernels entirely. This allows
#   layerwise DiT offload to function on 24GB GPUs.
#
# ARCHITECTURE:
#   - Text Encoder: Gemma 3 12B BF16, CPU-offloaded via FSDP
#   - Transformer:  22B FP8 (ModelOpt), layerwise GPU offload
#   - VAE:          LTX-2 Video VAE, layerwise offload
#   - Pipeline:     LTX2TwoStagePipeline (distilled LoRA + spatial upsampler)
#   - Device Mode:  snapshot (balance latency/VRAM for two-stage)
#
# Model: /mnt/data/models/native/ltx-2.3-fp8
# Port:  30010
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

PORT="${1:-30010}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/native/ltx-2.3-fp8}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-/mnt/data/models/native/ltx23-fp8-transformer}"

docker run -d --name sglang-ltx23 --gpus all --restart=no \
  -p ${PORT}:8080 \
  -v /mnt/data/models:/models:rw \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  \
  lmsysorg/sglang:latest \
  \
  sglang serve \
    --model-path /models/native/ltx-2.3-fp8 \
    --pipeline-class-name LTX2TwoStagePipeline \
    --transformer-path /models/native/ltx23-fp8-transformer \
    --text-encoder-cpu-offload true \
    --pin-cpu-memory true \
    --ltx2-two-stage-device-mode snapshot \
    --server-warmup false \
    --host 0.0.0.0 --port 8080

echo "LTX-2.3 ModelOpt FP8 Two-Stage on port ${PORT}"
echo "Pipeline: LTX2TwoStagePipeline (distilled + spatial upscaler)"
echo ""
echo "NOTE: First startup takes ~10 min (22GB transformer streaming at ~40 MiB/s)"
echo "NOTE: If OOM during loading, add --dit-layerwise-offload --dit-cpu-offload false"
echo "      and --layerwise-offload-components text_encoder dit"
