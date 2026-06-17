#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Cosmos3-Nano BF16 — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Cosmos3-Nano (4B world-model DiT) via vLLM-Omni 0.22.
#
# This is a TRANSITION slot — when Anima_Base lands, this script gets
# replaced by run_omni_anima_base.sh and the cosmos: route in
# inference_pools.yaml gets an anima-base: alias.
#
# VRAM strategy on RTX 4090 (24GB):
#   FP8 weight-only is currently BLOCKED on Cosmos3-Nano because its fused
#   LLM params (base_model.language_model.*) crash the FP8 prepare_layout
#   path in both vLLM-Omni and SGLang. Falling back to BF16 + CPU offload.
#
#   Memory layout:
#     * DiT (4B BF16, layerwise offload)   ~1-2 GB on GPU at a time
#     * VAE                                 ~0.5 GB
#     * Activations + KV cache              ~3 GB peak
#     * LLM text encoder (CPU)              ~0 GB on GPU
#     * Headroom                           ~18 GB
#
# Usage:
#   ./run_omni_cosmos.sh                         # defaults
#   ./run_omni_cosmos.sh /path/to/model          # override model dir
#   ./run_omni_cosmos.sh "" 8098                 # override port
#
# API endpoint:
#   POST http://localhost:8098/v1/videos/generations
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/cosmos3-nano}"
HOST_PORT="${2:-8098}"
CONTAINER_NAME="${3:-omni-cosmos}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Download the model first:" >&2
  echo "    huggingface-cli download nvidia/Cosmos3-Nano-Diffusers --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Cosmos3-Nano server..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  BF16 + layerwise CPU offload (FP8 blocked on fused LLM params)"
echo "  NOTE:      This slot becomes Anima_Base when ready — see inference_pools.yaml"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/cosmos:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e DIFFUSION_VAE_USE_TILING=1 \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/cosmos \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --enable-layerwise-offload \
    --vae-use-tiling \
    --dtype bfloat16

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:    curl http://localhost:$HOST_PORT/v1/models"
    echo "  Generate:  curl http://localhost:$HOST_PORT/v1/videos/generations"
    echo "  Logs:      docker logs -f $CONTAINER_NAME"
    exit 0
  fi
  status=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Status}}")
  if [[ "$status" == Exited* ]]; then
    echo "✗ Container exited unexpectedly:" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30 >&2
    exit 1
  fi
  sleep 2
done
echo "✗ Server failed to become healthy within 180s" >&2
docker logs "$CONTAINER_NAME" 2>&1 | tail -20 >&2
exit 1
