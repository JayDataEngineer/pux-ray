#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Wan2.1 I2V (Image-to-Video) — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Wan2.1 I2V 14B (BF16) via vLLM-Omni 0.22 with layerwise CPU offload.
#
# The I2V variant adds an image encoder (CLIP-ViT) on top of the T2V backbone.
# The image is encoded once at the start of denoising and injected as a
# conditioning signal in every transformer block. Same VRAM envelope as T2V
# (the image encoder is ~1.5GB and freed after encoding).
#
# VRAM strategy on RTX 4090 (24GB):
#   Identical to T2V: --enable-layerwise-offload swaps DiT blocks CPU↔GPU.
#   T5 text encoder + CLIP image encoder both moved to CPU after prefill.
#
# Usage:
#   ./run_omni_wan_i2v.sh                         # defaults
#   ./run_omni_wan_i2v.sh /path/to/model          # override model dir
#   ./run_omni_wan_i2v.sh "" 8097                 # override port
#
# API endpoint:
#   POST http://localhost:8097/v1/videos/generations (multipart: image + prompt)
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/video/wan2.1-i2v-14b}"
HOST_PORT="${2:-8097}"
CONTAINER_NAME="${3:-omni-wan-i2v}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Download the model first:" >&2
  echo "    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Wan2.1 I2V 14B server..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  BF16 + layerwise CPU offload + CLIP image encoder (CPU)"
if [[ "${OMNI_TEACACHE_THRESH:-0}" != "0" ]]; then
  echo "  TeaCache:  ON  (thresh=${OMNI_TEACACHE_THRESH})"
else
  echo "  TeaCache:  OFF (set OMNI_TEACACHE_THRESH=0.01 to enable)"
fi
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/wan-i2v:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMNI_TEACACHE_THRESH="${OMNI_TEACACHE_THRESH:-0}" \
  -e DIFFUSION_VAE_USE_TILING=1 \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/wan-i2v \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --enable-layerwise-offload \
    --vae-use-tiling \
    --dtype auto

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
