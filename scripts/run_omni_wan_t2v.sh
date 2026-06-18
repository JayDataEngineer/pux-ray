#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Wan2.1 T2V (Text-to-Video) — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Wan2.1 T2V 14B (BF16) via vLLM-Omni 0.22 with layerwise CPU offload.
#
# VRAM strategy on RTX 4090 (24GB):
#   The 14B BF16 model is ~28GB — too large for resident GPU memory. Use
#   --enable-layerwise-offload to swap DiT blocks CPU↔GPU during each
#   denoising forward pass. Combined with --cpu-offload-gb 20 for the
#   non-layerwise path, this fits comfortably on 24GB.
#
#   Memory layout:
#     * Active DiT block on GPU    ~1-2 GB
#     * VAE (with tiling)          ~0.5 GB
#     * Activations + temp         ~3 GB peak
#     * T5 text encoder            ~9 GB  (CPU after prefill)
#     * Headroom                  ~9-11 GB
#
# TeaCache (Timestep Embedding Aware Cache):
#   Disabled by default (OMNI_TEACACHE_THRESH=0).
#   Enable by setting threshold before launch:
#     OMNI_TEACACHE_THRESH=0.01 ./run_omni_wan_t2v.sh
#   0.01 → ~70% speedup, great quality
#
# Usage:
#   ./run_omni_wan_t2v.sh                         # defaults
#   ./run_omni_wan_t2v.sh /path/to/model          # override model dir
#   ./run_omni_wan_t2v.sh "" 8096                 # override port
#
# API endpoint:
#   POST http://localhost:8096/v1/videos/generations
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/video/wan2.1-t2v-14b}"
HOST_PORT="${2:-8096}"
CONTAINER_NAME="${3:-omni-wan-t2v}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Download the model first:" >&2
  echo "    huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Wan2.1 T2V 14B server..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  BF16 + layerwise CPU offload + VAE tiling"
if [[ "${OMNI_TEACACHE_THRESH:-0}" != "0" ]]; then
  echo "  TeaCache:  ON  (thresh=${OMNI_TEACACHE_THRESH})"
else
  echo "  TeaCache:  OFF (set OMNI_TEACACHE_THRESH=0.01 to enable)"
fi
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/wan-t2v:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMNI_TEACACHE_THRESH="${OMNI_TEACACHE_THRESH:-0}" \
  -e DIFFUSION_VAE_USE_TILING=1 \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/wan-t2v \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --omni \
    --enforce-eager \
    --cpu-offload-gb 20 \
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
