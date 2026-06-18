#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# LTX-Video 2.3 FP8 — vLLM-Omni server fallback (dual-engine)
# ════════════════════════════════════════════════════════════════════════════
# Serves LTX-2.3 22B ModelOpt FP8 via vLLM-Omni 0.22+.
# PRIMARY engine is SGLang (layerwise offload is faster).
# This script is the omni-vllm FALLBACK path.
#
# vLLM-Omni LTX-2.3 support (v0.22+):
#   - ModelOpt FP8 format
#   - VAE decode parallelism
#   - Auxiliary modules kept resident
#
# VRAM on RTX 4090 (24GB) with ModelOpt FP8:
#   * Transformer (22B FP8)         ~14 GB
#   * Text encoder (Gemma 3 12B)    ~7 GB (CPU offloaded)
#   * VAE + activations             ~3 GB
#   * Total on GPU:                 ~17 GB  ✓
#
# Usage:
#   ./run_omni_ltx23_fp8.sh                              # defaults
#   ./run_omni_ltx23_fp8.sh /path/to/model                # override model dir
#   ./run_omni_ltx23_fp8.sh "" 8099                       # override port
#
# API endpoint:
#   POST http://localhost:8099/v1/videos/generations
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/native/ltx-2.3-fp8}"
HOST_PORT="${2:-8099}"
CONTAINER_NAME="${3:-omni-ltx23}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Download the model first:" >&2
  echo "    huggingface-cli download Lightricks/LTX-2.3-fp8 --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting LTX-2.3 FP8 (omni-vllm fallback)..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Quant:     ModelOpt FP8"
echo "  Note:      Fallback path — SGLang is primary for this model"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/ltx23:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e DIFFUSION_VAE_USE_TILING=1 \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/ltx23 \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
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
