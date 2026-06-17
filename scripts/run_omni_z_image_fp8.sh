#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Z-Image Turbo FP8 — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Z-Image-Turbo (2B DiT, FP8 weight-only) via vLLM-Omni 0.22.
#
# Z-Image-Turbo is a 4-step distilled T2I DiT — fastest image path in the
# pool system. With FP8 weights + Cache-DiT, latency is dominated by the
# text encoder prefill; the 4 denoising steps themselves are negligible.
#
# VRAM layout on RTX 4090 (24GB):
#   * DiT (2B FP8)                 ~2 GB  (resident)
#   * VAE (with tiling/slicing)    ~0.5 GB
#   * Text encoder (T5-XXL)        ~9 GB  (CPU offload after prefill)
#   * Activations + KV             ~3 GB peak
#   * Headroom                    ~9 GB
#
# Cache-DiT + TaylorSeer compound with FP8: block-level caching skips
# recomputation for near-identical timesteps (esp. effective on 4-step
# turbo models where 3 of 4 steps share similarity structure).
#
# Usage:
#   ./run_omni_z_image_fp8.sh                       # defaults
#   ./run_omni_z_image_fp8.sh /path/to/model        # override model dir
#   ./run_omni_z_image_fp8.sh "" 8094               # override port
#
# API endpoint (OpenAI DALL-E compatible):
#   POST http://localhost:8094/v1/images/generations
# SCRIPT_EOF
cat >> scripts/run_omni_z_image_fp8.sh <<'SCRIPT_EOF'
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/native/z-image-turbo-fp8}"
HOST_PORT="${2:-8094}"
CONTAINER_NAME="${3:-omni-z-image-fp8}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Prepare the FP8 model first (cast weights to Float8_e4m3fn):" >&2
  echo "    python3 scripts/prepare_z_image_fp8.py" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Z-Image Turbo FP8 server..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  FP8 weight-only (2GB DiT resident) + T5 CPU offload"
echo "  Cache:     Cache-DiT (TaylorSeer enabled)"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/z-image-fp8:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_BATCH_INVARIANT=1 \
  -e DIFFUSION_VAE_USE_SLICING=1 \
  -e DIFFUSION_VAE_USE_TILING=1 \
  -e DIFFUSION_CACHE_BACKEND=cache_dit \
  -e DIFFUSION_CACHE_CONFIG='{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4, "enable_taylorseer": true}' \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/z-image-fp8 \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --quantization fp8 \
    --dtype auto

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:    curl http://localhost:$HOST_PORT/v1/models"
    echo "  Generate:  curl http://localhost:$HOST_PORT/v1/images/generations"
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
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/native/z-image-turbo-fp8}"
HOST_PORT="${2:-8094}"
CONTAINER_NAME="${3:-omni-z-image-fp8}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Prepare the FP8 model first (cast weights to Float8_e4m3fn):" >&2
  echo "    python3 scripts/prepare_z_image_fp8.py" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Z-Image Turbo FP8 server..."
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  FP8 weight-only (2GB DiT resident) + T5 CPU offload"
echo "  Cache:     Cache-DiT (TaylorSeer enabled)"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/z-image-fp8:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_BATCH_INVARIANT=1 \
  -e DIFFUSION_VAE_USE_SLICING=1 \
  -e DIFFUSION_VAE_USE_TILING=1 \
  -e DIFFUSION_CACHE_BACKEND=cache_dit \
  -e DIFFUSION_CACHE_CONFIG='{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4, "enable_taylorseer": true}' \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/z-image-fp8 \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --quantization fp8 \
    --dtype auto

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:    curl http://localhost:$HOST_PORT/v1/models"
    echo "  Generate:  curl http://localhost:$HOST_PORT/v1/images/generations"
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
