#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Qwen-Image-Edit-2511 — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Qwen-Image-Edit-2511 (20B MMDiT) via vLLM-Omni 0.22.
#
# VRAM strategy on RTX 4090 (24GB):
#   The 20B BF16 model is ~40GB — too large for resident GPU memory.
#   Layerwise CPU offload swaps DiT blocks between CPU RAM and GPU VRAM
#   during each denoising forward pass. Combined with Cache-DiT (which
#   caches intermediate block outputs across timesteps), the effective
#   per-step compute is minimized.
#
#   Memory layout with --enable-layerwise-offload:
#     * One DiT block on GPU          ~1-2 GB
#     * VAE (with tiling)             ~0.5 GB
#     * Activations + temp buffers    ~2-3 GB peak
#     * Text encoder (Qwen2.5-VL-7B)  ~9 GB  (loaded once, then cached)
#     * Headroom                      ~9-11 GB
#
# Cache-DiT configuration (from vLLM-Omni benchmarks):
#   --cache-backend cache_dit  → ~2.38x speedup (51.5s → 21.6s on single GPU)
#
# Usage:
#   ./run_omni_qwen_img_edit.sh                 # default model dir
#   ./run_omni_qwen_img_edit.sh /path/to/model  # override
#   ./run_omni_qwen_img_edit.sh "" 8092         # override port
#
# API endpoint (OpenAI DALL-E compatible):
#   POST http://localhost:8092/v1/images/edits
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/image-gen/qwen-image-edit/2511}"
HOST_PORT="${2:-8092}"
CONTAINER_NAME="${3:-omni-qwen-img-edit}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

# Reject if model dir doesn't exist
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo ""
  echo "  Download the model first:"
  echo "    python3 scripts/download_qwen_image_edit.py --model 2511"
  exit 1
fi

# Stop any existing container with this name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Qwen-Image-Edit-2511 server..."
echo "  Model:    $MODEL_DIR"
echo "  Port:     $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Cache:    Cache-DiT (TaylorSeer enabled)"
echo "  VRAM:     Layerwise CPU offload + VAE tiling/slicing"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/qwen-img-edit:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/qwen-img-edit \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --vae-use-slicing \
    --vae-use-tiling \
    --cache-backend cache_dit \
    --cache-config '{"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4, "enable_taylorseer": true}' \
    --enable-layerwise-offload

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:   curl http://localhost:$HOST_PORT/v1/models"
    echo "  Edit API: curl http://localhost:$HOST_PORT/v1/images/edits"
    echo "  Logs:     docker logs -f $CONTAINER_NAME"
    exit 0
  fi
  # Check if container died
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