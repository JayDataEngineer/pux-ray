#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Z-Image Turbo/Base FP8 — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Z-Image-Turbo (2B DiT, FP8 weight-only) and Z-Image-Base via
# vLLM-Omni with the shared FP8 weight-only patch + CPU text encoder.
#
# Architecture:
#   * Model:   Z-Image Turbo (4-step distilled) or Base (50-step)
#   * Engine:  vLLM-Omni 0.22 (vllm/vllm-omni:latest)
#   * Patch:   scripts/pipeline_z_image_patch.py (bind-mounted over in-image file)
#   * Shared:  scripts/fp8_weight_only_patch.py (DRY patch module)
#   * Quant:   FP8 weight-only (FP8 storage, BF16 matmul — no activation quant)
#
# VRAM layout on RTX 4090 (24GB):
#   * DiT (2B FP8 weight-only)     ~2 GB  (resident)
#   * VAE (with tiling/slicing)    ~0.5 GB
#   * Text encoder (CPU offload)   ~0 GB  (moved to RAM after prefill)
#   * Activations + temp buffers   ~3 GB  (peak)
#   * Headroom                    ~18 GB
#
# Usage:
#   ./run_omni_z_image_fp8.sh                        # Z-Image Turbo, port 8094
#   ./run_omni_z_image_fp8.sh /path/to/model 8094    # override model dir + port
#   ./run_omni_z_image_fp8.sh "" 8095                # default model, port 8095
#
# API (OpenAI DALL-E compatible):
#   POST http://localhost:8094/v1/images/generations
#     {"model":"/models/z-image-fp8","prompt":"a cat","n":1,"size":"1024x1024"}
#
#   POST http://localhost:8094/v1/images/edits        # img2img
#     -F "image=@input.png" -F "prompt=edit instruction" -F "n=1"
#
#   GET  http://localhost:8094/health
# SCRIPT_EOF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${1:-/mnt/data/models/native/z-image-turbo-fp8}"
HOST_PORT="${2:-8094}"
CONTAINER_NAME="${3:-omni-z-image-fp8}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "" >&2
  echo "  Available models:" >&2
  echo "    /mnt/data/models/native/z-image-turbo-fp8/   (turbo, 4-step)" >&2
  echo "    /mnt/data/models/native/z-image-base-fp8/    (base, 50-step)" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "═══ Z-Image FP8 Server ═══════════════════════════════════════════"
echo "  Model:     $MODEL_DIR"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Image:     $IMAGE"
echo "  Strategy:  FP8 weight-only (shared patch) + CPU text encoder"
echo "  Cache:     Cache-DiT (TaylorSeer enabled)"
echo "═══════════════════════════════════════════════════════════════════"
echo

# ── Mount pipeline patch over the in-image file ────────────────────────
# The patch applies FP8-weight-only + CPU text encoder offload.
# Both need the shared fp8_weight_only_patch.py module mounted alongside.
PIPELINE_PATCH="$SCRIPT_DIR/pipeline_z_image_patch.py"
SHARED_PATCH="$SCRIPT_DIR/fp8_weight_only_patch.py"
IN_IMAGE_PIPELINE="/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/z_image/pipeline_z_image.py"
IN_IMAGE_SHARED="/usr/local/lib/python3.12/dist-packages/fp8_weight_only_patch.py"

if [[ ! -f "$PIPELINE_PATCH" ]]; then
  echo "ERROR: pipeline patch not found: $PIPELINE_PATCH" >&2
  exit 1
fi
if [[ ! -f "$SHARED_PATCH" ]]; then
  echo "ERROR: shared patch not found: $SHARED_PATCH" >&2
  echo "  Run: python3 scripts/prepare_patches.py" >&2
  exit 1
fi

docker run -d --gpus all --ipc=host \
  --shm-size=8g \
  -v "$MODEL_DIR":/models/z-image-fp8:ro \
  -v "$PIPELINE_PATCH":$IN_IMAGE_PIPELINE:ro \
  -v "$SHARED_PATCH":$IN_IMAGE_SHARED:ro \
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
echo "Waiting for server to become healthy..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo ""
    echo "  Test:"
    echo "    curl -X POST http://localhost:$HOST_PORT/v1/images/generations \\"
    echo '      -H "Content-Type: application/json" \'
    echo '      -d '"'"'{"model":"/models/z-image-fp8","prompt":"a cat wearing a hat","n":1,"size":"1024x1024"}'"'"''
    echo ""
    echo "  Logs:  docker logs -f $CONTAINER_NAME"
    exit 0
  fi
  status=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Status}}")
  if [[ "$status" == Exited* ]]; then
    echo "✗ Container exited unexpectedly:" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50 >&2
    exit 1
  fi
  sleep 2
done
echo "✗ Server failed to become healthy within 180s" >&2
docker logs "$CONTAINER_NAME" 2>&1 | tail -30 >&2
exit 1
