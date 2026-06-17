#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# Wan2.1 VACE 14B FP8 — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════
# Serves direct-cast FP8 14B VACE on a single 24GB GPU via vLLM-Omni 0.22.
# The pipeline_wan2_2_vace_patch.py file is bind-mounted over the in-image
# pipeline and applies the FP8 weight-only patch (FP8 storage + BF16 matmul,
# no activation quantization) — required because vLLM's standard FP8 path
# NaN-cascades on DiT linear layers across denoising timesteps.
#
# Memory layout on RTX 4090 (24GB):
#   * FP8 transformer weights      ~14 GB (on GPU)
#   * VAE + tokenizer                ~1 GB (on GPU)
#   * Activations + temp BF16 dequant ~3 GB peak (per forward pass)
#   * Headroom / fragmentation       ~6 GB
#   `--cpu-offload-gb 20` lets vLLM spill up to 20 GB of weights to CPU
#   if needed; in practice GPU stays ~21 GB during inference.
#
# Usage:
#   ./run_omni_14b.sh                 # default model dir
#   ./run_omni_14b.sh /path/to/model  # override model dir
#   ./run_omni_14b.sh "" 8002         # override port
#
# TeaCache (Timestep Embedding Aware Cache):
#   Disabled by default (OMNI_TEACACHE_THRESH=0).
#   Enable by setting threshold before launch:
#     OMNI_TEACACHE_THRESH=0.01 ./run_omni_14b.sh
#   Recommended thresholds (raw L1 distance, no polynomial rescaling):
#     0.005 → ~48% speedup, identical quality
#     0.01  → ~70% speedup, great quality (sweet spot)
#     0.015 → ~73% speedup, slight quality trade-off
#   TeaCache caches DiT block outputs when consecutive timestep_proj
#   vectors are similar — skip recomputation for near-identical steps.
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/video/wan2.1-vace-14b-fp8-diffusers}"
HOST_PORT="${2:-8000}"
CONTAINER_NAME="${3:-omni-14b-vace-fp8}"
PATCH_FILE="/home/user/Documents/programs/ray/scripts/pipeline_wan2_2_vace_patch.py"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

# Reject if model dir doesn't exist
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: patch file not found: $PATCH_FILE" >&2
  exit 1
fi

# Stop any existing container with this name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Omni VACE 14B FP8 server..."
echo "  Model:    $MODEL_DIR"
echo "  Patch:    $PATCH_FILE"
echo "  Port:     $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
if [[ "${OMNI_TEACACHE_THRESH:-0}" != "0" ]]; then
  echo "  TeaCache: ON  (thresh=${OMNI_TEACACHE_THRESH})"
else
  echo "  TeaCache: OFF (set OMNI_TEACACHE_THRESH=0.01 to enable)"
fi
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/vace-fp8:ro \
  -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMNI_TEACACHE_THRESH="${OMNI_TEACACHE_THRESH:-0}" \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/vace-fp8 \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --cpu-offload-gb 20 \
    --quantization fp8 \
    --dtype auto

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:   curl http://localhost:$HOST_PORT/v1/models"
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
