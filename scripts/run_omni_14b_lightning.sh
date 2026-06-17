#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# Wan2.1 VACE 14B FP8 LIGHTNING — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════
# 4-step lightning distilled variant of VACE 14B for fast inference.
# Same FP8 weight-only patch as the base model — the patch is applied at
# the linear-layer level, so it works for any Wan2.1-VACE-14B checkpoint
# regardless of whether it has been distilled.
#
# A lightning checkpoint should be a direct-cast FP8 model directory
# containing the same structure as the base model (transformer/, vae/,
# text_encoder/, tokenizer/, scheduler/). The transformer weights must
# be the 4-step distilled variant (e.g., LightX2V LoRA merged in BF16,
# then re-cast to FP8).
#
# Usage:
#   ./run_omni_14b_lightning.sh                       # defaults
#   ./run_omni_14b_lightning.sh /path/to/model 8001   # override
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/video/wan2.1-vace-14b-fp8-lightning}"
HOST_PORT="${2:-8001}"
CONTAINER_NAME="${3:-omni-14b-vace-lightning}"
PATCH_FILE="/home/user/Documents/programs/ray/scripts/pipeline_wan2_2_vace_patch.py"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: lightning model dir not found: $MODEL_DIR" >&2
  echo "Hint: create one by merging LightX2V 4-step LoRA into the base" >&2
  echo "      BF16 model, then cast to FP8 (see convert_to_fp8.py)." >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: patch file not found: $PATCH_FILE" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Omni Lightning 14B FP8 server..."
echo "  Model:    $MODEL_DIR"
echo "  Patch:    $PATCH_FILE"
echo "  Port:     $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo

docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/vace-fp8-lightning:ro \
  -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:ro \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m vllm_omni.entrypoints.openai.api_server \
    --model /models/vace-fp8-lightning \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --cpu-offload-gb 20 \
    --quantization fp8 \
    --dtype auto

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ Lightning ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
    echo "  Models:   curl http://localhost:$HOST_PORT/v1/models"
    echo "  Logs:     docker logs -f $CONTAINER_NAME"
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
