#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Qwen-Image-Edit-2511 FP8 WEIGHT-ONLY — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Serves Qwen-Image-Edit-2511 (20B MMDiT) via vLLM-Omni 0.22 with FP8
# weight-only quantization + CPU text encoder offload.
#
# VRAM strategy on RTX 4090 (24GB):
#   Instead of layerwise offload (which swaps blocks CPU↔GPU every step
#   and is slow), we store ALL 60 DiT blocks on GPU in FP8 (Float8_e4m3fn).
#   At 20GB for the DiT, that leaves ~4GB for VAE + activations.
#
#   Memory layout (FP8 weight-only + CPU text encoder):
#     * DiT (20B FP8, 60 blocks)    ~20 GB  (stays resident, no swapping)
#     * VAE (with tiling)           ~0.3 GB
#     * Activations + temp buffers   ~3 GB
#     * Text encoder (CPU)          ~0 GB  (CPU RAM, moved after prefill)
#     * Headroom                     ~0.7 GB
#     ──────────────────────────────────────
#     Total on GPU:                 ~23 GB  ✓ fits on 24GB
#
#   vs the BF16+layerwise-offload baseline:
#     * DiT (20B BF16)              ~40 GB → OOM without offload
#     * Layerwise offload           ~1 block at a time → slow (CPU↔GPU swap)
#     * FP8 weight-only             ALL 60 blocks on GPU → fast (no swap)
#
# Why FP8 weight-only instead of vLLM's built-in FP8?
#   vLLM's built-in FP8 quantization includes ACTIVATION quantization,
#   which produces NaN in Diffusion Transformer linear layers (the latent
#   range shifts across denoising timesteps, compounding rounding errors).
#
#   The FP8 weight-only patch (pipeline_qwen_image_edit_plus_patch.py)
#   keeps FP8 weight storage but dequantizes to BF16 for matmul, adding
#   zero memory overhead while eliminating the NaN cascade.
#
#   VLLM_BATCH_INVARIANT=1 is set so that vLLM's Fp8LinearMethod.apply
#   itself takes the BF16-dequant + F.linear path (instead of CUTLASS FP8
#   scaled GEMM) for layers that don't use the custom
#   _Fp8WeightOnlyLinearMethod (modulation / img_in / txt_in / norm_out /
#   proj_out / timestep_embedder).
#
# Pipeline patch:
#   The patch file is bind-mounted over the vLLM-Omni pipeline file:
#     scripts/pipeline_qwen_image_edit_plus_patch.py
#     → vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py
#
#   It applies two changes at import time:
#     1. Monkey-patches Fp8Config.get_quant_method → FP8 weight-only for
#        DiT attention + MLP linear layers
#     2. Moves the text encoder to CPU and patches its forward method to
#        handle GPU→CPU→GPU device transfers transparently
#
# Cache-DiT acceleration (compounds with FP8):
#   --cache-backend cache_dit  → ~2.38x speedup (block-level caching)
#   With FP8 weight-only + all blocks on GPU, the effective speedup is
#   even better since there's no CPU offload overhead.
#
# Usage:
#   ./run_omni_qwen_img_edit_fp8.sh                   # default model dir
#   ./run_omni_qwen_img_edit_fp8.sh /path/to/model    # override
#   ./run_omni_qwen_img_edit_fp8.sh "" 8093           # override port
#
# API endpoint (OpenAI DALL-E compatible):
#   POST http://localhost:8093/v1/images/edits
#
# Preparation:
#   python3 scripts/prepare_qwen_img_edit_fp8.py
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL_DIR="${1:-/mnt/data/models/image-gen/qwen-image-edit/2511-fp8}"
HOST_PORT="${2:-8093}"
CONTAINER_NAME="${3:-omni-qwen-img-edit-fp8}"
IMAGE="vllm/vllm-omni:latest"
CONTAINER_PORT=8000
PATCH_FILE="/home/user/Documents/programs/ray/scripts/pipeline_qwen_image_edit_plus_patch.py"
LAUNCHER="/home/user/Documents/programs/ray/scripts/launch_qwen_img_edit_fp8.py"

# Reject if model dir doesn't exist
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo ""
  echo "  Prepare the FP8 model first:"
  echo "    python3 scripts/prepare_qwen_img_edit_fp8.py"
  exit 1
fi

# Reject if patch file doesn't exist
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: patch file not found: $PATCH_FILE" >&2
  exit 1
fi

# Stop any existing container with this name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting Qwen-Image-Edit-2511 FP8 weight-only server..."
echo "  Model:     $MODEL_DIR"
echo "  Patch:     $PATCH_FILE"
echo "  Port:      $HOST_PORT -> $CONTAINER_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Strategy:  FP8 weight-only (20GB DiT on GPU) + CPU text encoder"
echo "  Cache:     Cache-DiT (TaylorSeer enabled)"
echo "  No layerwise offload — all 60 blocks on GPU"
echo

# Launch container with FP8 quantization and pipeline patch
docker run -d --gpus all --ipc=host \
  -v "$MODEL_DIR":/models/qwen-img-edit-fp8:ro \
  -v "$PATCH_FILE":/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py:ro \
  -v "$(dirname "$PATCH_FILE")/fp8_weight_only_patch.py":/usr/local/lib/python3.12/dist-packages/fp8_weight_only_patch.py:ro \
  -v "$LAUNCHER":/launcher.py:ro \
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
  python3 /launcher.py \
    --model /models/qwen-img-edit-fp8 \
    --host 0.0.0.0 --port "$CONTAINER_PORT" \
    --enforce-eager \
    --dtype auto

echo
echo "Container started. Waiting for server..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "✓ FP8 ready on http://localhost:$HOST_PORT  ($(( i * 2 ))s)"
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