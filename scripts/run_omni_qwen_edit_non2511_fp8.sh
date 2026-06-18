#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Qwen-Edit (non-2511) FP8 WEIGHT-ONLY — vLLM-Omni server on RTX 4090 (24GB)
# ════════════════════════════════════════════════════════════════════════════
# Same architecture as 2511 (60-layer QwenImageTransformer2DModel) but trained
# from the non-2511 Qwen-Edit weights. ModelOpt FP8 source converted to
# compressed-tensors FP8 weight-only via scripts/prepare_qwen_edit_non2511_fp8.py.
#
# Reuses the SAME pipeline patch + launcher as 2511 (identical architecture).
#
# Preparation:
#   python3 scripts/prepare_qwen_edit_non2511_fp8.py
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-/mnt/data/models/image-gen/qwen-edit-non2511-fp8}"
HOST_PORT="${2:-8094}"
CONTAINER_NAME="${3:-omni-qwen-edit-non2511-fp8}"

# Delegate to the 2511 launcher (same patch, same launcher.py, same image).
# Only the model weights differ.
exec bash "$SCRIPT_DIR/run_omni_qwen_img_edit_fp8.sh" "$MODEL_DIR" "$HOST_PORT" "$CONTAINER_NAME"
