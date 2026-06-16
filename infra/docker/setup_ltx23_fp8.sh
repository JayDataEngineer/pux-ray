#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# setup_ltx23_fp8.sh — LTX-2.3 ModelOpt FP8 Setup (Repeatable)
# ════════════════════════════════════════════════════════════════════
# Downloads and configures the LTX-2.3 two-stage model with:
#   - BBuf/ltx23-two-stage-modelopt-fp8-sglang-transformer (22B FP8)
#   - Lightricks/LTX-2.3 base components (spatial upsampler, LoRA)
#   - Lightricks/LTX-2 shared components (text_encoder, tokenizer, vae, etc.)
#   - Local Gemma 3 12B BF16 text encoder (avoid 50GB HF download)
#
# The BBuf transformer uses NVIDIA ModelOpt FP8 quantization — the ONLY
# FP8 format compatible with SGLang's layerwise DiT offload on 24GB GPUs.
# Our convert_hf_to_fp8.py block-128 format CANNOT work here because the
# CUDA-only process_weights_after_loading kernel fails on CPU-staged weights.
#
# Output: /mnt/data/models/native/ltx-2.3-fp8/
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

MODELS_DIR="${1:-/mnt/data/models}"
LTX23_DIR="${MODELS_DIR}/native/ltx-2.3-fp8"
BBUF_DIR="${MODELS_DIR}/native/ltx23-fp8-transformer"

echo "=== LTX-2.3 ModelOpt FP8 Setup ==="
echo "Output: ${LTX23_DIR}"

# ── Step 1: Download BBuf ModelOpt FP8 Transformer ──────────────────
echo ""
echo "── Step 1: BBuf ModelOpt FP8 Transformer (21.7 GB) ──"
if [ -f "${BBUF_DIR}/model.safetensors" ]; then
    echo "  Already exists, skipping."
else
    docker run --rm --network host \
      -v "${MODELS_DIR}:/models:rw" \
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \
      lmsysorg/sglang:latest python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'BBuf/ltx23-two-stage-modelopt-fp8-sglang-transformer',
    local_dir='/models/native/ltx23-fp8-transformer',
)
print('Done')
"
fi

# ── Step 2: Download LTX-2.3 specific files ─────────────────────────
echo ""
echo "── Step 2: LTX-2.3 Spatial Upsampler + Distilled LoRA ──"
if [ ! -d "${MODELS_DIR}/ltx-2.3" ]; then
    docker run --rm --network host \
      -v "${MODELS_DIR}:/models:rw" \
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \
      lmsysorg/sglang:latest python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Lightricks/LTX-2.3',
    local_dir='/models/ltx-2.3',
    allow_patterns=[
        'ltx-2.3-spatial-upscaler-x2-1.0.safetensors',
        'ltx-2.3-spatial-upscaler-x2-1.1.safetensors',
        'ltx-2.3-22b-distilled-lora-384.safetensors',
    ],
)
print('Done')
"
else
    echo "  Already exists, skipping."
fi

# ── Step 3: Download LTX-2 shared components (if needed) ────────────
echo ""
echo "── Step 3: LTX-2 Shared Components (vae, vocoder, tokenizer, etc.) ──"
if [ ! -d "${MODELS_DIR}/ltx-2/vae" ]; then
    docker run --rm --network host \
      -v "${MODELS_DIR}:/models:rw" \
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \
      lmsysorg/sglang:latest python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Lightricks/LTX-2',
    local_dir='/models/ltx-2',
    allow_patterns=[
        'vae/*', 'vocoder/*', 'tokenizer/*',
        'audio_vae/*', 'connectors/*', 'scheduler/*',
        'latent_upsampler/*', 'model_index.json',
    ],
)
print('Done')
"
else
    echo "  Already exists, skipping."
fi

# ── Step 4: Build model directory ───────────────────────────────────
echo ""
echo "── Step 4: Build LTX-2.3 FP8 Directory ──"
mkdir -p "${LTX23_DIR}"

# Symlink shared components from LTX-2 (use container-relative paths)
for item in "${MODELS_DIR}/ltx-2"/*; do
    bn=$(basename "$item")
    [ "$bn" = "transformer" ] && continue
    [ "$bn" = ".cache" ] && continue
    ln -sf "/models/ltx-2/$bn" "${LTX23_DIR}/$bn"
done

# Text encoder: use local Gemma 3 12B BF16
GEMMA_DIR="${MODELS_DIR}/image-gen/comfyui/text_encoders/gemma-3-12b-it"
if [ -d "$GEMMA_DIR" ]; then
    rm -f "${LTX23_DIR}/text_encoder"
    ln -sf "/models/image-gen/comfyui/text_encoders/gemma-3-12b-it" "${LTX23_DIR}/text_encoder"
    echo "  text_encoder → local Gemma 3 12B BF16"
else
    echo "  WARNING: Gemma 3 12B not found at ${GEMMA_DIR}"
    echo "  Will use LTX-2 text_encoder symlink (50GB HF download on first serve)"
fi

# Transformer: BBuf ModelOpt FP8
mkdir -p "${LTX23_DIR}/transformer"
cp "${BBUF_DIR}/config.json" "${LTX23_DIR}/transformer/"
ln -sf "/models/native/ltx23-fp8-transformer/model.safetensors" "${LTX23_DIR}/transformer/model.safetensors"
ln -sf "/models/native/ltx23-fp8-transformer/model.safetensors.index.json" "${LTX23_DIR}/transformer/model.safetensors.index.json"

# LTX-2.3 two-stage components
ln -sf "/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "${LTX23_DIR}/"
ln -sf "/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors" "${LTX23_DIR}/"

echo ""
echo "=== Setup Complete ==="
echo "Model: ${LTX23_DIR}"
echo ""
echo "Components:"
for d in text_encoder tokenizer transformer vae vocoder audio_vae connectors scheduler latent_upsampler; do
    count=$(ls "${LTX23_DIR}/$d/" 2>/dev/null | wc -l)
    echo "  $d/: $count files"
done
echo ""
echo "Serve with: ./serve_ltx23_fp8.sh"
