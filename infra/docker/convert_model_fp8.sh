#!/bin/bash
# ════════════════════════════════════════════════════════════════
# convert_model_fp8.sh — Convert HF diffusion model to SGLang FP8
# ════════════════════════════════════════════════════════════════
# Uses SGLang's convert_hf_to_fp8.py to pre-quantize transformer
# weights to FP8 (block 128x128, per-channel scales).
#
# The output is a complete diffusers-format model directory with
# FP8 transformer + BF16 text_encoder/vae/tokenizer copied from
# source. SGLang loads this with --quantization fp8.
#
# Usage:
#   ./convert_model_fp8.sh <source_model_dir> <output_dir>
#
# Examples:
#   ./convert_model_fp8.sh /mnt/data/models/native/z-image-turbo /mnt/data/models/native/z-image-turbo-fp8
#   ./convert_model_fp8.sh /mnt/data/models/qwen-image-edit-2511 /mnt/data/models/qwen-image-edit-fp8
# ════════════════════════════════════════════════════════════════

set -euo pipefail

SRC="${1:?Usage: $0 <source_model_dir> <output_dir>}"
DST="${2:?Usage: $0 <source_model_dir> <output_dir>}"

echo "=== FP8 Conversion ==="
echo "Source: $SRC"
echo "Output: $DST"

# Convert transformer weights to FP8
docker run --rm --gpus all \
  -v "${SRC}:/models/src:ro" \
  -v "${DST}:/models/dst:rw" \
  lmsysorg/sglang:latest \
  bash -c '
    # Convert transformer
    python3 -m sglang.multimodal_gen.tools.convert_hf_to_fp8 \
      --model-dir /models/src/transformer \
      --save-dir /models/dst/transformer \
      --strategy block \
      --block-size 128 128 \
      --max-workers 4

    # Copy non-transformer components (text_encoder, vae, tokenizer, etc.)
    for item in /models/src/*; do
      bn=$(basename "$item")
      [ "$bn" != "transformer" ] && cp -r "$item" "/models/dst/$bn"
    done

    # Clean up any stale BF16 files from source symlinks
    cd /models/dst/transformer
    rm -f model-0000*-of-* model.safetensors.index.json diffusion_pytorch_model.safetensors 2>/dev/null || true

    echo "=== Verify ==="
    python3 -c "
import json
with open(\"config.json\") as f:
    cfg = json.load(f)
qc = cfg.get(\"quantization_config\", {})
print(f\"quant_method: {qc.get(\"quant_method\")}\")
print(f\"weight_block_size: {qc.get(\"weight_block_size\")}\")
print(f\"FP8 checkpoint: {\"fp8\" in qc.get(\"quant_method\", \"\")}\")
"
    echo "=== Files ==="
    ls -lh *.safetensors
  '

echo ""
echo "=== Done ==="
echo "Serve with:"
echo "  sglang serve --model-path $DST \\"
echo "    --transformer-weights-path $DST/transformer \\"
echo "    --quantization fp8 --attention-backend fa"
