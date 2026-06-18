#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_trellis2.sh — TRELLIS.2-4B (microsoft/TRELLIS.2-4B) image-to-3D
# ═════════════════════════════════════════════════════════════════════════════
# Native Trellis2 pipeline (vendor code mounted at /opt/trellis/), NOT Wan2GP.
# Outputs textured GLB via o_voxel.postprocess.to_glb.
#
# Mounts:
#   /opt/trellis/trellis2       ← vendor/trellis2 (the trellis2 Python package)
#   /mnt/data/models/3d/trellis ← host model root (TRELLIS.2-4B, dinov3, rmbg)
#   /opt/api_trellis2.py        ← FastAPI server
#
# Patches (applied at container start from infra/docker/patches/trellis2/apply.sh):
#   modules/sparse/conv/conv_flex_gemm.py    — adds needs_grad=False arg to _compute_neighbor_cache
#   representations/mesh/base.py             — guards cumesh ops so missing CuMesh doesn't crash
#
# Pipeline types:
#   512          → 512³ voxels (fastest, ~3s H100 / ~15-25s RTX 4090)
#   1024_cascade → 512→1024 cascade (default, ~17s H100 / ~60-90s RTX 4090)
#   1536_cascade → 512→1536 cascade (highest quality, slowest)
#
# Port: 8099 → 8099 (container internal)
# Output: GLB binary (model/gltf-binary) with X-Inference-Time-S header
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8099}"
CONTAINER_NAME="${2:-trellis2}"
PIPELINE_TYPE="${TRELLIS2_PIPELINE_TYPE:-1024_cascade}"
MODEL_ROOT="${TRELLIS2_MODEL_ROOT:-/mnt/data/models/3d/trellis}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="${TRELLIS2_VENDOR_DIR:-$(cd "${SCRIPT_DIR}/../../vendor/trellis2" && pwd)}"

if [ ! -f "${MODEL_ROOT}/TRELLIS.2-4B/ckpts/pipeline.json" ]; then
    echo "ERROR: ${MODEL_ROOT}/TRELLIS.2-4B/ckpts/pipeline.json not found"
    echo "Set TRELLIS2_MODEL_ROOT to the trellis model root."
    exit 1
fi
if [ ! -d "${VENDOR_DIR}/pipelines" ]; then
    echo "ERROR: ${VENDOR_DIR}/pipelines not found (expected vendor/trellis2/pipelines)"
    exit 1
fi

echo "Starting TRELLIS.2-4B on port ${PORT} (pipeline=${PIPELINE_TYPE})..."
echo "  Vendor: ${VENDOR_DIR}"
echo "  Model:  ${MODEL_ROOT}/TRELLIS.2-4B/ckpts/"

docker run -d --name "${CONTAINER_NAME}" --gpus all --restart=no \
  -p ${PORT}:8099 \
  -v "${VENDOR_DIR}:/opt/trellis/trellis2" \
  -v "${MODEL_ROOT}:/mnt/data/models/3d/trellis:ro" \
  -v "${SCRIPT_DIR}/api_trellis2.py:/opt/api_trellis2.py:ro" \
  -v "${SCRIPT_DIR}/patches/trellis2:/opt/patches:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ATTN_BACKEND=flash-attn \
  -e SPCONV_ALGO=native \
  -e OPENCV_IO_ENABLE_OPENEXR=1 \
  -e TRELLIS2_MODEL_PATH="/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts" \
  -e "TRELLIS_PIPELINE_ROOT=/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts" \
  -e "TRELLIS2_PIPELINE_TYPE=${PIPELINE_TYPE}" \
  -e "HF_HOME=/mnt/data/models/cache/huggingface" \
  -e "TRANSFORMERS_CACHE=/mnt/data/models/cache/huggingface" \
  --entrypoint bash \
  forge-reg.local:30500/tech-noir/gpu-all:latest \
  -c '
    set -e
    # Apply vendor patches (idempotent). Patches live in infra/docker/patches/trellis2/.
    chmod -R u+w /opt/trellis/trellis2 2>/dev/null || true
    bash /opt/patches/apply.sh /opt/trellis/trellis2
    exec python3 /opt/api_trellis2.py
  '

echo ""
echo "TRELLIS.2-4B running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test (1024 cascade → GLB):"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -F 'image=@/path/to/input.png' \\"
echo "    -o trellis2.glb"
echo ""
echo "Faster (512³):"
echo "  curl -X POST 'http://localhost:${PORT}/generate?resolution=512&decimation=500000' \\"
echo "    -F 'image=@/path/to/input.png' \\"
echo "    -o trellis2_fast.glb"
