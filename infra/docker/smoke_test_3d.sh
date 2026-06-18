#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# smoke_test_3d.sh — quick end-to-end test for the Diffusers-tier services.
# Verifies Kimodo (motion), HY-Motion (motion), TRELLIS.2 (image-to-3D),
# and See-Through (anime layer decomposition → PSD).
# Run after serve_kimodo.sh / serve_hymotion.sh / serve_trellis2.sh /
#       serve_seethrough.sh.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

KIMODO_PORT="${KIMODO_PORT:-8098}"
HYMOTION_PORT="${HYMOTION_PORT:-8097}"
TRELLIS_PORT="${TRELLIS_PORT:-8099}"
SEETHROUGH_PORT="${SEETHROUGH_PORT:-8100}"
OUTDIR="${OUTDIR:-/tmp/3d_smoke}"
mkdir -p "$OUTDIR"

echo "═══ Kimodo (SOMA 60f @ 25 steps) ═══"
curl -sS -X POST http://localhost:${KIMODO_PORT}/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a person walks forward slowly","num_frames":60,"num_denoising_steps":25,"seed":42}' \
  -D "${OUTDIR}/kimodo_headers.txt" \
  -o "${OUTDIR}/kimodo.npz"
grep -i 'x-inference' "${OUTDIR}/kimodo_headers.txt"
ls -la "${OUTDIR}/kimodo.npz"
echo ""

echo "═══ HY-Motion Lite (2s motion) ═══"
curl -sS -X POST http://localhost:${HYMOTION_PORT}/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a person waving hello","duration":2.0,"format":"npz"}' \
  -o "${OUTDIR}/hymotion.npz"
ls -la "${OUTDIR}/hymotion.npz"
echo ""

echo "═══ TRELLIS.2 (512³ → GLB) ═══"
TEST_IMG="${TEST_IMG:-/home/user/Documents/programs/ray/infra/repos/TRELLIS.2/assets/example_image/T.png}"
if [ ! -f "$TEST_IMG" ]; then
    echo "Test image not found at $TEST_IMG — skipping Trellis2"
else
    curl -sS -X POST "http://localhost:${TRELLIS_PORT}/generate?resolution=512&decimation=500000&texture_size=2048&seed=42" \
      -F "image=@${TEST_IMG}" \
      -D "${OUTDIR}/trellis_headers.txt" \
      -o "${OUTDIR}/trellis2.glb"
    grep -i 'x-inference' "${OUTDIR}/trellis_headers.txt"
    ls -la "${OUTDIR}/trellis2.glb"
fi
echo ""

echo "═══ See-Through (anime → layered PSD, 1024 res, 15 steps) ═══"
# Use bundled See-Through test image (or override)
ST_TEST_IMG="${ST_TEST_IMG:-/tmp/seethrough_test/test_image.png}"
if [ ! -f "$ST_TEST_IMG" ]; then
    # Fallback to copying from the gpu-all image if not on host
    mkdir -p /tmp/seethrough_test
    docker run --rm -v /tmp/seethrough_test:/out \
      forge-reg.local:30500/tech-noir/gpu-all:latest \
      cp /opt/seethrough/common/assets/test_image.png /out/test_image.png 2>/dev/null
fi
if [ -f "$ST_TEST_IMG" ]; then
    # Use lower resolution + fewer steps for smoke test (faster, ~30-40s)
    curl -sS -X POST "http://localhost:${SEETHROUGH_PORT}/generate" \
      -F "image=@${ST_TEST_IMG}" \
      -F "resolution=1024" \
      -F "resolution_depth=768" \
      -F "inference_steps=15" \
      -F "seed=42" \
      -D "${OUTDIR}/seethrough_headers.txt" \
      -o "${OUTDIR}/seethrough.psd" || echo "  (See-Through failed)"
    grep -i 'x-inference' "${OUTDIR}/seethrough_headers.txt" 2>/dev/null || true
    ls -la "${OUTDIR}/seethrough.psd" 2>/dev/null || true
else
    echo "  Test image unavailable — skipping See-Through"
fi
echo ""

echo "═══ Outputs ═══"
ls -la "${OUTDIR}/"
