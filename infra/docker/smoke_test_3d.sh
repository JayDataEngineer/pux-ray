#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# smoke_test_3d.sh — quick end-to-end test for the 3 new Diffusers-tier services.
# Verifies Kimodo (motion), HY-Motion (motion), and TREllis.2 (image-to-3D).
# Run after serve_kimodo.sh / serve_hymotion.sh / serve_trellis2.sh.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

KIMODO_PORT="${KIMODO_PORT:-8098}"
HYMOTION_PORT="${HYMOTION_PORT:-8097}"
TRELLIS_PORT="${TRELLIS_PORT:-8099}"
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

echo "═══ Outputs ═══"
ls -la "${OUTDIR}/"
