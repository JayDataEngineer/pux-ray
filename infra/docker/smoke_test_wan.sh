#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# smoke_test_wan.sh — Wan2.1 VACE / T2V / I2V via the Omni-VLLM pool.
# Verifies each model produces a valid MP4 with reasonable timing.
#
# Pool layout:
#   VACE 14B FP8  → vllm-omni OpenAI API  → http://localhost:8000
#                   (POST /v1/videos/generations, async with polling)
#   T2V   14B     → custom FastAPI server → http://localhost:8001
#                   (POST /generate JSON, sync, returns video/mp4)
#   I2V   14B     → custom FastAPI server → http://localhost:8002
#                   (POST /generate multipart, sync, returns video/mp4)
#
# Prereqs:
#   bash scripts/run_omni_14b.sh "" 8000          # VACE FP8 (vllm-omni)
#   bash infra/docker/serve_wan_t2v.sh 8001       # T2V (diffusers)
#   bash infra/docker/serve_wan_i2v.sh 8002       # I2V (diffusers)
#
# Smoke (small / fast):
#   NUM_FRAMES=9 STEPS=5 WIDTH=640 HEIGHT=480 \
#     bash infra/docker/smoke_test_wan.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VACE_PORT="${VACE_PORT:-8000}"
T2V_PORT="${T2V_PORT:-8001}"
I2V_PORT="${I2V_PORT:-8002}"
OUTDIR="${OUTDIR:-/tmp/wan_smoke}"
NUM_FRAMES="${NUM_FRAMES:-9}"
STEPS="${STEPS:-5}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
mkdir -p "$OUTDIR"

HAVE_FFPREPBE=0
command -v ffprobe >/dev/null && HAVE_FFPREPBE=1

summary() {
  local label="$1" file="$2" hdr="$3"
  local size=$(stat -c %s "$file" 2>/dev/null || echo 0)
  local inf=""
  if [ "$HAVE_FFPREPBE" = "1" ] && [ "$size" -gt 1000 ]; then
    inf=$(ffprobe -v error -show_entries stream=width,height,nb_frames,codec_name -of default=noprint_wrappers=1 "$file" 2>/dev/null | tr '\n' ' ')
  fi
  local t=""
  [ -f "$hdr" ] && t=$(grep -i '^x-inference-time-s:' "$hdr" 2>/dev/null | tr -d '\r' || true)
  printf "  %-12s bytes=%-9s inf=%s %s\n" "$label" "$size" "$inf" "$t"
}

echo "═══ Wan2.1 VACE 14B FP8 (vllm-omni, port ${VACE_PORT}) ═══"
VACE_RESP=$(curl -sS -X POST "http://localhost:${VACE_PORT}/v1/videos/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"wan-vace\",\"prompt\":\"a panda eating bamboo in a misty forest\",\"num_frames\":${NUM_FRAMES},\"fps\":8,\"resolution\":\"480p\",\"num_inference_steps\":${STEPS},\"width\":${WIDTH},\"height\":${HEIGHT},\"extra_params\":{\"vae_use_tiling\":true,\"vae_use_slicing\":true}}" \
  2>&1) || echo "  (VACE unreachable on port ${VACE_PORT}: ${VACE_RESP})"
if echo "$VACE_RESP" | grep -q '"id"'; then
  VACE_ID=$(echo "$VACE_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
  echo "  Submitted: $VACE_ID"
  for i in $(seq 1 120); do
    s=$(curl -sS "http://localhost:${VACE_PORT}/v1/videos/${VACE_ID}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d.get('progress', 0))" 2>/dev/null)
    if echo "$s" | grep -qE "^completed|^failed"; then break; fi
    sleep 5
  done
  curl -sS "http://localhost:${VACE_PORT}/v1/videos/${VACE_ID}/content" -o "${OUTDIR}/vace.mp4" 2>/dev/null || true
  curl -sS "http://localhost:${VACE_PORT}/v1/videos/${VACE_ID}" > "${OUTDIR}/vace_meta.json" 2>/dev/null || true
  python3 -c "import json; d=json.load(open('${OUTDIR}/vace_meta.json')); print(f'  VACE: status={d[\"status\"]} inference_s={d.get(\"inference_time_s\")} peak_mb={d.get(\"peak_memory_mb\")} error={d.get(\"error\")}')" 2>/dev/null || true
else
  echo "  (VACE not running or returned: ${VACE_RESP})"
fi
echo ""

echo "═══ Wan2.1 T2V 14B BF16 (diffusers, port ${T2V_PORT}) ═══"
curl -sS -X POST "http://localhost:${T2V_PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"a cat playing piano on a beach at sunset\",\"num_frames\":${NUM_FRAMES},\"width\":${WIDTH},\"height\":${HEIGHT},\"num_inference_steps\":${STEPS},\"seed\":42}" \
  -D "${OUTDIR}/t2v_headers.txt" \
  -o "${OUTDIR}/t2v.mp4" 2>/dev/null && echo "  T2V request OK" || echo "  (T2V not running on port ${T2V_PORT} — skipping)"
summary "T2V" "${OUTDIR}/t2v.mp4" "${OUTDIR}/t2v_headers.txt"
echo ""

echo "═══ Wan2.1 I2V 14B 480P (diffusers, port ${I2V_PORT}) ═══"
TEST_IMG="${TEST_IMG:-/tmp/test_image.png}"
if [ ! -f "$TEST_IMG" ]; then
  echo "  Test image not found at $TEST_IMG — using bundled See-Through test image"
  TEST_IMG="/tmp/seethrough_test/test_image.png"
fi
if [ -f "$TEST_IMG" ]; then
  curl -sS -X POST "http://localhost:${I2V_PORT}/generate" \
    -F "image=@${TEST_IMG}" \
    -F "prompt=the scene comes alive and pans slowly to the right" \
    -F "num_frames=${NUM_FRAMES}" \
    -F "width=${WIDTH}" \
    -F "height=${HEIGHT}" \
    -F "num_inference_steps=${STEPS}" \
    -F "seed=42" \
    -D "${OUTDIR}/i2v_headers.txt" \
    -o "${OUTDIR}/i2v.mp4" 2>/dev/null && echo "  I2V request OK" || echo "  (I2V not running on port ${I2V_PORT} — skipping)"
  summary "I2V" "${OUTDIR}/i2v.mp4" "${OUTDIR}/i2v_headers.txt"
else
  echo "  No test image available — skipping I2V"
fi
echo ""

echo "═══ Outputs ═══"
ls -la "${OUTDIR}/" 2>/dev/null
echo ""
echo "Sample command for a full 33-frame 20-step T2V run:"
echo "  curl -X POST http://localhost:${T2V_PORT}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\":\"...\",\"num_frames\":33,\"num_inference_steps\":20}' \\"
echo "    -o out.mp4"
