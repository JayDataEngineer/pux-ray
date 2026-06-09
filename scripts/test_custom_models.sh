#!/usr/bin/env bash
# Test all custom Wan2GP models through the Forge API
set -euo pipefail

FORGE="http://10.0.0.107:8000/forge"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

forge_call() {
    local service="$1"
    local payload="$2"
    local timeout="${3:-120}"
    curl -s --max-time "$timeout" "$FORGE" -H "Content-Type: application/json" -d "$payload"
}

echo "========================================"
echo "  Custom Wan2GP Model Tests"
echo "========================================"
echo ""

# ─── TTS Models ────────────────────────────────

echo -n "[1/12] kokoro (CPU TTS) ... "
R=$(forge_call kokoro '{"service":"wan2gp","model":"kokoro","prompt":"Hello from the forge","model_mode":"en"}' 60)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[2/12] espeak (CPU TTS) ... "
R=$(forge_call espeak '{"service":"wan2gp","model":"espeak","prompt":"Testing espeak synthesis","model_mode":"en"}' 60)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[3/12] faster_qwen3_tts (GPU TTS) ... "
R=$(forge_call faster_qwen3_tts '{"service":"wan2gp","model":"faster_qwen3_tts","prompt":"Testing Qwen3 TTS","model_mode":"en"}' 120)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[4/12] vibevoice_asr (GPU ASR) ... "
# ASR needs audio input — just test that the model loads
R=$(forge_call vibevoice_asr '{"service":"wan2gp","model":"vibevoice_asr","prompt":"test"}' 120)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'audio' in str(d).lower() or 'No audio' in str(d), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[5/12] faster_whisper (CPU ASR) ... "
# ASR needs audio — test load
R=$(forge_call faster_whisper '{"service":"wan2gp","model":"faster_whisper","prompt":"test"}' 120)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'audio' in str(d).lower() or 'No audio' in str(d), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[6/12] moss (GPU Audio) ... "
R=$(forge_call moss '{"service":"wan2gp","model":"moss","prompt":"A door slamming shut"}' 120)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

# ─── 3D / Vision Models ───────────────────────

echo -n "[7/12] trellis (GPU 3D) ... "
# TRELLIS needs an image — test that it loads and gives a useful error
R=$(forge_call trellis '{"service":"wan2gp","model":"trellis","prompt":"test","steps":4}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'image' in str(d).lower(), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[8/12] anigen (GPU 3D) ... "
# AniGen needs an image — test load
R=$(forge_call anigen '{"service":"wan2gp","model":"anigen","prompt":"test"}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'image' in str(d).lower(), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[9/12] see_through (GPU Image) ... "
# See-Through needs an anime image — test load
R=$(forge_call see_through '{"service":"wan2gp","model":"see_through","prompt":"test"}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'image' in str(d).lower(), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[10/12] hy_motion (GPU Motion) ... "
R=$(forge_call hy_motion '{"service":"wan2gp","model":"hy_motion","prompt":"A person waving hello","steps":4}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[11/12] pixal3d (GPU 3D) ... "
# Pixal3D needs an image — test load
R=$(forge_call pixal3d '{"service":"wan2gp","model":"pixal3d","prompt":"test"}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' or 'image' in str(d).lower(), d" 2>/dev/null; then
    echo "OK (load test)"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

echo -n "[12/12] kimodo (GPU Motion) ... "
R=$(forge_call kimodo '{"service":"wan2gp","model":"kimodo","prompt":"A person dancing","steps":4}' 180)
if echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok', d" 2>/dev/null; then
    echo "OK"; PASS=$((PASS+1))
else
    echo "FAIL: $(echo "$R" | head -c 200)"; FAIL=$((FAIL+1))
fi

# ─── Summary ──────────────────────────────────
echo ""
echo "========================================"
echo "  RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"
exit $FAIL
