#!/usr/bin/env bash
# Forge E2E Test Suite — tests all services through the Forge endpoint.
# Usage: bash scripts/test_forge_e2e.sh [FORGE_URL] [TIMEOUT]
set -euo pipefail

FORGE_URL="${1:-http://localhost:8000/forge}"
TIMEOUT="${2:-600}"

pass=0
fail=0
skip=0
results=()

test_service() {
    local name="$1"
    local payload="$2"
    local expect="${3:-}"

    echo ">>> Testing: $name"

    result=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
        -H 'Content-Type: application/json' \
        -d "$payload" 2>&1) || true

    if [ -z "$result" ]; then
        echo "    FAIL: empty response (timeout/crash)"
        results+=("FAIL:$name:empty response")
        ((fail++)) || true
        return
    fi

    status=$(echo "$result" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(d.get('status', '?'))
" 2>/dev/null) || status="parse_error"

    if [ "$status" = "success" ] || [ "$status" = "ok" ]; then
        detail=$(echo "$result" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
media = d.get('media_type', d.get('text', '')[:50])
datalen = len(d.get('data', ''))
print(f'media={media} data_len={datalen}')
" 2>/dev/null)
        echo "    PASS: $detail"
        results+=("PASS:$name:$detail")
        ((pass++)) || true
    else
        err=$(echo "$result" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(d.get('error', d.get('message', str(d)[:200]))[:200])
" 2>/dev/null)
        echo "    FAIL: $err"
        results+=("FAIL:$name:$err")
        ((fail++)) || true
    fi
}

echo "======================================"
echo " Forge E2E Tests — $(date)"
echo " Endpoint: $FORGE_URL"
echo " Timeout: ${TIMEOUT}s"
echo "======================================"
echo ""

# ── CPU Services (via wan2gp) ──────────────────────────────────────────────

test_service "kokoro" \
    '{"service":"wan2gp","model_type":"kokoro","text":"Testing one two three"}'

test_service "espeak" \
    '{"service":"wan2gp","model_type":"espeak","text":"Hello world test"}'

test_service "faster_whisper" \
    '{"service":"wan2gp","model_type":"faster_whisper","audio_b64":"UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="}'

# ── GPU Services (via wan2gp) ─────────────────────────────────────────────

# faster_qwen3_tts test removed — engine dropped (MOSS VoiceGenerator replaces)

test_service "ace_step" \
    '{"service":"wan2gp","model_type":"ace_step","text":"gentle piano melody","duration":5}'

test_service "moss" \
    '{"service":"wan2gp","model_type":"moss","text":"gentle rain falling on leaves","duration":2}'

test_service "hy_motion" \
    '{"service":"wan2gp","model_type":"hy_motion","text":"a person waves hello"}'

# TRELLIS — needs a real-ish image. 64x64 gradient PNG (minimal).
test_service "trellis" \
    '{"service":"wan2gp","model_type":"trellis","image_b64":"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAXElEQVR4nO3PAQnAMAAEsV+pf82TcRQCMZBv29152T3bwwRqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6j9r1Ai9BCWhI4AAAAASUVORK5CYII=","steps":4}'

# ── Model Switching (regression test) ─────────────────────────────────────

echo ">>> Testing: model_switch (moss→hy_motion→moss)"
switch_fail=0
r1=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"moss/moss-soundeffect","text":"test switch 1","duration":1.0}' 2>&1)
s1=$(echo "$r1" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
r2=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"hy_motion/hy-motion-1.0","text":"test switch 2"}' 2>&1)
s2=$(echo "$r2" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
r3=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"moss/moss-soundeffect","text":"test switch 3","duration":1.0}' 2>&1)
s3=$(echo "$r3" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
ok() { [ "$1" = "success" ] || [ "$1" = "ok" ]; }
if ok "$s1" && ok "$s2" && ok "$s3"; then
    echo "    PASS: moss→hy_motion→moss all success"
    results+=("PASS:model_switch:moss→hy_motion→moss")
    ((pass++)) || true
else
    echo "    FAIL: moss=$s1 hy_motion=$s2 moss=$s3"
    results+=("FAIL:model_switch:moss=$s1 hy_motion=$s2 moss=$s3")
    ((fail++)) || true
fi

# ── LLM (via llama.cpp subprocess) ────────────────────────────────────────

test_service "llm" \
    '{"service":"llm","messages":[{"role":"user","content":"Say hello in 3 words"}]}'

echo ""
echo "======================================"
echo " RESULTS"
echo "======================================"
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  PASS: $pass  FAIL: $fail  TOTAL: $((pass + fail))"
