#!/usr/bin/env bash
# Test each model via Forge endpoint. One at a time, report pass/fail.
set -euo pipefail

FORGE_URL="${1:-http://100.86.69.57:30080/forge}"
TIMEOUT="${2:-300}"

pass=0
fail=0
skip=0
results=()

test_model() {
    local name="$1"
    local payload="$2"

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

    if [ "$status" = "success" ]; then
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
echo " Model E2E Tests — $(date)"
echo " Forge: $FORGE_URL"
echo "======================================"
echo ""

# ── CPU Services (no GPU needed) ──────────────────────────────────────────────

# 1. Kokoro TTS (universal 'text' field)
test_model "kokoro" \
    '{"service":"wan2gp","model_type":"kokoro","text":"Testing one two three"}'

# 2. eSpeak TTS
test_model "espeak" \
    '{"service":"wan2gp","model_type":"espeak","text":"Hello world test"}'

# 3. faster_whisper ASR — needs valid audio; test with minimal WAV header
#    (will likely fail with invalid audio — acceptable for smoke test)
test_model "faster_whisper" \
    '{"service":"wan2gp","model_type":"faster_whisper","audio_b64":"UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="}'

# ── GPU Services ──────────────────────────────────────────────────────────────

# 4. faster_qwen3_tts — test removed (engine dropped, MOSS VoiceGenerator replaces)

# 5. TRELLIS 3D — 64x64 gradient PNG (realistic enough for feature extraction)
test_model "trellis" \
    '{"service":"wan2gp","model_type":"trellis","image_b64":"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAXElEQVR4nO3PAQnAMAAEsV+pf82TcRQCMZBv29152T3bwwRqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6j9r1Ai9BCWhI4AAAAASUVORK5CYII=","steps":4}'

# 6. ACE-Step music (aliased from 'ace_step' → tts/ace_step_v1_5)
test_model "ace_step" \
    '{"service":"wan2gp","model_type":"ace_step","text":"gentle piano melody","duration":5}'

# 7. MOSS sound effect
test_model "moss" \
    '{"service":"wan2gp","model_type":"moss","text":"gentle rain falling on leaves"}'

# 8. HY-Motion (text to 3D motion)
test_model "hy_motion" \
    '{"service":"wan2gp","model_type":"hy_motion","text":"a person waves hello"}'

# 9. Model switching (MOSS → HY-Motion → MOSS) — regression test for flush_torch_caches
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
if [ "$s1" = "success" ] && [ "$s2" = "success" ] && [ "$s3" = "success" ]; then
    echo "    PASS: moss→hy_motion→moss all success"
    results+=("PASS:model_switch:moss→hy_motion→moss")
    ((pass++)) || true
else
    echo "    FAIL: moss=$s1 hy_motion=$s2 moss=$s3"
    results+=("FAIL:model_switch:moss=$s1 hy_motion=$s2 moss=$s3")
    ((fail++)) || true
fi

# 10. LLM (via llama.cpp subprocess — needs ~20GB VRAM, will fail if another model is loaded)
test_model "llm" \
    '{"service":"llm","messages":[{"role":"user","content":"Say hello in 3 words"}]}'

# 11. IndexTTS2 (Wan2GP vendor handler fails during load — known issue)
test_model "index_tts2" \
    '{"service":"wan2gp","model_type":"index_tts2","text":"Hello world"}'

echo ""
echo "======================================"
echo " RESULTS"
echo "======================================"
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  PASS: $pass  FAIL: $fail  TOTAL: $((pass + fail))"
