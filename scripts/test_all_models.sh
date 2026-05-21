#!/usr/bin/env bash
# Comprehensive Forge E2E Test — every model with weights + all Forge services.
# Usage: bash scripts/test_all_models.sh [FORGE_URL] [TIMEOUT]
set -euo pipefail

FORGE_URL="${1:-http://localhost:8000/forge}"
TIMEOUT="${2:-600}"

pass=0
fail=0
results=()

test_model() {
    local name="$1"
    local payload="$2"

    echo ">>> $name"

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
text = d.get('text', '')[:60].replace('\n', ' ')
if media:
    print(f'media={media} data={datalen}')
elif text:
    print(f'text={text}')
else:
    print(f'data={datalen}')
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

# ── Minimal test image (64x64 gradient PNG) ────────────────────────────────
IMG_B64="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAXElEQVR4nO3PAQnAMAAEsV+pf82TcRQCMZBv29152T3bwwRqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6gJ1ARqAjWBmkBNoCZQE6j9r1Ai9BCWhI4AAAAASUVORK5CYII="
# Minimal WAV header (silent, ~0.05s)
AUDIO_B64="UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="

echo "=========================================="
echo " Full Forge E2E — $(date)"
echo " Endpoint: $FORGE_URL  Timeout: ${TIMEOUT}s"
echo "=========================================="
echo ""

# ── CPU Services (no GPU needed) ───────────────────────────────────────────

test_model "kokoro" \
    '{"service":"wan2gp","model_type":"kokoro","text":"Hello world test"}'

test_model "espeak" \
    '{"service":"wan2gp","model_type":"espeak","text":"Hello world test"}'

test_model "faster_whisper" \
    "{\"service\":\"wan2gp\",\"model_type\":\"faster_whisper\",\"audio_b64\":\"$AUDIO_B64\"}"

# ── GPU TTS Services ───────────────────────────────────────────────────────

test_model "faster_qwen3_tts" \
    '{"service":"wan2gp","model_type":"faster_qwen3_tts","text":"Hello world","speaker":"Serena"}'

test_model "ace_step" \
    '{"service":"wan2gp","model_type":"ace_step","text":"gentle piano melody","duration":5}'

test_model "moss_soundeffect" \
    '{"service":"wan2gp","model_type":"moss","text":"gentle rain falling on leaves","duration":2}'

test_model "moss_tts" \
    '{"service":"wan2gp","model":"moss/moss-tts","text":"Hello from MOSS TTS"}'

test_model "moss_tts_nano" \
    '{"service":"wan2gp","model":"moss/moss-tts-nano","text":"Hello from MOSS nano"}'

test_model "moss_tts_realtime" \
    '{"service":"wan2gp","model":"moss/moss-tts-realtime","text":"Hello realtime"}'

test_model "moss_tts_local_transformer" \
    '{"service":"wan2gp","model":"moss/moss-tts-local-transformer","text":"Hello local transformer"}'

test_model "moss_voicegenerator" \
    '{"service":"wan2gp","model":"moss/moss-voicegenerator","text":"Hello voice generator"}'

test_model "vibevoice_tts" \
    '{"service":"wan2gp","model_type":"vibevoice_tts","text":"Hello from VibeVoice"}'

test_model "vibevoice_asr" \
    "{\"service\":\"wan2gp\",\"model_type\":\"vibevoice_asr\",\"audio_b64\":\"$AUDIO_B64\"}"

# ── GPU Motion / 3D Services ───────────────────────────────────────────────

test_model "hy_motion" \
    '{"service":"wan2gp","model_type":"hy_motion","text":"a person waves hello"}'

test_model "hy_motion_lite" \
    '{"service":"wan2gp","model":"hy_motion/hy-motion-1.0-lite","text":"a person jumps"}'

test_model "trellis" \
    "{\"service\":\"wan2gp\",\"model_type\":\"trellis\",\"image_b64\":\"$IMG_B64\",\"steps\":4}"

test_model "anigen" \
    "{\"service\":\"wan2gp\",\"model_type\":\"anigen\",\"image_b64\":\"$IMG_B64\"}"

test_model "see_through" \
    "{\"service\":\"wan2gp\",\"model_type\":\"see_through\",\"image_b64\":\"$IMG_B64\"}"

# ── GPU Image / Video Services ─────────────────────────────────────────────

test_model "wan_t2v" \
    '{"service":"wan2gp","model":"wan/t2v","text":"a cat walking","duration":1}'

test_model "wan_t2v_2_2" \
    '{"service":"wan2gp","model":"wan/t2v_2_2","text":"a dog running","duration":1}'

test_model "lance_image" \
    '{"service":"wan2gp","model":"lance/lance-image-awq","text":"a beautiful sunset"}'

test_model "lance_video" \
    '{"service":"wan2gp","model":"lance/lance-video-awq","text":"a car driving","duration":1}'

# ── Other Forge Services (non-wan2gp) ──────────────────────────────────────

test_model "llm" \
    '{"service":"llm","messages":[{"role":"user","content":"Say hello in 3 words"}]}'

# ── Model Switching ────────────────────────────────────────────────────────

echo ">>> model_switch (moss→hy_motion→moss)"
r1=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"moss/moss-soundeffect","text":"switch test 1","duration":1}' 2>&1)
s1=$(echo "$r1" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
r2=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"hy_motion/hy-motion-1.0","text":"switch test 2"}' 2>&1)
s2=$(echo "$r2" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
r3=$(curl -s --max-time "$TIMEOUT" "$FORGE_URL" \
    -H 'Content-Type: application/json' \
    -d '{"service":"wan2gp","model":"moss/moss-soundeffect","text":"switch test 3","duration":1}' 2>&1)
s3=$(echo "$r3" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null || echo "error")
ok() { [ "$1" = "success" ] || [ "$1" = "ok" ]; }
if ok "$s1" && ok "$s2" && ok "$s3"; then
    echo "    PASS: moss→hy_motion→moss"
    results+=("PASS:model_switch:moss→hy_motion→moss")
    ((pass++)) || true
else
    echo "    FAIL: moss=$s1 hy_motion=$s2 moss=$s3"
    results+=("FAIL:model_switch:moss=$s1 hy_motion=$s2 moss=$s3")
    ((fail++)) || true
fi

echo ""
echo "=========================================="
echo " RESULTS"
echo "=========================================="
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  PASS: $pass  FAIL: $fail  TOTAL: $((pass + fail))"
[ "$fail" -eq 0 ] && echo "  ALL PASS" || echo "  SOME FAILURES"
