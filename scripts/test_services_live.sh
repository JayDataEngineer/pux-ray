#!/usr/bin/env bash
# Integration test: hit every service via Ray Serve route prefixes.
# Run on the Ray server with port-forward active:
#   kubectl port-forward -n ai-services svc/tech-noir-ray-serve-svc 18080:8000 &
#   bash scripts/test_services_live.sh
set -euo pipefail

BASE="http://localhost:18080"
PASS=0
FAIL=0
RESULTS=()

test_service() {
    local name="$1"
    local route="$2"
    local payload="$3"

    printf "  %-25s " "$name"
    resp=$(curl -s --max-time 120 -X POST "${BASE}${route}" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1) || resp="CURL_TIMEOUT"

    status=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('status', 'NO_STATUS'))
except:
    print('NOT_JSON')
" 2>/dev/null)

    if [ "$status" = "success" ]; then
        echo "PASS ($status)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS $name")
    else
        echo "FAIL ($status)"
        echo "    $(echo "$resp" | head -c 200)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL $name — $status")
    fi
}

echo "=========================================="
echo " Tech Noir Live Integration Test"
echo " Base URL: $BASE"
echo "=========================================="
echo ""

# ── CPU Services (always running on head) ──
echo "── CPU Services ──────────────────────────"
test_service "kokoro" "/tts/kokoro/" \
    '{"action":"generate","input":{"text":"Hello world"}}'

test_service "espeak" "/tts/espeak/" \
    '{"action":"generate","input":{"text":"Integration test"}}'

test_service "faster_whisper" "/asr/whisper/" \
    '{"action":"generate","input":{"audio_b64":"UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="}}'

echo ""
echo "── GPU Services ──────────────────────────"

test_service "index_tts" "/tts/index-tts/" \
    '{"action":"generate","input":{"text":"Hello world"},"config":{"low_resource":true}}'

test_service "qwen_tts" "/tts/qwen-tts/" \
    '{"action":"generate","input":{"text":"Hello world"},"config":{"low_resource":true}}'

test_service "vibevoice" "/tts/vibevoice/" \
    '{"action":"generate","input":{"text":"Hello world"},"config":{"low_resource":true}}'

test_service "gpt_sovits" "/tts/gpt-sovits/" \
    '{"action":"generate","input":{"text":"Hello world"},"config":{"low_resource":true}}'

test_service "vibevoice_asr" "/asr/vibevoice/" \
    '{"action":"generate","input":{"audio_b64":"UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="},"config":{"low_resource":true}}'

test_service "qwen_asr" "/asr/qwen/" \
    '{"action":"generate","input":{"audio_b64":"UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="},"config":{"low_resource":true}}'

test_service "moss_sfx" "/audio/moss-sfx/" \
    '{"action":"generate","input":{"prompt":"thunder clap"},"config":{"low_resource":true}}'

test_service "trellis" "/3d/trellis/" \
    '{"action":"generate","input":{"image_b64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},"config":{"low_resource":true}}'

test_service "hy_motion" "/3d/hy-motion/" \
    '{"action":"generate","input":{"text":"walk forward"},"config":{"low_resource":true}}'

test_service "anigen" "/3d/anigen/" \
    '{"action":"generate","input":{"image_b64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},"config":{"low_resource":true}}'

test_service "ace_step" "/music/ace-step/" \
    '{"action":"generate","input":{"prompt":"ambient pad"},"config":{"low_resource":true}}'

test_service "see_through" "/creative/see-through/" \
    '{"action":"generate","input":{"image_b64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},"config":{"low_resource":true}}'

test_service "florence2" "/vision/florence2/" \
    '{"action":"generate","input":{"image_b64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},"config":{"low_resource":true}}'

test_service "phi4mm" "/multimodal/phi4mm/" \
    '{"action":"generate","input":{"text":"What is 2+2?"},"config":{"low_resource":true}}'

echo ""
echo "=========================================="
echo " Results: $PASS passed, $FAIL failed"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
