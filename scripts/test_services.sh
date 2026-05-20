#!/bin/bash
SERVE="http://localhost:30080"
PASS=0
FAIL=0

report() {
    local num="$1" name="$2" code="$3"
    if [ "$code" = "200" ]; then
        echo "  $num. $name: 200 OK"
        PASS=$((PASS + 1))
    else
        echo "  $num. $name: $code FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== CPU Services ==="
report 1 "Kokoro TTS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/kokoro/ -H "Content-Type: application/json" -d '{"text":"Test.","voice":"af_heart"}' --max-time 30)
report 2 "eSpeak TTS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/espeak/ -H "Content-Type: application/json" -d '{"text":"Test."}' --max-time 10)
report 3 "Faster-Whisper ASR" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/asr/whisper/ -F "file=@/tmp/test_silence.wav" --max-time 30)

echo ""
echo "=== GPU Services ==="

# Direct-load services
report 4 "IndexTTS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/index-tts/ -H "Content-Type: application/json" -d '{"text":"Hello test."}' --max-time 300)
report 5 "MOSS-SFX" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/audio/moss-sfx/ -H "Content-Type: application/json" -d '{"prompt":"thunder"}' --max-time 120)
report 6 "Florence-2" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/vision/florence2/ -H "Content-Type: application/json" -d '{"prompt":"describe a cat"}' --max-time 120)
report 7 "VibeVoice ASR" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/asr/vibevoice/ -F "file=@/tmp/test_silence.wav" --max-time 300)
report 8 "Qwen ASR" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/asr/qwen/ -F "file=@/tmp/test_silence.wav" --max-time 300)
report 9 "Qwen-TTS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/qwen-tts/ -H "Content-Type: application/json" -d '{"input":"Hello test."}' --max-time 300)
report 10 "VibeVoice TTS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/vibevoice/ -H "Content-Type: application/json" -d '{"input":"Hello test."}' --max-time 300)

# Subprocess proxy services
report 11 "HY-Motion" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/3d/hy-motion/generate -H "Content-Type: application/json" -d '{"prompt":"a person walking","duration":3}' --max-time 600)
report 12 "TRELLIS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/3d/trellis/generate -H "Content-Type: application/json" -d '{"prompt":"a red cube"}' --max-time 600)
report 13 "AniGen" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/3d/anigen/generate -H "Content-Type: application/json" -d '{"prompt":"a running person"}' --max-time 600)
report 14 "ACE-Step" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/music/ace-step/generate -H "Content-Type: application/json" -d '{"prompt":"jazz piano","duration":5}' --max-time 300)
report 15 "See-Through" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/creative/see-through/decompose -H "Content-Type: application/json" -d '{"prompt":"test"}' --max-time 300)
report 16 "GPT-SoVITS" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/tts/gpt-sovits/tts -H "Content-Type: application/json" -d '{"text":"Hello test.","language":"en"}' --max-time 300)
report 17 "ComfyUI" $(curl -s -o /dev/null -w "%{http_code}" $SERVE/comfyui/ --max-time 30)
report 18 "Phi-4-MM" $(curl -s -o /dev/null -w "%{http_code}" -X POST $SERVE/multimodal/phi4mm/ -H "Content-Type: application/json" -d '{"text":"Describe what you see."}' --max-time 300)

echo ""
echo "=== Results: $PASS passed, $FAIL failed out of $((PASS + FAIL)) ==="
