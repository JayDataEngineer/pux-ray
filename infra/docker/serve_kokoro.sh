#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# serve_kokoro.sh — Kokoro TTS via sherpa-onnx (CPU-only, Tier A standalone)
# ═════════════════════════════════════════════════════════════════════════════
# Replaces the previous PyTorch Kokoro handler that ran inside the wan2gp
# deployment. Same OpenAI-compatible /v1/audio/speech API, same 53 voices,
# but in a 600 MB container with no GPU dependencies.
#
# Model (bind-mounted read-only):
#   /mnt/data/models/tts/kokoro-sherpa/
#     ├── model.onnx           (311 MB — kokoro-multi-lang-v1_0)
#     ├── voices.bin            (27 MB — 53 voice embeddings)
#     ├── tokens.txt
#     ├── espeak-ng-data/
#     ├── lexicon-us-en.txt, lexicon-gb-en.txt, lexicon-zh.txt
#     └── date-zh.fst, number-zh.fst, phone-zh.fst
#
# Download (one-time):
#   mkdir -p /mnt/data/models/tts/kokoro-sherpa && cd $_
#   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
#   tar xf kokoro-multi-lang-v1_0.tar.bz2 --strip-components=1 && rm *.tar.bz2
#
# API (OpenAI-compatible):
#   GET  /health
#   GET  /v1/audio/voices             → list of 53 voice names
#   POST /v1/audio/speech             → {"input":"...","voice":"af_bella","speed":1.0}
#   POST /synthesize                  → legacy alias for /v1/audio/speech
#
# Port: 8060 → 8060 (container internal)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="${1:-8060}"
CONTAINER_NAME="${2:-inference-kokoro}"
MODEL_ROOT="${KOKORO_MODEL_ROOT:-/mnt/data/models/tts/kokoro-sherpa}"
IMAGE="${KOKORO_IMAGE:-forge-reg.local:30500/tech-noir/kokoro:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "${MODEL_ROOT}/model.onnx" ]; then
    echo "ERROR: ${MODEL_ROOT}/model.onnx not found"
    echo "Download with:"
    echo "  mkdir -p ${MODEL_ROOT} && cd ${MODEL_ROOT}"
    echo "  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
    echo "  tar xf kokoro-multi-lang-v1_0.tar.bz2 --strip-components=1 && rm *.tar.bz2"
    exit 1
fi

# Stop any existing container with the same name
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Kokoro (sherpa-onnx) on port ${PORT}..."

docker run -d --name "${CONTAINER_NAME}" \
  -p ${PORT}:8060 \
  -v "${MODEL_ROOT}:/models/tts/kokoro-sherpa:ro" \
  -e KOKORO_MODEL_DIR=/models/tts/kokoro-sherpa \
  -e KOKORO_THREADS="${KOKORO_THREADS:-2}" \
  -e KOKORO_PORT=8060 \
  --restart=no \
  "${IMAGE}"

# Wait for /health to come up
echo -n "Waiting for health check on port ${PORT}..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " OK (${i}s)"
        break
    fi
    echo -n "."
    sleep 1
    if [ "$i" = "30" ]; then
        echo " TIMEOUT"
        echo "Container logs:"
        docker logs "${CONTAINER_NAME}" --tail 20
        exit 1
    fi
done

echo ""
echo "Kokoro running on port ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo ""
echo "Test:"
echo "  curl -X POST http://localhost:${PORT}/v1/audio/speech \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\":\"Hello world\",\"voice\":\"af_bella\"}' \\"
echo "    -o speech.wav"
echo ""
echo "List voices:"
echo "  curl http://localhost:${PORT}/v1/audio/voices"
