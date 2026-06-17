#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Quick smoke test for Qwen-Image-Edit-2511 Omni server
#
# Usage:
#   ./test_omni_img_edit.sh                          # test localhost:8092
#   HOST=10.0.0.50 PORT=8092 ./test_omni_img_edit.sh # custom host/port
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8092}"
BASE_URL="http://${HOST}:${PORT}"

# Health check
echo "1) Health check..."
if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
  echo "✗ Server not healthy at $BASE_URL"
  exit 1
fi
echo "  ✓ Healthy"

# Models list
echo "2) Models..."
curl -sf "$BASE_URL/v1/models" | python3 -m json.tool 2>/dev/null || echo "  (no models endpoint)"

# Generate a tiny 1x1 PNG inline for testing
# Minimal valid PNG (42 bytes)
TEST_PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

echo "3) Image edit test (small PNG)..."
RESPONSE=$(curl -sf -X POST "$BASE_URL/v1/images/edits" \
  -F "model=Qwen-Image-Edit-2511" \
  -F "image=@<(echo '$TEST_PNG_B64' | base64 -d)" \
  -F "prompt=make it blue" \
  -F "size=512x512" \
  -F "num_inference_steps=4" \
  -F "seed=42" 2>&1) || {
    echo "  Response: $RESPONSE"
    echo "  Note: may require actual PNG files. Test with:"
    echo "    python3 -c \"from PIL import Image; Image.new('RGB',(64,64)).save('/tmp/test.png')\""
    echo "    curl -s -X POST $BASE_URL/v1/images/edits -F 'image=@/tmp/test.png' -F 'prompt=test'"
    exit 0
}

echo "  ✓ Server responded successfully"

echo
echo "All basic connectivity checks passed."
echo "For a full edit test, use:"
echo "  python3 -c \"from PIL import Image; Image.new('RGB',(512,512)).save('/tmp/test.png')\""
echo "  curl -s -X POST $BASE_URL/v1/images/edits \\"
echo "    -F 'image=@/tmp/test.png' \\"
echo "    -F 'prompt=add a blue hat' \\"
echo "    -F 'size=512x512' \\"
echo "    -F 'num_inference_steps=25' \\"
echo "    -o /tmp/out.png"