#!/usr/bin/env bash
# Test Omni VACE 14B generation
set -euo pipefail

echo "Testing Omni VACE 14B generation..."
docker exec omni-14b-vace python3 -c "
import requests, json, time, base64
t0 = time.time()
resp = requests.post('http://localhost:8000/v1/images/generations', json={
    'model': '/models/vace-fp8',
    'prompt': 'A cat walking on a beach, sunset',
    'n': 1, 'size': '832x480', 'num_frames': 33,
    'steps': 18, 'guidance_scale': 5.0, 'fps': 16
}, timeout=600)
elapsed = time.time() - t0
data = resp.json()
b64 = data['data'][0].get('b64_json', '')
print(f'Time: {elapsed:.1f}s | Output: {len(b64)//1024} KB')
with open('/tmp/output.png', 'wb') as f:
    f.write(base64.b64decode(b64))
print('Saved to /tmp/output.png')
"
docker cp omni-14b-vace:/tmp/output.png /tmp/omni_14b_output.png
echo "Output copied to /tmp/omni_14b_output.png"
