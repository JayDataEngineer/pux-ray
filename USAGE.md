# Tech Noir Usage Guide

Base URL: `http://100.86.69.57:30080` (Tailscale)
Auth: `X-API-Key: <key>` header or `?api_key=<key>` param (dev mode = no key)

GPU management is automatic. Just call an endpoint — the platform acquires the GPU lease,
loads the model, runs inference, and returns the result. The first request to a heavy GPU
service will be slow (10-30s for model loading). Subsequent requests are fast.

---

## Quick Reference

| What | Endpoint | Method |
|------|----------|--------|
| Health check | `/health` | GET |
| System status (GPU, VRAM) | `/status` | GET |
| List all services | `/v1/services` | GET |
| Service details | `/v1/services/{name}` | GET |
| Dashboard (GPU metrics) | `/dashboard` | GET |
| Studio (GPU switcher UI) | `/studio` | GET |

---

## LLM (Chat Completions)

OpenAI-compatible. Use any OpenAI client library.

```bash
# Chat
curl http://100.86.69.57:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-q5_k_s",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Streaming
curl http://100.86.69.57:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-q5_k_s",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs"}],
    "stream": true
  }'

# List available models
curl http://100.86.69.57:30080/v1/models

# Switch model (loads a different GGUF)
curl http://100.86.69.57:30080/v1/llm/configure \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen3.6-27b-q5_k_s"}'
```

Python:
```python
from openai import OpenAI
client = OpenAI(base_url="http://100.86.69.57:30080/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3.6-27b-q5_k_s",
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(response.choices[0].message.content)
```

---

## Text-to-Speech

```bash
# Kokoro (default, fast CPU, multi-voice)
curl http://100.86.69.57:30080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-01-kokoro", "input": "Hello world"}' \
  --output speech.wav

# eSpeak (instant CPU, many languages)
curl http://100.86.69.57:30080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-01-espeak", "input": "Bonjour le monde"}' \
  --output speech.wav

# Via TNAP generate endpoint
curl http://100.86.69.57:30080/v1/kokoro/generate \
  -H "Content-Type: application/json" \
  -d '{"action": "generate", "input": {"text": "Hello world"}}'
```

Available TTS models: `tts-01-kokoro`, `tts-01-espeak`, `tts-01-vibevoice`

---

## Speech-to-Text (ASR)

```bash
# Whisper (CPU, fast)
curl http://100.86.69.57:30080/v1/audio/transcriptions \
  -F "model=whisper-1" \
  -F "file=@recording.wav"
```

---

## Image Generation (ComfyUI)

ComfyUI is proxied — use its full native API through the ingress:

```bash
# ComfyUI web UI
open http://100.86.69.57:30080/comfyui/

# ComfyUI API (any ComfyUI endpoint works)
curl http://100.86.69.57:30080/comfyui/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": {...your workflow JSON...}}'
```

---

## 3D Generation (TRELLIS, AniGen)

Via the forge master router — heavy GPU services that swap in on demand.

```bash
# TRELLIS — image to 3D mesh
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "trellis", "prompt": "a cat", "image_b64": "..."}'

# AniGen — anime image to rigged 3D
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "anigen", "prompt": "anime character", "image_b64": "..."}'
```

---

## Music & Audio Generation

```bash
# ACE-Step — text to music
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "ace_step", "prompt": "upbeat jazz piano solo"}'

# MOSS-SoundEffect — text to sound effects
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "moss_soundeffect", "prompt": "thunder storm with rain"}'
```

---

## Motion & Layer Decomposition

```bash
# HY-Motion — text to 3D human motion
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "hy_motion", "prompt": "person doing a backflip"}'

# See-Through — anime layer decomposition
curl http://100.86.69.57:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service": "see_through", "prompt": "decompose", "image_b64": "..."}'
```

---

## Generic TNAP Generate

Every service can be called through the generic endpoint:

```
POST /v1/{service}/generate
```

Request body (TNAP format):
```json
{
  "action": "generate",
  "input": {
    "prompt": "...",
    "text": "...",
    "image_b64": "base64-encoded-image",
    "audio_b64": "base64-encoded-audio",
    "model": "model-name",
    "voice": "voice-name",
    "seed": 42,
    "steps": 50,
    "guidance": 7.5,
    "messages": [{"role": "user", "content": "..."}],
    "stream": false
  },
  "config": {
    "precision": "bf16",
    "quantization": null,
    "low_resource": false
  }
}
```

All `input` fields are optional — send only what the service needs.

---

## Admin Endpoints

```bash
# Load a GPU service (pre-warm before first request)
curl -X POST http://100.86.69.57:30080/admin/load \
  -H "Content-Type: application/json" \
  -d '{"service": "trellis"}'

# Unload GPU (free VRAM)
curl -X POST http://100.86.69.57:30080/admin/unload
```

---

## Service Catalog

| Service | Category | GPU | Route |
|---------|----------|-----|-------|
| kokoro | TTS | No | `/v1/kokoro/generate` |
| espeak | TTS | No | `/v1/espeak/generate` |
| faster_qwen3_tts | TTS | Yes | `/v1/faster_qwen3_tts/generate` |
| index_tts | TTS | Yes | `/v1/index_tts/generate` |
| vibevoice_cpp_gpu | TTS+ASR | No | `/v1/vibevoice_cpp_gpu/generate` |
| faster_whisper | ASR | No | `/v1/faster_whisper/generate` |
| llm | LLM | Yes | `/v1/chat/completions` or `/forge` |
| comfyui | Image | Yes | `/comfyui/*` |
| trellis | 3D | Yes | `/forge` |
| anigen | 3D | Yes | `/forge` |
| hy_motion | Motion | Yes | `/forge` |
| ace_step | Music | Yes | `/forge` |
| moss_soundeffect | Audio | Yes | `/forge` |
| see_through | Image | Yes | `/forge` |

---

## How GPU Coordination Works

You don't need to think about it. Here's what happens under the hood:

1. **CPU services** (kokoro, espeak, faster_whisper) — always available, no GPU needed
2. **Lightweight GPU services** (faster_qwen3_tts, index_tts, vibevoice_cpp_gpu) — use
   small VRAM slices, coexist on the GPU without conflicts
3. **Heavy GPU services** (trellis, ace_step, comfyui, llm, etc.) — only one runs at a time.
   The platform automatically unloads the previous model and loads the new one when you
   switch. This swap takes 10-30 seconds on the first request.

The `/studio` page shows which GPU service is currently loaded and lets you manually
switch if needed. The `/dashboard` page shows real-time GPU metrics.

---

## Tailscale URLs

| Service | URL |
|---------|-----|
| API Ingress | http://100.86.69.57:30080 |
| Dashboard | http://100.86.69.57:30080/dashboard |
| Studio | http://100.86.69.57:30080/studio |
| ComfyUI | http://100.86.69.57:30080/comfyui/ |
| Ray Dashboard | http://100.86.69.57:30080/ray-dashboard/ |
| Wan2GP Studio (MCP) | http://100.86.69.57:30080/studio/ |
| Grafana | http://100.86.69.57:30080/grafana |
