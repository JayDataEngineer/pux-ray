# Inference Engine Profiling Results

Comprehensive benchmark and configuration reference for every inference engine
in the Tech Noir pool system. Updated: 2026-06-17.

---

## Quick Reference

| Engine | Status | Latency | VRAM | Port |
|--------|--------|---------|------|------|
| **Qwen-Image-Edit** (20B DiT) | ✅ Working | ~46s @512px | 21.2 GB | 8093 |
| **MOSS SoundEffect-v2** | ✅ Working | 4.8s warm | 13 GB | 8050 |
| **CrispASR** (whisper base) | ✅ Working | 71ms | 200 MB | 8051 |
| **Z-Image-Turbo** (omni-vllm) | ❌ Blocked | — | — | 8094 |
| **Z-Image-Turbo** (SGLang) | ⏳ Not tested | — | — | 8081 |
| **ACE-Step** | ⏳ Building | — | — | 8056 |
| **MOSS TTS** | ⚠️ Model format issue | — | — | 8050 |
| **Wan-VACE** (14B) | ⏳ Not tested | — | ~15 GB | 8093 |
| **Ideogram4** (SGLang) | ⏳ Not tested | — | ~16 GB | 8081 |
| **Cosmos** (omni-vllm) | ⏳ Not tested | — | ~4 GB | 8093 |

---

## Qwen-Image-Edit-2511 (20B MMDiT) — vLLM-Omni

**Status:** ✅ WORKING
**Container:** `omni-qwen-img-edit-fp8` (or `inference-omni-vllm`)
**Pool:** omni-vllm (Tier B, port 8093)

### Required Configuration

```bash
# Image (IMPORTANT: use latest, NOT fork-v1)
IMAGE="vllm/vllm-omni:latest"

# Pipeline patch (CRITICAL — makes FP8 work)
-v scripts/pipeline_qwen_image_edit_plus_patch.py:/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py:ro

# Launcher (for env-override patches)
-v scripts/launch_qwen_img_edit_fp8.py:/launcher.py:ro

# Env vars
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-e VLLM_BATCH_INVARIANT=1
-e DIFFUSION_VAE_USE_SLICING=1
-e DIFFUSION_VAE_USE_TILING=1
-e DIFFUSION_CACHE_BACKEND=cache_dit
-e DIFFUSION_CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"enable_taylorseer":true}'

# Model path (base FP8 weights + modelopt transformer overlays)
-v /mnt/data/models/image-gen/qwen-image-edit/2511-fp8:/models/qwen-img-edit-fp8:ro
# NOTE: modelopt overlay is NOT needed when using pipeline patch
# The pipeline patch handles FP8 weight-only natively
```

### VRAM Budget
```
DiT (20B FP8, 60 blocks)       20 GB  (stays resident)
VAE (with tiling)              0.3 GB
Activations + temp buffers      3 GB
Text encoder (CPU)              0 GB  (CPU RAM, moved after prefill)
Headroom                       0.7 GB
──────────────────────────────────────
Total on GPU:                  23 GB  ✓ fits on 24 GB
```

### Benchmarks (512×512, 1 image)

| Steps | Latency | Output | Notes |
|-------|---------|--------|-------|
| 4 | 46,113 ms | 787 KB PNG | Cache-DiT warmup dominates |
| 20 | 46,369 ms | 787 KB PNG | Same latency due to block caching |

Both step counts show similar latency because the 4-step Cache-DiT warmup phase
dominates total wall time. After warmup, subsequent steps reuse cached block outputs.

### API
```bash
# Edit (inpainting / image editing)
curl -X POST http://localhost:8093/v1/images/edits \
  -F "image=@input.png" \
  -F "prompt=your instruction" \
  -F "n=1" -F "size=512x512"

# Generate (text-to-image)
curl -X POST http://localhost:8093/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/qwen-img-edit-fp8","prompt":"...","n":1,"size":"512x512"}'
```

### Known Issues
- **Cache-DiT warning:** `Failed to refresh cache; requires num_inference_steps` — minor, inference works
- **Cold start:** First request after container start takes longer (model already loaded, but CUDA graphs need compilation)

---

## MOSS Audio (SoundEffect + TTS)

**Status:** ✅ SoundEffect-v2 WORKING | ⚠️ TTS blocked (model format)
**Container:** `inference-moss`
**Pool:** moss (Tier A, port 8050)

### Required Setup

```bash
IMAGE="forge-reg.local:30500/tech-noir/moss:latest"

# Model volume
-v /mnt/data/models/audio:/models/audio

# Runtime fixes (container is missing these)
docker exec inference-moss pip install diffusers
docker exec inference-moss apt-get install -y build-essential  # for Triton JIT
```

### Available Models

| Model | Path | Status | Notes |
|-------|------|--------|-------|
| moss-soundeffect-v2 | `/models/audio/moss-soundeffect-v2` | ✅ Working | Has `model_index.json` |
| moss-soundeffect | `/models/audio/moss-soundeffect` | ❌ Wrong format | Has `bf16/` subdir, no pipeline |
| moss-tts | `/models/audio/moss-tts` | ❌ No `model_index.json` | Custom model, not diffusers pipeline |
| moss-tts-realtime | `/models/audio/moss-tts-realtime` | ❌ Not tested | |
| moss-voicegenerator | `/models/audio/moss-voicegenerator` | ❌ Not tested | |

### Benchmarks (SoundEffect-v2, 3s audio)

| Run | Latency | Notes |
|-----|---------|-------|
| Model load | 12.7 s | Loads into VRAM |
| Cold inference | 53.7 s | Triton JIT compilation |
| Warm inference | 4.8 s | Pure inference, ~1.5× realtime |

VRAM usage: ~13 GB (53% of 24 GB)

### API
```bash
# Load model
curl -X POST http://localhost:8050/load \
  -H "Content-Type: application/json" \
  -d '{"model": "moss-soundeffect-v2"}'

# Generate sound
curl -X POST http://localhost:8050/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "rain on tin roof", "model": "moss-soundeffect-v2", "seconds": 3, "seed": 42}'

# Health
curl http://localhost:8050/health
```

---

## CrispASR (Speech Recognition & Diarization)

**Status:** ✅ WORKING (whisper backend) | ⚠️ VibeVoice diarization needs model download
**Container:** `inference-diarization`
**Pool:** diarization (Tier A, port 8051)

### Required Configuration

```bash
# Use upstream image directly (our Dockerfile.asr is a thin wrapper)
IMAGE="ghcr.io/crispstrobe/crispasr:main-cuda-12"

# For auto-download of default model
docker run --gpus all \
  -e CRISPASR_AUTO_DOWNLOAD=1 \
  -p 8051:8080 \
  ghcr.io/crispstrobe/crispasr:main-cuda-12 \
  crispasr --server --host 0.0.0.0 --port 8080 -m auto --backend whisper --auto-download
```

### Available Backends
```
whisper, nemotron, parakeet, canary, lfm2-audio, mini-omni2, cohere,
granite, granite-4.1, granite-4.1-plus, granite-4.1-nar, voxtral,
voxtral4b, qwen3, qwen3-1.7b, mega-asr, fastconformer-ctc, wav2vec2, ...
```

### VibeVoice ASR Model Issue
The GGUFs at `/mnt/data/models/vibevoice-cpp/`:
- `vibevoice-asr-q8_0.gguf` (13 GB) — **TTS-only model** (no ASR encoder tensors)
- `vibevoice-asr-q4_k.gguf` (9.7 GB) — base ASR/diarization model
- `vibevoice-realtime-0.5B-q8_0.gguf` (1.6 GB) — low-latency streaming variant

Error when used with the wrong backend:
```
error: 'model.gguf' is a TTS-only model (no at_enc.*/st_enc.* tensors).
Use --backend vibevoice-tts for this model
```

**Note:** The error message is misleading. `--backend vibevoice-tts` does NOT
do TTS — it is a misnamed ASR backend for the vibevoice-asr model (the "base"
diarization tier). TTS is not a CrispASR feature; use OpenMOSS for TTS. The
"turbo" diarization tier is triton + pyannote3 (a separate stack, not CrispASR).

**Fix:** For ASR/diarization, use `--backend vibevoice` with `vibevoice-asr-*.gguf`.
For TTS, use OpenMOSS. For whisper, the whisper backend auto-downloads on first run.

### Benchmarks

| Audio Input | Duration | Latency | Transcription |
|-------------|----------|---------|-------------|
| 440 Hz sine wave | 2 s | 74 ms | (empty — no speech) |
| TTS sample (speech) | ~5 s | 71 ms | Accurate text |

### API
```bash
# Transcription (OpenAI-compatible)
curl -X POST http://localhost:8051/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "model=whisper"

# Hot-swap model
curl -X POST http://localhost:8051/load \
  -H "Content-Type: application/json" \
  -d '{"backend": "parakeet", "model": "/models/asr/model.gguf"}'
```

---

## Z-Image-Turbo (W8A8 Block FP8)

**Status:** ❌ BLOCKED on omni-vllm
**Container:** Would be `omni-z-image-fp8` on port 8094
**Pool:** omni-vllm (Tier B primary), sglang (Tier C fallback)

### Blockers

1. **Triton 3.6.0 fp8e4nv** (`fork-v1` image, `_w8a8_triton_block_scaled_mm`)
2. **Fork start method** (`fork-v1` image, `multiproc_executor.py:191`)
3. **No pipeline patch** exists for z-image (unlike qwen which has `pipeline_qwen_image_edit_plus_patch.py`)

### Model Files

| File | Size | Role |
|------|------|------|
| `/mnt/data/models/native/z-image-turbo-fp8/` | 26 GB | Model root |
| `transformer/config.json` | — | `quant_method: "fp8"`, W8A8 Block FP8 config |
| `transformer/diffusion_pytorch_model-*.safetensors` | 5 shards | Weights |

### Unblocking Approaches

**A. Create pipeline patch (recommended)** — Follow qwen pattern:
1. Create `pipeline_z_image_plus_patch.py` that monkey-patches Fp8Config → weight-only
2. Bind-mount over the in-image pipeline file
3. Use `vllm/vllm-omni:latest` (not fork-v1)

**B. ModelOpt conversion** — Run through user's conversion pipeline to produce
ModelOpt-format checkpoint (ltx23-fp8-transformer pattern)

**C. ComfyUI fallback** — Use existing ComfyUI infrastructure if available

**D. SGLang path** — SGLang has working z-image benchmark (1.61s @ 8 steps),
but user prefers omni-vllm only for z-image.

---

## ACE-Step (Music Generation)

**Status:** ⏳ Docker image building, GGUF models pending download
**Container:** Would be `inference-ace-step` on port 8056
**Pool:** ace-step (Tier A)

### Required Setup

```bash
# 1. Build Docker image (CUDA 12.8, CMake 3.31+)
docker build -f infra/docker/Dockerfile.acetep.fixed \
  -t forge-reg.local:30500/tech-noir/ace-step:latest .

# 2. Download GGUF models from HF
#    hf://Serveurperso/ACE-Step-1.5-GGUF/
#    Files needed:
#      - acestep-v15-sft-Q8_0.gguf    (SFT DiT, 50-step)
#      - acestep-v15-turbo-Q8_0.gguf  (Turbo DiT, 8-step)
#      - acestep-5Hz-lm-1.7B-Q8_0.gguf (LM model)
#      - Qwen3-Embedding-0.6B-Q8_0.gguf (text encoder)
#      - vae-BF16.gguf                (VAE)

# 3. Run container
docker run -d --gpus all \
  -v /mnt/data/models/audio/acestep-cpp:/models/audio/acestep-cpp \
  -p 8056:8080 \
  --name inference-ace-step \
  forge-reg.local:30500/tech-noir/ace-step:latest
```

### Model Files on Disk
- `/mnt/data/models/audio/acestep/` (9.4 GB) — safetensors format (NOT GGUF)
  - `acestep-5Hz-lm-1.7B/` — LM model (safetensors)
  - `acestep-v15-turbo/` — Turbo DiT (safetensors)
  - `vae/` — VAE (safetensors)
  - `Qwen3-Embedding-0.6B/` — Text encoder

These are Python/diffusers format files. The C++ server (acestep.cpp) needs GGUF format.

### API (once running)
```bash
# Step 1: Generate music codes
curl -X POST http://localhost:8056/lm \
  -H "Content-Type: application/json" \
  -d '{"caption": "upbeat electronic", "lyrics": "", "dit_model": "acestep-v15-turbo-Q8_0"}'

# Step 2: Render audio
curl -X POST http://localhost:8056/synth \
  -H "Content-Type: application/json" \
  -d '{"codes": [...from step 1...], "vae_model": "vae-BF16"}'
```

---

## SGLang (High-throughput serving)

**Status:** ⏳ Not tested
**Container:** Would be `inference-sglang` on port 8081
**Pool:** sglang (Tier C)

### Models

| Model | Config | Notes |
|-------|--------|-------|
| ideogram4 | NF4 via bitsandbytes | Needs HF_TOKEN, 16 GB VRAM |
| z-image-turbo | FP8 fallback | User prefers omni-vllm primary |
| z-image-base | FP8 fallback | Non-distilled weights |
| ltx-video | ModelOpt FP8 | Two-stage pipeline |

### Benchmark (from pool config)
```yaml
z-image:
  benchmark:
    - steps: 8
      time_s: 1.61
      note: "RTX 4090, beats MI300X"
```

---

## Diffusers (Catch-all)

**Status:** ⏳ Not tested
**Pool:** diffusers (Tier D, port 8095)

### Models

| Model | Type | VRAM Estimate |
|-------|------|--------------|
| kimodo | 3D motion generation | ~4 GB |
| kokoro | TTS (sherpa-onnx, CPU-only, 53 voices EN+ZH) | 0 GB (CPU ONNX Runtime) |
| see-through | Anime layer decomposition | ~2 GB |
