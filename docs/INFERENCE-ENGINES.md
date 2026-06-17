# Tech Noir — Inference Engines Reference

Single source of truth for which Docker / engine serves each model.
The authoritative config is `config/inference_pools.yaml`.

---

## Engine Overview

| Tier | Pool | Engine / Project | Docker Image | Port (host) | API Style |
|------|------|-----------------|--------------|-------------|-----------|
| A | `moss` | Python MOSS server (our build) | `tech-noir/moss:latest` | 8050 | Custom `/generate` |
| A | `diarization` | CrispASR (crispstrobe/CrispASR) | `tech-noir/asr:latest` | 8051 | OpenAI `/v1/audio/*` |
| A | `diarization-turbo` | CrispASR | `tech-noir/asr:latest` | 8055 | OpenAI `/v1/audio/*` |
| A | `llama` | llama.cpp (upstream) | `tech-noir/gpu-all:latest` | 8052 | OpenAI `/v1/chat/*` |
| A | `llama-bee` | BeeLlama.cpp (Anbeeld fork) | `tech-noir/gpu-all:latest` | 8053 | OpenAI `/v1/chat/*` |
| A | `comfyui` | ComfyUI 0.20.1 | `tech-noir/gpu-all:latest` | 8054 | ComfyUI `/prompt` |
| A | `ace-step` | acestep.cpp (ServeurpersoCom) | `tech-noir/ace-step:latest` | 8056 | Two-step `/lm` + `/synth` |
| B | `omni-vllm` | vLLM-Omni 0.22 | `vllm/vllm-omni:latest` | 8093 | OpenAI `/v1/images/*` `/v1/videos/*` |
| C | `sglang` | SGLang | `lmsysorg/sglang:latest` | 8081 | OpenAI `/v1/images/*` |
| D | `diffusers` | Diffusers (gpu-all catch-all) | `tech-noir/gpu-all:latest` | 8095 | Custom `/v1/generate` |

**Note on OpenMOSS (pwilkin/openmoss):** C++ GGUF port of MOSS-TTS. Planned migration
target for the `moss` pool. Uses `/v1/audio/speech` (OpenAI TTS compat). Must be built
from source — no pre-built Docker image. See `infra/docker/Dockerfile.moss` for current
Python implementation; a `Dockerfile.openmoss` will replace it when ready.

---

## Tier A — Specialized Dockers

### MOSS Audio (`moss` pool, port 8050)
| Item | Value |
|------|-------|
| Source | Our custom build — `infra/docker/Dockerfile.moss` |
| Framework | Python + HuggingFace transformers (torch) |
| Container | `inference-moss` |
| API | `POST /generate` → `{"audio": "<base64 wav>", "sample_rate": 48000}` |
| Health | `GET /health` |
| Switch model | `POST /load {"model": "<name>"}` · `POST /release` |
| Forge adapter | `services/audio/forge_moss.py` |

**Supported models** (all via `MossSoundEffectPipeline`):

| Model key | Path on disk | VRAM | Status |
|-----------|-------------|------|--------|
| `moss-soundeffect-v2` | `audio/moss-soundeffect-v2/` | ~7GB | ✅ Tested |
| `moss-soundeffect` | `audio/moss-soundeffect/bf16/` | ~18GB | ✅ |
| `moss-tts` | `audio/moss-tts/` | ~18GB | ✅ |
| `moss-ttsd` | `audio/moss-ttsd/` | ~18GB | ✅ |
| `moss-voicegenerator` | `audio/moss-voicegenerator/` | ~4GB | ✅ |
| `moss-tts-realtime` | `audio/moss-tts-realtime/` | ~4GB | ✅ |
| `moss-tts-local-transformer` | `audio/moss-tts-local-transformer/` | ~4GB | ✅ |
| `moss-tts-nano` | `audio/moss-tts-nano/` | ~0.2GB | ⚠️ Different pipeline |

---

### CrispASR Diarization (`diarization` / `diarization-turbo` pools)
| Item | Value |
|------|-------|
| Source | `ghcr.io/crispstrobe/crispasr:main-cuda-12` (upstream, unmodified) |
| Our Dockerfile | `infra/docker/Dockerfile.asr` (thin wrapper setting defaults) |
| Framework | C++ GGUF — 26 ASR backends + pyannote/sherpa diarization |
| API | `POST /v1/audio/transcriptions` (OpenAI-compatible, multipart form) |
| Health | `GET /health` |
| Forge adapter | `services/audio/forge_asr.py` |

**Two modes** (different GGUF models, different containers):

| Mode | Pool | Port | Model | Notes |
|------|------|------|-------|-------|
| **base** | `diarization` | 8051 | `vibevoice-cpp/vibevoice-asr-q4_k.gguf` | VibeVoice-7B Q4_K · 9.19% DER · joint ASR+diarization |
| **turbo** | `diarization-turbo` | 8055 | `vibevoice-cpp/vibevoice-realtime-0.5B-q8_0.gguf` | VibeVoice-Realtime 0.5B · low-latency streaming |

**CrispASR request format** (`/v1/audio/transcriptions`):
```
POST multipart/form-data
  file:           audio file (wav/mp3/…)
  model:          model path (optional — uses server default)
  response_format: verbose_json
  diarize:        true
  diarize_method: vibevoice  (or pyannote / sherpa / ecapa)
  language:       en  (optional)
```

---

### llama.cpp / BeeLlama (`llama` / `llama-bee` pools)
| Pool | Port | Binary | Notes |
|------|------|--------|-------|
| `llama` | 8052 | `llama-server-upstream` | Upstream ggml-org/llama.cpp stable |
| `llama-bee` | 8053 | `llama-server` (BeeLlama fork) | Anbeeld fork — DFlash + TurboQuant KV cache |

API: OpenAI-compatible `/v1/chat/completions`, `/v1/completions`.
Models: any GGUF in `llm/` — selected at runtime via `/configure`.

---

### ComfyUI (`comfyui` pool, port 8054)
ComfyUI node-based pipeline. API: `POST /prompt` (workflow JSON graph).
See `config/workflows/comfyui/manifest.yaml` for available workflows.

---

### ACE-Step C++ (`ace-step` pool, port 8056)
| Item | Value |
|------|-------|
| Source | `github.com/ServeurpersoCom/acestep.cpp` (upstream, unmodified) |
| Our Dockerfile | `infra/docker/Dockerfile.acetep` (two-stage CUDA build) |
| Framework | C++ GGML — GGUF models from `hf://Serveurperso/ACE-Step-1.5-GGUF/` |
| API | Two-step: `POST /lm` (caption → music codes) then `POST /synth` (codes → WAV) |
| Health | `GET /health` |
| Models | `GET /props` (lists GGUFs in `--models` dir) |
| Forge adapter | `services/audio/forge_acetep.py` |

**Two variants** (same container, selected per request by DiT model name):

| Variant | Route | DiT GGUF | Steps | Notes |
|---------|-------|----------|-------|-------|
| **turbo** | `ace-step-turbo` | `acestep-v15-turbo-Q8_0` | 8 | Fast music generation |
| **base** | `ace-step` | `acestep-v15-sft-Q8_0` | 50 | High-quality SFT model |

**GGUF model files** (all in `audio/acestep-cpp/`):
| File | Size | Purpose |
|------|------|---------|
| `acestep-5Hz-lm-1.7B-Q8_0.gguf` | 1.9 GB | LM — generates music codes |
| `acestep-v15-turbo-Q8_0.gguf` | 3.9 GB | DiT turbo (8-step distilled) |
| `acestep-v15-sft-Q8_0.gguf` | 3.9 GB | DiT SFT (50-step base) |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 0.7 GB | Text encoder |
| `vae-BF16.gguf` | 0.4 GB | Audio decoder (VAE) |

---

## Tier B — vLLM-Omni (DiT models, port 8093)

One container serves one DiT model at a time (each launcher starts a fresh container).
The container is `inference-omni-vllm`. Launch scripts live in `scripts/`.

| Model route | Variant | Launch script | Model dir on disk | Optimization |
|-------------|---------|--------------|-------------------|-------------|
| `qwen-image-edit` | **base** (20-step) | `run_omni_qwen_img_edit_fp8.sh` | `image-gen/qwen-image-edit/2511-fp8` | FP8 weight-only + Cache-DiT |
| `qwen-image-edit-turbo` | **turbo** (4-step) | `run_omni_qwen_img_edit.sh` | `image-gen/qwen-image-edit/2511-fp8-lightning` | FP8 + Lightning distilled |
| `wan-vace` | **base** (25-step) | `run_omni_14b.sh` | `video/wan2.1-vace-14b-fp8-diffusers` | FP8 weight-only + TeaCache |
| `wan-vace-turbo` | **turbo** (4-step) | `run_omni_14b_lightning.sh` | `video/wan2.1-vace-14b-fp8-lightning` | FP8 Lightning |
| `z-image` | **turbo** (8-step) | `run_omni_z_image_fp8.sh` | `native/z-image-turbo-fp8` | FP8 + Cache-DiT |
| `z-image-base` | **base** (50-step) | `run_omni_z_image_fp8.sh` | `native/z-image-base-fp8` | Same pipeline, non-distilled weights |
| `wan-t2v` | — | `run_omni_wan_t2v.sh` | (wan2.1 t2v) | — |
| `wan-i2v` | — | `run_omni_wan_i2v.sh` | (wan2.1 i2v) | — |
| `cosmos` | BF16 CPU offload | `run_omni_cosmos.sh` | `cosmos3-nano` | BF16, awaiting FP8 SGLang support |

> **Z-Image note**: `z-image` and `z-image-base` share the same omni-vllm pipeline (same architecture), but use different checkpoints (turbo = 8-step distilled, base = 50-step non-distilled). Both fall back to SGLang when omni-vllm is occupied.

**Patch files** bind-mounted into container (override in-image pipeline py files):
- `scripts/pipeline_qwen_image_edit_plus_patch.py` → Qwen FP8 weight-only + CPU text encoder
- `scripts/pipeline_wan2_2_vace_patch.py` → VACE FP8 weight-only

**API**: OpenAI-compatible images/videos — `POST /v1/images/generations`, `POST /v1/images/edits`, `POST /v1/videos/generations`

---

## Tier C — SGLang (port 8081)

Single `lmsysorg/sglang:latest` container. Each model launcher starts a fresh container.
Container: `inference-sglang`.

| Model route | Variant | Launch script | Model dir | Notes |
|-------------|---------|--------------|-----------|-------|
| `ideogram4` | NF4 | `infra/docker/serve_ideogram4.sh` | HF pull at startup | Typography-aware T2I |
| `z-image` (fallback) | turbo | `infra/docker/serve_zimage_fp8.sh` | `native/z-image-turbo-fp8` | Fallback when omni-vllm busy |
| `z-image-base` (fallback) | base | `infra/docker/serve_zimage_base_fp8.sh` | `native/z-image-base-fp8` | Fallback when omni-vllm busy |
| `ltx-video` | FP8 ModelOpt | `infra/docker/serve_ltx23_fp8.sh` | `native/ltx-2.3-fp8` | Two-stage FP8 |
| `cosmos` (alt) | BF16 CPU | `infra/docker/serve_cosmos3_nano.sh` | `cosmos3-nano` | SGLang fallback |

**API**: `sglang serve` — OpenAI-compatible `/v1/images/generations`, `/v1/videos/generations`

---

## Tier D — Diffusers / Catch-all (port 8095)

`tech-noir/gpu-all:latest` runs small GPU + CPU models. Container: `inference-diffusers`.

| Model route | Type | Notes |
|-------------|------|-------|
| `kimodo` | 3D motion | Viser interactive UI |
| `kokoro` | TTS | CPU-capable |
| `vibevoice-tts` | TTS | VibeVoice-7B TTS |
| `see-through` | Image decomp | Anime layer decomposition |

> **Removed from Tier D**: ACE-Step (now Tier A acestep.cpp), IndexTTS (retired), faster-whisper (now via CrispASR diarization pool).

---

## Model → Engine Quick Reference

| Model | Variant | Tier | Pool | Engine |
|-------|---------|------|------|--------|
| moss-tts | — | A | moss | Python MOSS (→ OpenMOSS C++ planned) |
| moss-ttsd | — | A | moss | Python MOSS |
| moss-soundeffect-v2 | — | A | moss | Python MOSS |
| moss-soundeffect | — | A | moss | Python MOSS |
| moss-tts-realtime | — | A | moss | Python MOSS |
| moss-voicegenerator | — | A | moss | Python MOSS |
| diarization / whisper / faster-whisper | base | A | diarization | CrispASR + VibeVoice-7B Q4_K |
| diarization | turbo | A | diarization-turbo | CrispASR + VibeVoice-Realtime 0.5B |
| ace-step | base (50-step) | A | ace-step | acestep.cpp C++ GGUF |
| ace-step-turbo | turbo (8-step) | A | ace-step | acestep.cpp C++ GGUF |
| llama | — | A | llama | llama.cpp C++ GGUF |
| llama-bee | — | A | llama-bee | BeeLlama.cpp (DFlash) |
| comfyui | — | A | comfyui | ComfyUI Python |
| qwen-image-edit | base (20-step) | B | omni-vllm | vLLM-Omni + FP8 patch |
| qwen-image-edit-turbo | turbo (4-step) | B | omni-vllm | vLLM-Omni + Lightning |
| wan-vace | base (25-step) | B | omni-vllm | vLLM-Omni + FP8 patch + TeaCache |
| wan-vace-turbo | turbo (4-step) | B | omni-vllm | vLLM-Omni + Lightning |
| z-image | turbo (8-step) | B→C | omni-vllm / sglang | vLLM-Omni FP8 (SGLang fallback) |
| z-image-base | base (50-step) | B→C | omni-vllm / sglang | vLLM-Omni FP8 same pipeline, different ckpt (SGLang fallback) |
| cosmos | BF16 CPU offload | B/C | omni-vllm / sglang | BF16 (FP8 blocked on fused LLM params) |
| ideogram4 | NF4 | C | sglang | SGLang NF4 |
| ltx-video | FP8 ModelOpt | C | sglang | SGLang ModelOpt FP8 |
| kimodo | — | D | diffusers | Viser 3D UI |
| kokoro | — | D | diffusers | Kokoro TTS |
| see-through | — | D | diffusers | Anime layer decomposition |

---

## Key Config Files

| File | Purpose |
|------|---------|
| `config/inference_pools.yaml` | Pool definitions, model launchers, routes |
| `config/model_registry.yaml` | All model paths, sizes, sources (for `task models:pull`) |
| `services/inference/dispatch.py` | Logical service → model name mapping |
| `services/inference/config.py` | Pool/Launcher/Route dataclasses |
| `services/inference/manager.py` | Model → pool resolver |
| `services/inference/launcher.py` | Docker container lifecycle |
| `services/audio/forge_moss.py` | Forge adapter → MOSS server HTTP |
| `services/audio/forge_asr.py` | Forge adapter → CrispASR HTTP (diarization + whisper) |
| `services/audio/forge_acetep.py` | Forge adapter → acestep.cpp HTTP (`/lm` + `/synth`) |
| `infra/docker/Dockerfile.asr` | CrispASR Docker (thin wrapper on ghcr.io/crispstrobe/crispasr:main-cuda-12) |
| `infra/docker/Dockerfile.acetep` | ACE-Step C++ Docker (builds acestep.cpp from source) |
| `infra/docker/Dockerfile.moss` | Python MOSS Docker (to be replaced by Dockerfile.openmoss) |

---

## Planned Migrations

| Current | Target | Status |
|---------|--------|--------|
| Python MOSS server (`/generate`) | OpenMOSS C++ GGUF (`/v1/audio/speech`) | 🔜 Pending — no upstream Docker image yet |
| forge_asr.py → K8s ASR service | inference_pools diarization pool | 🔜 In progress |
| forge_moss.py → K8s MOSS service | inference_pools moss pool | 🔜 In progress |
| ACE-Step diffusers (Tier D) | ACE-Step C++ GGUF (Tier A, `acestep.cpp`) | ✅ Done — pool wired, Dockerfile ready, forge adapter written |
| Wan2GP custom moss_handler.py | Removed — MOSS runs in dedicated pool | ✅ Done (handler file retained for reference) |
| faster-whisper / whisper service | CrispASR diarization pool | ✅ Done — routed through diarization |
| IndexTTS | Retired — removed from pool and routes | ✅ Done |
