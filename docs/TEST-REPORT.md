# Inference Test Report

**Date:** 2026-06-18 (updated)  
**GPU:** NVIDIA RTX 4090 (24 GB VRAM)  
**Models root:** `/mnt/data/models/` (1.8 TB NVMe, ~663 GB used)

## Test Methodology

Each model is tested by:
1. Starting the inference container (cold boot) — load time recorded
2. Sending a generation request — inference time recorded
3. Verifying the output is valid (image saved, audio playable, etc.)
4. Stopping the container and evicting GPU memory between tests

All times are wall-clock measured from client-side. VRAM usage from `nvidia-smi`.

---

## Test Results

### 1. Qwen-Image-Edit 2511 FP8 (20B MMDiT)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Quantization** | FP8 weight-only (custom build) |
| **Load time (cold)** | 89 seconds |
| **Inference (512×512 edit)** | 47 seconds |
| **VRAM usage** | ~21 GB |
| **Output** | 512×512 PNG — valid |
| **Notes** | Cache-DiT + TaylorSeer acceleration enabled. Lightning distilled 4-step variant available. |
| **Status** | ✅ PASS |

### 2. Z-Image Turbo FP8 (8-step distilled)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Quantization** | FP8 weight-only (custom build) |
| **Load time (cold)** | 57 seconds |
| **Inference (1024×1024)** | 40 seconds |
| **VRAM usage** | ~20 GB |
| **Output** | 1024×1024 PNG — valid |
| **Notes** | 8-step distilled generation. Pipeline patch required for FP8. |
| **Status** | ✅ PASS |

### 3. Cosmos3-Nano (BF16, 27B)

| Metric | Value |
|--------|-------|
| **Engine** | SGLang |
| **Quantization** | BF16 (no FP8 available due to fused LLM params) |
| **Load time (cold)** | 244 seconds |
| **Inference** | OOM — CUDA out of memory |
| **VRAM** | 22.2 GB allocated before OOM |
| **Notes** | 27B BF16 transformer needs ~54GB. CPU offload not sufficient on 24GB card. Needs FP8 or model update. |
| **Status** | ❌ FAIL (OOM) |

### 4. MOSS SoundEffect V2 (TTS)

| Metric | Value |
|--------|-------|
| **Engine** | MOSS Docker (Tier A) |
| **Framework** | Diffusers + custom pipeline |
| **Load time (cold)** | 2 seconds |
| **Setup time** | +30 seconds (pip install diffusers inside container) |
| **Inference (3s audio)** | 55.6 seconds |
| **Output** | WAV — valid, realistic sound effects |
| **VRAM** | ~13 GB |
| **Notes** | First run required pip install diffusers. Inference is slow due to iterative generation. |
| **Status** | ✅ PASS |

### 5. CrispASR (VibeVoice ASR + Diarization)

| Metric | Value |
|--------|-------|
| **Engine** | CrispASR Docker (ghcr.io/crispstrobe/crispasr) |
| **Model** | VibeVoice-7B Q4_K |
| **Load time** | Pre-loaded (container already running) |
| **Inference (12s audio)** | 0.14 seconds |
| **VRAM** | ~1.5 GB |
| **Output** | JSON transcription — accurate |
| **Notes** | C++ backend, extremely fast. Always-on container. |
| **Status** | ✅ PASS |

### 6. LTX-Video 2.3 FP8 (22B Video)

| Metric | Value |
|--------|-------|
| **Engine** | SGLang (primary) / vLLM-Omni (fallback) |
| **Quantization** | ModelOpt FP8 |
| **Load time (cold)** | ~10 min (text encoder FSDP + transformer streaming) |
| **Inference** | OOM — CUDA out of memory during transformer loading |
| **VRAM at failure** | 23.42 GB / 23.52 GB (99.6%) |
| **Root cause** | 22B transformer + 12B Gemma text encoder exceed 24 GB even with layerwise offload. Transformer loading requires full model allocation before offloading can start, causing OOM with only 27 MiB free. |
| **vLLM-Omni fallback** | Fails — "Model class LTX2TwoStagePipeline not found in diffusion model registry" |
| **Notes** | LTX 2.3 needs >24 GB VRAM or more aggressive offloading support. SGLang's layerwise offload helps during inference but not during weight loading. ModelOpt FP8 format is 27 GB on disk (~12 GB VRAM equivalent) but the 12B text encoder FSDP shards add 3.27 GB minimum. Total minimum: ~16 GB + activation buffers >24 GB. |
| **Resolution** | Needs: (a) 32 GB+ GPU, (b) smaller text encoder variant, or (c) upstream SGLang fix for on-disk ModelOpt loading without full GPU allocation. |
| **Status** | ❌ FAIL (OOM) |

### 7. ACE-Step (Music Generation)

| Metric | Value |
|--------|-------|
| **Engine** | acestep.cpp GGML/CUDA (Tier A, Docker) |
| **GGUF files** | All 9 files downloaded (27 GB, `/mnt/data/models/audio/acestep-cpp/`) |
| **API** | Async — `POST /lm` → `GET /job?id=N` → `GET /job?id=N&result=1` → `POST /synth` |
| **Output** | MP3 (128 kbps, 48 kHz, Joint Stereo) or WAV |
| **VRAM** | ~4–8 GB (depends on model size) |

#### Performance by model combination

| LM | DiT | Steps | LM time | DiT time | Synth total | Audio length | Status |
|----|-----|-------|---------|----------|-------------|-------------|--------|
| 1.7B Q8 | v15-Turbo Q8 | 8 | ~4s | ~900ms | ~6s | ~81s | ✅ PASS |
| 1.7B Q8 | v15-SFT Q8 | 50 | ~4s | ~1900ms | ~4s | ~114s | ✅ PASS |
| 1.7B Q8 | v15-XL-Turbo Q8 | 8 | ~2s | ~400ms | ~2s | ~18s | ✅ PASS |
| 1.7B Q8 | v15-XL-SFT Q8 | 50 | ~2s | — | ~2s | ~18s | ✅ PASS |
| 1.7B Q8 | v15-XL-Base Q8 | 50 | ~6s | ~4000ms | ~6s | ~18s | ✅ PASS |
| 4B Q8 | v15-Turbo Q8 | 8 | ~12s | ~1778ms | ~4s | ~106s | ✅ PASS |

**Notes:**
- **1.7B Q8 LM**: ~4–5s for full CoT + code generation (256 codes). Fastest option for short prompts.
- **4B Q8 LM**: ~12s for same task. Richer musical understanding but 3× slower.
- **v15-Turbo (8-step)**: ~900ms DiT time for ~80s audio. ~8× realtime generation.
- **v15-SFT (50-step)**: ~1900ms DiT time for ~114s audio. Higher quality but only ~2× slower than turbo.
- **XL variants**: Larger 4B DiT models. XL-Base slower (~4s DiT) than XL-Turbo/Xl-SFT (~2s) due to 50-step.
- All models are ~Q8_0 quantization. VRAM usage stays ~6 GB for 1.7B LM + Turbo DiT, ~8 GB for XL variants.
- Async API: job IDs are 64-bit random hex. Completed jobs evicted FIFO after 32.
- `--keep-loaded` flag keeps LM in VRAM between requests (recommended for throughput).
- Server web UI available at `http://<host>:8056/` (embedded in ace-server binary).
- Docker image: `forge-reg.local:30500/tech-noir/ace-step:latest` (7.69 GB).
- All 9 GGUF files tested: 5 DiT models (SFT, Turbo, XL-SFT, XL-Turbo, XL-Base), 2 LM models (1.7B, 4B), VAE, text encoder.

| **Status** | ✅ PASS (all 6 model combinations) |

### 8. Z-Image Base FP8 (50-step, non-distilled)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Quantization** | FP8 weight-only (custom build) |
| **Load time (cold)** | 42 seconds |
| **Inference (1024×1024)** | 40 seconds |
| **VRAM usage** | ~20 GB |
| **Output** | 1024×1024 PNG (3.1 MB) — valid |
| **Notes** | 50-step base generation (non-distilled). Uses same pipeline as Turbo but with different weights. Config needed conversion from ModelOpt `modules_to_not_convert` → `ignored_layers` format for vLLM-Omni 0.22 compat. |
| **Status** | ✅ PASS |

### 9. ComfyUI (SDXL via node-based pipeline)

| Metric | Value |
|--------|-------|
| **Engine** | ComfyUI Docker (Tier A, gpu-all:latest) |
| **Checkpoint tested** | SDXL 1.0 base |
| **Load time (cold)** | 20 seconds |
| **Inference (1024×1024, 20 steps)** | 9.3 seconds |
| **VRAM usage** | ~8 GB |
| **Output** | 1024×1024 PNG (1.8 MB) — valid |
| **Notes** | ComfyUI supports 60+ checkpoints/loras/upscalers. Default output path is `/opt/ComfyUI/output/`. |
| **Status** | ✅ PASS |

### 10. Ideogram 4 (NF4)

| Metric | Value |
|--------|-------|
| **Engine** | SGLang |
| **Quantization** | NF4 (bitsandbytes) |
| **Source** | Gated — requires HF token with accepted terms |
| **Model on disk** | No |
| **VRAM** | ~11 GB (text_encoder 5.5GB + transformer 5.2GB) |
| **Status** | ⏳ PENDING (needs HF login + acceptance) |

### 11. Tangoflux (Text-to-Audio)

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D, gpu-all) |
| **Quantization** | FP32 |
| **Model on disk** | Yes (3.4 GB tangoflux.safetensors + 624 MB vae.safetensors) |
| **Source** | hf://declare-lab/TangoFlux |
| **VRAM** | ~6 GB |
| **Status** | ⏳ PENDING (needs `pip install datasets` + container setup) |

### 12. Wan VACE 14B FP8 (Video Editing)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Quantization** | FP8 weight-only |
| **Model on disk** | No — directory empty (`/mnt/data/models/video/wan2.1-vace-14b-fp8-diffusers/`) |
| **Source** | hf://Wan-AI/Wan2.1-VACE-14B |
| **VRAM** | ~16 GB estimated |
| **Resolution** | Needs manual FP8 direct-cast conversion before testing |
| **Status** | ⏳ PENDING (needs FP8 conversion + download) |

### 13. Wan T2V / I2V (Video Generation)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Model on disk** | No |
| **Source** | hf://Wan-AI/Wan2.1-T2V-14B / hf://Wan-AI/Wan2.1-I2V-14B |
| **Status** | ⏳ PENDING (needs download + launcher script) |

### 14. Qwen2.5-VL 7B (Vision-Language)

| Metric | Value |
|--------|-------|
| **Engine** | llama.cpp (Tier A) |
| **Format** | GGUF Q8_0 |
| **Model on disk** | Yes (7.6 GB, `/mnt/data/models/native/qwen2.5-vl-7b-gguf/`) |
| **VRAM** | ~8 GB |
| **Status** | ⏳ PENDING (needs llama.cpp container) |

### 15. Qwen-Edit (ModelOpt FP8, non-2511)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) |
| **Model on disk** | Yes — 28 GB transformer (`qwen-edit-modelopt-fp8-transformer`) + 5.5 GB NF4 text encoder (`qwen-edit-nf4-textenc`) |
| **VRAM** | ~16 GB estimated |
| **Notes** | This is the original Qwen-Edit (not 2511). ModelOpt FP8 quantization needs `modules_to_not_convert` conversion. |
| **Status** | ⏳ PENDING (needs launcher + FP8 config conversion) |

### 16. MOSS TTS / TTSD / Realtime

| Metric | Value |
|--------|-------|
| **Engine** | MOSS Docker (Tier A) |
| **Model on disk** | Yes — Audio Tokenizer (7.1 GB) + Audio Tokenizer Nano (0.3 GB) |
| **VRAM** | ~6 GB |
| **Notes** | MOSS Docker handles all TTS variants. Tokenizers loaded to CPU to conserve VRAM. |
| **Status** | ⏳ PENDING (MOSS container not currently running) |

### 17. Diarization-Turbo (VibeVoice-Realtime 0.5B)

| Metric | Value |
|--------|-------|
| **Engine** | CrispASR (Tier A) |
| **Model on disk** | Yes — `/mnt/data/models/vibevoice-cpp/vibevoice-realtime-0.5B-q8_0.gguf` |
| **VRAM** | ~1.5 GB |
| **Status** | ⏳ PENDING (needs separate container on port 8055) |

### 18. LLM Models (llama.cpp / BeeLlama)

| Metric | Value |
|--------|-------|
| **Engine** | llama.cpp / BeeLlama (Tier A) |
| **Models on disk** | Qwen3.6-27B Q5_K_S, Qwen3.6-35B-A3B UD-IQ4_NL, Gemma-4-26B, Gemma-4-31B, DFlash draft |
| **VRAM** | ~4–16 GB (varies by model) |
| **Status** | ⏳ PENDING (needs llama container on port 8052/8053) |

### 19. Tier D Diffusers Models (Kimodo, Kokoro, VibeVoice-TTS, See-Through)

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D, gpu-all) |
| **Models on disk** | Partial — Kokoro (in `/mnt/data/models/tts/kokoro/`), VibeVoice (in `/mnt/data/models/tts/vibevoice/`) |
| **Status** | ⏳ PENDING (needs diffusers container on port 8095) |

---

## Notes

- FP8 weight-only patch is required for Qwen, Z-Image, and VACE models on RTX 4090
  due to Triton fp8e4nv kernel crash. See `scripts/fp8_weight_only_patch.py`.
- Cosmos3-Nano cannot run on 24GB without FP8 — fused LLM params block quantization.
- LTX-2.3 symlinks were repaired after initial test; now all 20 resolve correctly.
- ACE-Step Dockerfile at `infra/docker/Dockerfile.acetep` — custom build with static ggml linking.
- 9/19 served model groups tested. Remaining 10 are blocked by: missing downloads (Wan VACE/T2V/I2V), gated access (Ideogram 4), container not running (MOSS, llama, diffusers), or pending setup (Tangoflux, Diarization-Turbo).
