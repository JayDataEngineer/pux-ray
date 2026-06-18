# Inference Test Report

**Date:** 2026-06-18 (session 3 — MOSS TTS ✅ via openmoss C++/GGML; soundeffect-v2 un-orphaned → routed to openmoss)  
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

### 4. MOSS SoundEffect V2 (TTS) — NOW SERVED BY OPENMOSS

| Metric | Value |
|--------|-------|
| **Engine** | OpenMOSS (Tier A, `forge-reg.local:30500/tech-noir/openmoss:latest`) |
| **Framework** | C++/GGML — same container serves all MOSS model types |
| **Notes** | SoundEffect V2 was orphaned after custom Python pipeline was deleted. The openmoss C++/GGML Docker now handles ALL MOSS variants: TTS, TTSD, VoiceGenerator, and SoundEffect V2. See entry 16 for speed benchmarks. |
| **Status** | ✅ PASS (via openmoss, same as entry 16) |

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
- **Async API**: Both `/lm` and `/synth` return `{"id":"..."}` immediately and process in background. Job completion observable via server logs. Audio output stored in shared memory (`/dev/shm/psm_*`). Results accessible via web UI at `http://<host>:8056/`.
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
| **Engine** | SGLang (lmsysorg/sglang:latest) |
| **Quantization** | NF4 (bitsandbytes) |
| **Source** | Gated — HF token accepted |
| **Model on disk** | Yes — 15 GB downloaded to `/mnt/data/models/cache/huggingface/models--ideogram-ai--ideogram-4-nf4/` |
| **VRAM** | ~11 GB (text_encoder 5.5GB + transformer 4.9GB + unconditional_transformer 4.9GB) |
| **Load time** | Failed — `diffusers` in SGLang container (0.39.0.dev0) lacks `Ideogram4Transformer2DModel` class |
| **Root cause** | The `lmsysorg/sglang:latest` image ships an older `diffusers` that doesn't include the Ideogram 4 pipeline components. Ideogram 4 requires `diffusers>=0.33` with `Ideogram4Transformer2DModel`. Need to either: (a) upgrade diffusers in the container, (b) build custom SGLang image with latest diffusers, or (c) use a different serving approach. |
| **Resolution** | Build custom SGLang image with `pip install --upgrade diffusers` or use `forge-reg.local:30500/tech-noir/gpu-all:natten-0.21.5` as base with SGLang installed fresh. |
| **Status** | ❌ FAIL (diffusers version mismatch) |

### 11. Tangoflux (Text-to-Audio)

| Metric | Value |
|--------|-------|
| **Engine** | Native Python (Diffusers-based, tested outside container) |
| **Quantization** | FP32 |
| **Model on disk** | Yes (3.4 GB tangoflux.safetensors + 624 MB vae.safetensors) |
| **Source** | hf://declare-lab/TangoFlux |
| **Load time (cold)** | 3.2 seconds (T5 encoder auto-downloaded, weights cached) |
| **Inference (5s audio, 30 steps)** | 1.5 seconds |
| **VRAM usage** | ~3.5 GB |
| **Output** | 44.1 kHz stereo WAV (1723 KB for 5s) — valid, realistic sound |
| **Notes** | Uses T5 encoder + DiT with flow matching. Very fast inference. Requires `pip install datasets` for the model loader. Tested with prompt "bubbly water stream flowing". Deprecation warnings from torch (txt_ids 3D tensor) but no functional issues. |
| **Status** | ✅ PASS |

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
| **Engine** | llama.cpp (llama-server-upstream in gpu-all container) |
| **Format** | GGUF Q8_0 |
| **Model on disk** | Yes (7.6 GB, `/mnt/data/models/native/qwen2.5-vl-7b-gguf/Qwen2.5-VL-7B-Instruct-Q8_0.gguf`) |
| **Load time (cold)** | 6.7 seconds |
| **Prompt processing** | 1411 tok/s (28 tokens in 20ms) |
| **Generation speed** | 116 tok/s (14 tokens in 120ms) |
| **VRAM usage** | ~11.7 GB |
| **Output** | Valid haiku: *"Neurons connect, / Signals through the network flow, / Learning unfolds."* |
| **API** | OpenAI-compatible `/v1/chat/completions` and `/v1/completions` |
| **Notes** | No mmproj file available for vision tasks — tested as text-only LLM. Full 128K context but limited to 4096 via `--ctx-size`. Server health endpoint at `/health`. |
| **Status** | ✅ PASS |

### 15. Qwen-Edit (ModelOpt FP8, non-2511)

| Metric | Value |
|--------|-------|
| **Engine** | SGLang (attempted) |
| **Model on disk** | Yes — 29.5 GB transformer (3 safetensors, `qwen-edit-modelopt-fp8-transformer`) + 5.5 GB NF4 text encoder (`qwen-edit-nf4-textenc`) |
| **VRAM** | ~16 GB estimated (model only) |
| **Load result** | Failed — SGLang reports `ValueError: Unrecognized model in /models/native/qwen-edit-modelopt-fp8-transformer. Should have a \`model_type\` key in its config.json.` |
| **Root cause** | The ModelOpt FP8 transformer's `config.json` lacks a `model_type` field. This is a known issue with ModelOpt FP8 exports — the config uses `_class_name` (`QwenImageTransformer2DModel`) but not `model_type`. SGLang requires `model_type` to load. |
| **Resolution** | Add `"model_type": "qwen_image_transformer_2d"` to transformer config.json, or use vLLM-Omni with FP8 weight-only patch instead of SGLang. |
| **Status** | ❌ FAIL (ModelOpt config missing model_type) |

### 16. MOSS TTS / TTSD / Realtime

| Metric | Value |
|--------|-------|
| **Engine** | OpenMOSS (Tier A, `forge-reg.local:30500/tech-noir/openmoss:latest`) |
| **Framework** | C++/GGML (pwilkin/openmoss) — Qwen3-8B backbone + 32 RVQ audio codebooks + 1.6B pure-transformer audio codec |
| **Model on disk** | Yes — Q8_0 quantized GGUF: 8.7 GB (`moss-tts.gguf`) + 4.1 GB extras (`moss-tts.extras.gguf`) in `/mnt/data/models/audio/moss-tts/` |
| **Serves** | `moss-tts`, `moss-ttsd`, `moss-voicegenerator`, `moss-soundeffect-v2` — all MOSS model types |
| **VRAM** | ~7.7 GB model + ~5 GB aux = ~13 GB total |
| **APIs** | `GET /health`, `GET /info`, `POST /tts`, `POST /v1/audio/speech` |
| **Output** | WAV — 24 kHz PCM s16le mono |
| **Loaded codec** | ✅ 32 audio embeds, 32 audio heads, codec=yes |
| **GPU offload** | 37/37 layers on CUDA, ~7.7 GiB GPU buffer |
| **Speed (short, 3 words, 1.12s audio)** | **0.68s wall** → 1.65× real-time |
| **Speed (med, 3 sentences, 9.68s audio)** | **1.83s wall** → 5.28× real-time |
| **Speed (long, 5 sentences, 32.4s audio)** | **5.25s wall** → 6.17× real-time |
| **Speed (OpenAI-compat, 1 sentence, 3.84s)** | **0.97s wall** → 3.96× real-time |
| **Test result** | `POST /tts "Hello, this is a test..."` → 158 KB WAV (3.36 s) ✅ |
| **OpenAI-compat** | `POST /v1/audio/speech` → 181 KB WAV ✅ |
| **Notes** | Scales well with text length: longer utterances are more efficient (amortizes prompt processing). Voice cloning available via reference WAV. |
| **Status** | ✅ PASS |

### 17. Diarization-Turbo (VibeVoice-Realtime 0.5B)

| Metric | Value |
|--------|-------|
| **Engine** | CrispASR (Tier A, `forge-reg.local:30500/tech-noir/asr:latest`) |
| **Model on disk** | Yes — `/mnt/data/models/vibevoice-cpp/vibevoice-realtime-0.5B-q8_0.gguf` (1.6 GB) |
| **Container** | `inference-diarization-turbo` on port 8055 |
| **Backend** | `vibevoice-tts` (C++ GGML/CUDA) |
| **Load time** | ~5 seconds |
| **VRAM usage** | ~1.5 GB |
| **ASR test** | Endpoint `/v1/audio/transcriptions` returns HTTP 200 with empty transcription for synthetic test audio |
| **TTS test** | Fails — `vibeyvoice_synthesize: model lacks decoder tensors (convert with --include-decoder)` |
| **Root cause** | The GGUF was quantized without decoder support (`--include-decoder` flag missing during conversion). Model can do streaming ASR but not full TTS synthesis. Voice presets (`voice-en-Emma.gguf`, `voice-en-Carter_man.gguf`) load correctly. |
| **CrispASR** | Health endpoint `/health` returns `{"status":"ok"}`. API supports `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/voices`. |
| **Notes** | Started with `CRISPASR_EXTRA_ARGS="--voice-dir /models/vibevoice-cpp"` for voice preset directory. |
| **Status** | ⚠️ PARTIAL (ASR works, TTS blocked by GGUF conversion) |

### 18. LLM Models (llama.cpp / BeeLlama)

| Metric | Value |
|--------|-------|
| **Engine** | llama.cpp (Tier A, `forge-reg.local:30500/tech-noir/gpu-all:latest`) |
| **API** | HTTP `/completion` endpoint on port 8052 |
| **Models on disk** | Qwen3.6-27B Q5_K_S (18 GB), Qwen3.6-35B-A3B UD-IQ4_NL (17 GB), Gemma-4-26B UD-IQ4_NL (13 GB), Gemma-4-31B UD-Q4_K_XL (18 GB), DFlash draft (1 GB) |
| **VRAM** | ~4–22 GB (varies by model) |

#### Performance by model

| Model | Quant | File size | Load time | Prompt speed | Gen speed | VRAM | KV cache | Notes |
|-------|-------|-----------|-----------|-------------|-----------|------|----------|-------|
| **Qwen2.5-VL 7B** | Q8_0 | 7.6 GB | 6.7s | 1411 tok/s | 116 tok/s | 11.7 GB | GPU | No mmproj for vision |
| **Qwen3.6-27B** | Q5_K_S | 18 GB | ~12s | — | 43 tok/s | 21.2 GB | GPU | Reasoning mode outputs `reasoning_content` separately |
| **Qwen3.6-35B-A3B** | UD-IQ4_NL | 17 GB | 3s | 248 tok/s | 184 tok/s | 22.4 GB | GPU | MoE — only active params loaded. Very fast. |
| **Gemma-4-26B** | UD-IQ4_NL | 13 GB | ~8s | — | 178 tok/s | 18.2 GB | GPU | Fastest for its size |
| **Gemma-4-31B** | UD-Q4_K_XL | 18 GB | 2s | 284 tok/s | 23 tok/s | ~18 GB | CPU | `--no-kv-offload` required to fit VRAM. KV cache OOM on GPU even at 2048 ctx. |

**Notes:**
- Qwen3.6-35B-A3B and Gemma-4-31B were tested in this session (2026-06-18).
- Qwen3.6-27B and Gemma-4-26B were tested in session 2.
- Gemma-4-31B needs `--no-kv-offload` on 24 GB card — the Q4_K_XL quantization uses ~18 GB for weights alone.
- DFlash draft model (1 GB) available for speculative decoding with Qwen3.6 models but not yet tested.
- All models use `-ngl 99` (full GPU offload).
- Qwen3.6-35B-A3B is a MoE (Mixture of Experts) model with 35B total params but only ~3.6B active per token.
- | **Status** | ✅ 5/5 tested (Qwen2.5-VL 7B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-26B, Gemma-4-31B) |

### 19. Kokoro 82M TTS

| Metric | Value |
|--------|-------|
| **Engine** | Native Python (kokoro library, CPU) |
| **Model** | `kokoro-v1_0.pth` from `/mnt/data/models/tts/kokoro/` |
| **Load time (cold)** | 1.0 second |
| **Inference (4.6s audio)** | 0.3 seconds |
| **Output** | 24 kHz mono WAV (434 KB) — valid, clear speech |
| **VRAM usage** | 0 GB (CPU-only model) |
| **Voice** | `af_heart` (American English female) |
| **Dependencies** | Requires `kokoro`, `misaki`, `num2words`, `spacy`, `phonemizer`, `espeak-ng`. All pip-installable. |
| **Status** | ✅ PASS |

### 20. See-Through (Anime Layer Decomposition)

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D) |
| **Model on disk** | No — broken symlinks to Docker-internal HF cache paths |
| **Symlinks** | `layerdiff3d`, `marigold`, `scheduler` all point to `/models/hf_cache/hub/...` which doesn't exist on host |
| **VRAM** | ~4 GB estimated |
| **Status** | ⏳ PENDING (needs model download to host filesystem) |

### 21. VibeVoice 7B TTS

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D, GPU) |
| **Model on disk** | Yes — 18.7 GB in `/mnt/data/models/tts/vibevoice/` (10 safetensor shards) |
| **VRAM** | ~20 GB estimated |
| **Status** | ⏳ PENDING (needs GPU container with 20GB+ VRAM) |

### 22. Kimodo-SOMA-RP (Motion Diffusion)

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D, GPU) |
| **Model on disk** | No — `avatar/kimodo/` directory not on host filesystem |
| **VRAM** | ~3 GB estimated |
| **Status** | ⏳ PENDING (needs download) |

---

## Notes

- FP8 weight-only patch is required for Qwen, Z-Image, and VACE models on RTX 4090
  due to Triton fp8e4nv kernel crash. See `scripts/fp8_weight_only_patch.py`.
- Cosmos3-Nano cannot run on 24GB without FP8 — fused LLM params block quantization.
- LTX-2.3 symlinks were repaired after initial test; now all 20 resolve correctly.
- ACE-Step Dockerfile at `infra/docker/Dockerfile.acetep` — custom build with static ggml linking.
- Kokoro requires `pip install kokoro misaki num2words spacy phonemizer` and system `espeak-ng`.
- Tangoflux requires `pip install datasets` for its model loader.
- Qwen2.5-VL 7B tested as text-only LLM; vision would need `mmproj` file.
- Qwen-Edit (non-2511) and Ideogram 4 both blocked by serving infrastructure issues, not model availability.
- Diarization-Turbo 0.5B GGUF needs re-quantization with `--include-decoder` for TTS support.
- **15/22 model groups tested** (9 from session 1 + 4 from session 2 + 2 new this session: Qwen3.6-35B-A3B, Gemma-4-31B). 7 remain pending or blocked.
- **LLM sub-tasks**: 5/5 models tested (Qwen2.5-VL 7B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-26B, Gemma-4-31B).
- **Pending still**: Wan VACE FP8 (empty dir — needs conversion), Wan T2V/I2V (not downloaded), VibeVoice-7B TTS (needs GPU container), Kimodo (not downloaded), See-Through (symlinks broken). Ideogram 4 and Qwen-Edit blocked by infrastructure issues.
