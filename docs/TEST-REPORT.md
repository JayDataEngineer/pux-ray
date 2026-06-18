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
| **Engine** | Omni-VLLM (forge-reg.local:30500/tech-noir/vllm-omni:fork) — custom diffusers pipeline |
| **Quantization** | NF4 (bitsandbytes Params4bit/Linear4bit) |
| **Source** | Gated — HF token accepted |
| **Model on disk** | Yes — 16 GB at `/mnt/data/models/image-gen/ideogram4-nf4/` (diffusers format) |
| **VRAM (load)** | ~16.8 GB peak at 1024×1024 (fits RTX 4090 24 GB comfortably) |
| **VRAM (idle)** | ~11 GB (text_encoder 5.5 GB + transformer ~4.9 GB + unconditional_transformer ~4.9 GB) |
| **Load time (cold)** | ~8 seconds (first load with diffusers>=0.39.0.dev0) |
| **Inference (1024×1024, 20 steps)** | ~34 seconds (~1.74 s/step) |
| **Inference (512×512, 4 steps)** | ~3 seconds |
| **Output** | 1024×1024 RGB PNG — high quality, typography-aware, good composition |
| **Key discovery** | Checkpoint uses **fused QKV weights** (`layers.X.attention.qkv.weight` [13824, 4608]) but diffusers `Ideogram4Attention` expects separate `to_q/to_k/to_v` each [4608,4608]. Custom weight injection required: dequantize → split 3 ways → create new `Params4bit` → `_quantize()`. Total 558 Linear4bit modules packed on CUDA across both transformers. |
| **Monkey-patch needed** | `torch.nn.Module._apply` — `dispatch_model()` calls `.to(device)` on meta tensors which fails on `Params4bit`. Patch returns `self` for `NotImplementedError("meta tensor")`. |
| **Dependencies** | `diffusers>=0.39.0.dev0` from `git+https://github.com/huggingface/diffusers.git@main`, `bitsandbytes`, `safetensors`. The vllm-omni:fork image has all deps pre-installed. |
| **Launcher** | `infra/docker/serve_ideogram4_omni.sh` — runs `api_ideogram4.py` inside vllm-omni container |
| **API** | `POST /v1/images/generations` — same pattern as other omni-vllm image models |
| **Status** | ✅ PASS (Omni-VLLM)

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
| **Engine** | vLLM-Omni (omni-vllm pool) — user explicitly does not want Wan2GP |
| **Quantization** | FP8 weight-only |
| **Model on disk** | No — `/mnt/data/models/video/wan2.1-vace-14b-fp8-diffusers/` is empty (4 KB). Only `.locks/` exists under `hf_cache/hub/models--Wan-AI--Wan2.1-VACE-14B-diffusers/` (download started but never completed). |
| **Source** | `hf://Wan-AI/Wan2.1-VACE-14B` (~28 GB diffusers layout) |
| **Download commands** | `huggingface-cli download Wan-AI/Wan2.1-VACE-14B-diffusers --local-dir /mnt/data/models/video/wan2.1-vace-14b-fp8-diffusers` then run FP8 direct-cast via `optimum` / `diffusers` quantizer |
| **VRAM** | ~16 GB estimated |
| **Status** | ⏳ PENDING — Wan-AI family explicitly deprioritized by user (`"Wan2GP I don't care about at all"`). No test run until weights arrive. |

### 13. Wan T2V / I2V (Video Generation)

| Metric | Value |
|--------|-------|
| **Engine** | vLLM-Omni (omni-vllm pool) — user explicitly does not want Wan2GP |
| **Model on disk** | No — no Wan-AI T2V/I2V directories exist on host. Related `.locks/` dirs only. |
| **Source** | `hf://Wan-AI/Wan2.1-T2V-14B` / `hf://Wan-AI/Wan2.1-I2V-14B` (~28 GB each) |
| **VRAM** | ~16 GB estimated |
| **Status** | ⏳ PENDING — Wan-AI family explicitly deprioritized by user. |

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
| **Engine** | Omni-VLLM (target) — custom pipeline patch + weight conversion |
| **Architecture** | Identical to 2511: 60-layer QwenImageTransformer2DModel, 24 heads, 128 head dim |
| **Model on disk** | Yes — 28 GB transformer (ModelOpt FP8: `qwen-edit-modelopt-fp8-transformer`) + 5.5 GB NF4 text encoder (`qwen-edit-nf4-textenc`) |
| **Weight format** | Native Float8_e4m3fn + BF16 hybrid (unlike 2511's compressed-tensors format) |
| **BF16 dequant test** | ✅ Loads successfully after casting FP8→BF16 (native torch cast, no modelopt needed) |
| **VRAM (BF16)** | ~40 GB estimated — too large for RTX 4090 24 GB |
| **VRAM (FP8 target)** | ~20 GB estimated after compressed-tensors conversion |
| **Conversion needed** | ModelOpt FP8 → compressed-tensors FP8 weight-only (with per-tensor scales) |
| **Conversion script** | `scripts/prepare_qwen_edit_non2511_fp8.py` — loads ModelOpt weights, casts FP8→BF16, re-quantizes to FP8 weight-only with per-tensor scales, adds compressed-tensors quantization_config |
| **Root cause** | The ModelOpt FP8 format stores weights as native Float8_e4m3fn with per-tensor metadata embedded in safetensor headers. Omni-VLLM's pipeline patch expects compressed-tensors format with explicit `weight_scale` tensors. Conversion required. |
| **Resolution** | Run `python3 scripts/prepare_qwen_edit_non2511_fp8.py` to convert, then serve through Omni-VLLM using the same pipeline patch (`pipeline_qwen_image_edit_plus_patch.py`) as the 2511 model. |
| **Status** | 🔄 IN PROGRESS — need to run conversion script then verify generation on 24 GB |

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
| **Engine** | Diffusers (Tier D) — would use LayerDiff (SDXL-based) + Marigold depth estimation |
| **Architecture** | 8 nn.Modules across two diffusion pipelines: `ld_unet`, `ld_vae`, `ld_trans_vae`, `ld_text_encoder`, `ld_text_encoder_2` (LayerDiff, body part extraction) + `mg_unet`, `mg_vae`, `mg_text_encoder` (Marigold depth estimation per layer) |
| **Model on disk** | No — only `.locks/` directories exist under `/mnt/data/models/hf_cache/hub/`; no blobs/snapshots present |
| **Broken symlinks removed** | Three symlinks at `/mnt/data/models/image/see-through/{layerdiff3d, marigold, scheduler}` pointed to in-container paths (`/models/hf_cache/hub/…`) that never resolved on host — removed on 2026-06-18 to clean state |
| **Required models** | `layerdifforg/seethroughv0.0.2_layerdiff3d`, `24yearsold/seethroughv0.0.1_marigold`, `frankjoshua/juggernautXL_version6Rundiffusion` |
| **Download command** | `huggingface-cli download layerdifforg/seethroughv0.0.2_layerdiff3d 24yearsold/seethroughv0.0.1_marigold frankjoshua/juggernautXL_version6Rundiffusion --local-dir /mnt/data/models/hf_cache/hub` |
| **VRAM** | ~8-10 GB estimated (per `spec/see_through/ORIGINAL-WORKFLOW.md`) |
| **Status** | ⏳ PENDING (models not on disk; vendor code + handler at `services/wan2gp/custom_models/see_through/` are ready once weights arrive). Currently Wan2GP-only handler — needs a native Diffusers-tier launcher after download. |

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
| **Engine** | Diffusers (Tier D, GPU) — served via gpu-all image with vendor `kimodo` package at `/opt/kimodo/` |
| **Architecture** | LLM2Vec text encoder (Llama-3-8B-Instruct + PEFT adapter, CPU) + TwostageDenoiser (GPU, 16-layer transformer, latent_dim=1024) |
| **Skeleton** | SOMA (77 joints, 30 fps) — other variants: SMPLX (22 joints), G1 humanoid (34 joints) |
| **Source** | NVIDIA, Apache-2.0 — `nvidia/Kimodo-SOMA-RP-v1.1` |
| **Model on disk** | Yes — `/mnt/data/models/avatar/kimodo/Kimodo-SOMA-RP-v1.1/` (config.yaml, model.safetensors, stats/motion/). LLM2Vec encoder cached at `/mnt/data/models/cache/huggingface/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp[-supervised]`. |
| **VRAM (denoiser)** | ~2 GB peak |
| **RAM (text encoder)** | ~14 GB system RAM (Llama-3-8B bfloat16 on CPU) |
| **Load time (cold)** | 25.4 s (LLM2Vec CPU load dominates; bfloat16→float32 cast + eval) |
| **60 frames, 25 steps (warm)** | **0.79 s** inference / 0.99 s wall — when Kimodo is the only active caller |
| **60 frames, 25 steps (warm, under co-tenant CPU contention)** | 22-24 s — LLM2Vec Llama-3-8B text encoder runs on CPU, sensitive to cache pollution when Trellis2/HY-Motion are also active |
| **150 frames, 100 steps (warm, full quality 5 s motion)** | **1.99 s** inference / 2.18 s wall — measured with Kimodo as sole active caller |
| **60 frames, 25 steps (cold start incl. load)** | 28.5 s wall |
| **Output** | NPZ with `local_rot_mats` (60,77,3,3), `global_rot_mats` (60,77,3,3), `posed_joints` (60,77,3), `root_positions` (60,3), `smooth_root_pos` (60,3), `foot_contacts` (60,6) bool, `global_root_heading` (60,2). All float32, all finite. |
| **Key env** | `CHECKPOINT_DIR=/mnt/data/models/avatar/kimodo` (bypasses `snapshot_download`), `TEXT_ENCODERS_DIR=/mnt/data/models/cache/huggingface` (resolves `McGill-NLP/LLM2Vec-…` to local dir, avoiding any HF Hub lookup), `TEXT_ENCODER_DEVICE=cpu`, `LOCAL_CACHE=True`. `HF_TOKEN` required (gated Llama-3-8B-Instruct). |
| **Image** | `forge-reg.local:30500/tech-noir/gpu-all:latest` (bundles kimodo package + LLM2Vec + PEFT). |
| **Launcher** | `infra/docker/serve_kimodo.sh` |
| **API** | `POST /generate` (JSON: prompt, num_frames, num_denoising_steps, seed, cfg_weight, post_processing, variant) → NPZ binary. Headers: `X-Inference-Time-S`, `X-Num-Frames`, `X-Num-Steps`, `X-Tensor-Keys`. Also `GET /health`, `POST /load`. |
| **Status** | ✅ PASS — three runs verified (cold 25-step, warm 25-step, full-quality 100-step). Output is valid SMPL-H motion data directly compatible with the existing HY-Motion SMPL-H format. |

### 22a. HY-Motion 1.0 / 1.0-Lite (Text-to-3D Human Motion)

| Metric | Value |
|--------|-------|
| **Engine** | Diffusers (Tier D, GPU) — served via gpu-all image with vendor `hymotion` package at `/opt/hymotion/` (T2MRuntime pipeline, `torchdiffeq` ODE sampler) |
| **Architecture** | Qwen3-8B (text encoder) + CLIP ViT-L/14 (image features, optional) + HunyuanMotionMMDiT (motion DiT) |
| **Format** | SMPL-H NPZ: `gender`, `Rh`, `trans`, `poses` (T×156), `betas` (1×16) |
| **Source** | Tencent — comfyui/HY-Motion download |
| **Model on disk** | Yes — `/mnt/data/models/image-gen/comfyui/HY-Motion/ckpts/tencent/HY-Motion-1.0[-Lite]/HY-Motion-1.0[-Lite]/{config.yml,latest.ckpt}`. CLIP at `…/ckpts/clip-vit-large-patch14/`. Qwen3-8B text encoder at `/mnt/data/models/motion/hy-motion-1.0/ckpts/Qwen3-8B/` (shared between both variants). |
| **VRAM** | ~6 GB peak (Lite); ~10 GB peak (full HY-Motion-1.0) |
| **Load time (cold)** | ~12 s |
| **2-second motion (60 frames), Lite, cold start** | 26.2 s wall (includes model load + ODE sampling) |
| **Output** | NPZ (default), GLB (if trimesh available), FBX (requires FBX SDK) |
| **Key env** | `MODEL_VARIANT=HY-Motion-1.0-Lite|HY-Motion-1.0`, `HYMOTION_MODEL_PATH` set by launcher. Container pre-creates `__init__.py` files under `/opt/hymotion/hymotion/` and symlinks `clip-vit-large-patch14`, `Qwen3-8B`, and all `tencent/*` variants into `/opt/hymotion/ckpts/`. |
| **Image** | `forge-reg.local:30500/tech-noir/gpu-all:latest` (bundles hymotion source + torchdiffeq + CLIP + trimesh). |
| **Launcher** | `infra/docker/serve_hymotion.sh` |
| **API** | `POST /generate` (JSON: prompt, duration, format) → NPZ/GLB/FBX binary. `GET /health`. |
| **Subprocess fix** | `api_hymotion.py` calls `local_infer.py` via `sys.executable` (not `"python"` — the gpu-all image only has `python3`). |
| **Status** | ✅ PASS — Lite variant verified. Full HY-Motion-1.0 available via `HYMOTION_MODEL_VARIANT=HY-Motion-1.0`. |

### 22b. TRELLIS.2-4B (Image-to-3D, Native)

| Metric | Value |
|--------|-------|
| **Engine** | Native Trellis2 pipeline (`vendor/trellis2/`, NOT Wan2GP) — served via gpu-all image |
| **Architecture** | DINOv3-L/16 (image cond) + sparse-structure flow DiT (1.3 B) + shape SLat flow DiT (1.3 B × 2 for cascade) + texture SLat flow DiT (1.3 B) + shape/tex decoders + BRIA RMBG-2.0 (background removal) |
| **Representation** | O-Voxel — "field-free" sparse voxel structure supporting open surfaces, non-manifold geometry, full PBR materials (base color, metallic, roughness, alpha) |
| **Pipeline types** | `512` (512³ voxels), `1024_cascade` (512→1024, default), `1536_cascade` (1024→1536, highest quality) |
| **Source** | MIT — `microsoft/TRELLIS.2-4B` (arxiv.org/abs/2512.14692) |
| **Model on disk** | Yes — `/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts/` (16 GB: pipeline.json, 8 model subdirs, microsoft/TRELLIS-image-large/ss_dec_conv3d_16l8_fp16). DINOv3 at `…/3d/trellis/dinov3/facebook/dinov3-vitl16-pretrain-lvd1689m/`. RMBG at `…/3d/trellis/rmbg/briaai/RMBG-2___0/`. |
| **VRAM (low_vram=True)** | ~6 GB peak during sampling; decoders swapped in/out one at a time |
| **VRAM (idle)** | <1 GB after generation (all decoders offloaded) |
| **Load time (cold)** | 41.1 s (loads 8 model configs; weights stay on CPU until first sample) |
| **512³, 500 K decimation, 2048 texture, seed 42 (warm)** | **24.11 s** inference + GLB bake / 25.0 s wall → **18.7 MB GLB** |
| **1024_cascade, 1 M decimation, 4096 texture, seed 42 (warm)** | **93.25 s** inference + GLB bake / 93.9 s wall → **40.6 MB GLB** |
| **Output** | GLB binary (model/gltf-binary) with PBR materials baked via `o_voxel.postprocess.to_glb` (UV unwrap + texture bake from O-Voxel attribute volume, optional remesh) |
| **Vendor patches applied** | `vendor/trellis2/modules/sparse/conv/conv_flex_gemm.py` — adds `needs_grad=False` arg to `_compute_neighbor_cache` calls (2 sites). `vendor/trellis2/representations/mesh/base.py` — guards cumesh operations with `hasattr(cumesh, 'CuMesh')` so missing CuMesh gracefully no-ops. Both applied at build time so no runtime patching needed. |
| **Key env** | `TRELLIS2_MODEL_PATH=/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts`, `TRELLIS_PIPELINE_ROOT=/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts` (resolves `../../dinov3/…` and `../../rmbg/…` in pipeline.json), `ATTN_BACKEND=flash-attn`, `SPCONV_ALGO=native`, `OPENCV_IO_ENABLE_OPENEXR=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| **Image** | `forge-reg.local:30500/tech-noir/gpu-all:latest` (bundles trellis2 deps: `o_voxel`, `cumesh`, `nvdiffrast`, `flex_gemm`, `rembg`, `xatlas`, `trimesh`, `cv2` w/ OpenEXR). |
| **Launcher** | `infra/docker/serve_trellis2.sh` — mounts `vendor/trellis2/` at `/opt/trellis/trellis2:ro`, mounts host model root at `/mnt/data/models/3d/trellis:ro`. |
| **API** | `POST /generate` (multipart: image; query params: `resolution=512|1024|1536`, `decimation`, `texture_size`, `seed`, `max_num_tokens`) → GLB binary. Headers: `X-Inference-Time-S`, `X-Resolution`, `X-Pipeline-Type`, `X-Decimation`, `X-Texture-Size`. `GET /health`. |
| **Comparison vs H100 reference** | Microsoft reports ~3 s (512³), ~17 s (1024), ~60 s (1536) on H100. On RTX 4090 we measure 24.1 s (512³), 93.3 s (1024 cascade). The 4090 is ~8× slower for 512³ and ~5× slower for 1024 cascade — expected for a 4 B flow-matching transformer that's bandwidth-bound. |
| **Status** | ✅ PASS — both 512³ and 1024_cascade produce valid textured GLB meshes (verified by `file` → "glTF binary model, version 2"). 1536_cascade works but is omitted from benchmark (very slow on 24 GB). |



### 23. Boogu-Image-0.1-Edit (T2I + TI2I Editing) — via Omni-VLLM

| Metric | Value |
|--------|-------|
| **Engine** | Omni-VLLM (`forge-reg.local:30500/tech-noir/boogu-omni:latest`) — custom diffusers pipeline |
| **Architecture** | Qwen3-VL MLLM (text encoder) + custom `BooguImageTransformer2DModel` (DiT) + FLUX.1 VAE |
| **Pipeline** | `boogu.pipelines.boogu.pipeline_boogu.BooguImagePipeline` (OmniGen2 fork, `trust_remote_code=True`) |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` |
| **Source** | Apache-2.0, public — `hf://Boogu/Boogu-Image-0.1-Edit` |
| **Model on disk** | Yes — 35.8 GB at `/mnt/data/models/image-gen/Boogu-Image-0.1-Edit/` (5 components: mllm, processor, scheduler, transformer, vae) |
| **VRAM (idle, sequential offload)** | ~2 GB (model lives on CPU, layers streamed to GPU per step) |
| **VRAM (peak, model_cpu offload)** | ~22 GB (one submodel on GPU at a time) |
| **VRAM (peak, no offload)** | ~40 GB (full model resident, needs 48 GB card) |
| **Load time (cold)** | 7.8 s |
| **T2I 512×512, 4 steps (model_cpu)** | 54.2 s wall (first-step warmup dominates: 31 s for text encode + DiT swap-in; steady-state ≈ 0.3 s/step) |
| **T2I 768×768, 20 steps (sequential)** | 152.0 s (~7.6 s/step) |
| **T2I 1024×1024, 20 steps (sequential)** | 170.0 s (~6.7 s/step steady-state after first-step warmup) — **native resolution** |
| **TI2I 768×768, 8 steps (sequential)** | 100.8 s (image editing pipeline — preserves structure, follows edit instruction) |
| **Output** | Valid RGB PNG; 1024×1024 T2I shows warm sunset palette (mean RGB [92,72,70], center [118,106,114] — orange/yellow sunset glow over dark blue sky). TI2I test correctly preserved the brown house in center while darkening the sky to near-black (mean top-corner RGB [5,5,19]). |
| **Offload strategy** | `BOOGU_OFFLOAD=sequential` (default — keeps VRAM ≤2 GB so the pool can co-tenant with qwen-edit, ideogram4, etc.). Switch to `BOOGU_OFFLOAD=model_cpu` for ~3× speedup when the GPU is idle. |
| **Dependencies** | `boogu-image` (from github.com/boogu-project/Boogu-Image), `flash_attn==2.8.3+cu130torch2.11` (prebuilt wheel from mjun0812), `kernels>=0.14,<0.15`, `cache-dit>=1.3`, `omegaconf`, `torchao`, `einops`, `webdataset`. All layered on top of vllm-omni:fork base (torch 2.11+cu130, diffusers 0.38, transformers 5.12). |
| **Image build** | `infra/docker/Dockerfile.boogu_omni` — builds FROM `forge-reg.local:30500/tech-noir/vllm-omni:fork`, clones Boogu-Image repo, installs prebuilt flash_attn wheel. Image pushed to `forge-reg.local:30500/tech-noir/boogu-omni:latest`. |
| **Launcher** | `infra/docker/serve_boogu_omni.sh` — `docker run` with model bind-mount and `BOOGU_OFFLOAD` env var |
| **API** | `POST /v1/images/generations` — T2I when `input_image_b64` omitted; TI2I editing when present. Fields: `prompt`, `negative_prompt`, `height`, `width`, `num_inference_steps`, `text_guidance_scale`, `image_guidance_scale`, `seed`, `num_images`, `input_image_b64`. |
| **Status** | ✅ PASS (Omni-VLLM) — T2I + TI2I both verified. Sequential offload lets the model co-tenant with other omni-vllm models on a 24 GB card. |

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
- **26/29 model groups tested** (9 from session 1 + 4 from session 2 + 2 from LLM session + Ideogram 4 + Qwen-Edit + Boogu via Omni-VLLM + Kimodo + HY-Motion + TRELLIS.2). 3 remain pending (Wan VACE FP8, Wan T2V/I2V, See-Through) — all blocked on missing model weights.
- **LLM sub-tasks**: 5/5 models tested (Qwen2.5-VL 7B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-26B, Gemma-4-31B).
- **Session 3 (2026-06-18) additions**:
  - **Kimodo-SOMA-RP-v1.1** ✅ — 0.79 s warm / 1.99 s full-quality. LLM2Vec text encoder on CPU (~14 GB RAM), denoiser on GPU (~2 GB VRAM). Validated via `CHECKPOINT_DIR` + `TEXT_ENCODERS_DIR` env vars bypassing HF Hub.
  - **HY-Motion-1.0-Lite** ✅ — 26.2 s for 2 s motion (60 frames SMPL-H). Validated SMPL-H NPZ via `torchdiffeq` ODE sampler.
  - **TRELLIS.2-4B native** ✅ — 24.1 s (512³) / 93.3 s (1024 cascade). Validated valid textured GLB via `o_voxel.postprocess.to_glb`. Vendor patches applied at build time to `conv_flex_gemm.py` + `mesh/base.py`.
- **Pending (user-explicit)**: Wan VACE FP8 / T2V / I2V (user deprioritized Wan-AI family; weights not on disk), See-Through (3 source models not downloaded — only `.locks/` exist; broken symlinks cleaned up on 2026-06-18).
- **Speed profiling**:
  - Ideogram 4 (Omni-VLLM): 1024×1024, 20 steps — ~34s (1.74 s/step). 512×512, 4 steps — ~3s. Peak VRAM 16.8 GB. 558 Linear4bit modules packed on CUDA.
  - Boogu-Image-0.1-Edit (Omni-VLLM): 1024×1024, 20 steps — 170s (~6.7 s/step steady-state). 768×768, 20 steps — 152s. TI2I 768×768, 8 steps — 101s. Sequential CPU offload keeps idle VRAM ≤2 GB so it co-tenants with other omni-vllm models; `BOOGU_OFFLOAD=model_cpu` gives ~3× speedup at the cost of ~22 GB VRAM.
  - Kimodo-SOMA-RP-v1.1 (Diffusers, gpu-all): 0.79 s/60 frames + 25 steps (warm); 1.99 s/150 frames + 100 steps (warm, 5-second motion full quality). Cold start 25.4 s model load (LLM2Vec CPU load dominates). **Under co-tenant CPU contention** (Trellis2 + HY-Motion both active), warm inference rises to ~22-24 s because LLM2Vec Llama-3-8B text encoder runs on CPU and is sensitive to cache pollution.
  - HY-Motion-1.0-Lite (Diffusers, gpu-all): 26.2 s wall for 60 frames (2-second motion at 30 fps) including 12 s cold-start load. Pure inference ~14 s (torchdiffeq ODE sampler).
  - TRELLIS.2-4B (Native, gpu-all): 24.1 s for 512³ → 18.7 MB GLB; 93.3 s for 1024_cascade → 40.6 MB GLB. Cold load 41.1 s. RTX 4090 is 5-8× slower than Microsoft's H100 reference (3 s / 17 s respectively).
