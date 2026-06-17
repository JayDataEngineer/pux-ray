# Inference Engine Profiling Results

## Qwen-Image-Edit-2511 (20B MMDiT) — vLLM-Omni

**Status:** ✅ WORKING on `vllm/vllm-omni:latest`

### Configuration
| Setting | Value |
|---------|-------|
| Image | `vllm/vllm-omni:latest` |
| Port | 8093 |
| Model | `2511-fp8` (base FP8 weights) |
| Quantization | FP8 weight-only (via pipeline patch) |
| Memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| VAE | Slicing + Tiling enabled |
| Cache | Cache-DiT with TaylorSeer (Fn=1, Bn=0, W=4) |
| Text encoder | CPU offloaded (via pipeline patch) |
| VRAM usage | 21.2 GB / 24 GB (86%) |

### Benchmarks (512×512, 1 image)

| Steps | Latency | Output Size | Notes |
|-------|---------|-------------|-------|
| 4 | ~46 s | 787 KB | Cold start (model already loaded) |
| 20 | ~46 s | 787 KB | Identical latency (Cache-DiT dominates) |

Both step counts show similar latency due to Cache-DiT caching transformer blocks after warmup steps. The 4-step warmup phase dominates the wall-clock time.

### Observations
- Pipeline patch at `scripts/pipeline_qwen_image_edit_plus_patch.py` is **critical** — it monkey-patches `Fp8Config` to be weight-only (FP8 storage, BF16 matmul), avoiding Triton fp8e4nv `tl.dot()` crash, and offloads text encoder to CPU.
- Container uses `vllm/vllm-omni:latest` (not `fork-v1`). The `fork-v1` image forces `mp.set_start_method("fork")` which is CUDA-incompatible. `latest` correctly uses `mp.set_start_method("spawn")`.
- Cache-DiT warnings: "requires num_inference_steps to be passed explicitly" — minor, doesn't block inference.

---

## Z-Image-Turbo (W8A8 Block FP8) — vLLM-Omni

**Status:** ❌ BLOCKED on `forge-reg.local:30500/tech-noir/vllm-omni:fork-v1`

### Root Cause
Container `fork-v1` has Triton 3.6.0 which lacks `fp8e4nv` support in `tl.dot()`. The W8A8 Block FP8 kernel at `fp8_utils.py:779` fails with:
```
AssertionError: Unsupported lhs dtype fp8e4nv
```

### Blocking Issues
1. **Triton 3.6.0** — `tl.dot()` only supports `int8, uint8, float16, bfloat16, float32`. No `fp8e4nv`.
2. **Kernel configs** — No RTX 4090 configurations in `get_w8a8_block_fp8_configs()`.
3. **Fork issue** — `multiproc_executor.py:191` forces `mp.set_start_method("fork")` causing CUDA re-init error.

### Potential Solutions
1. **Convert via user's FP8 pipeline** (ModelOpt format) — same approach as qwen-image-edit
2. **Use ComfyUI fallback** for z-image-turbo
3. **Upgrade container** to `vllm/vllm-omni:latest` (has spawn fix, but may still have Triton issue)
4. **Patch z-image pipeline** similar to qwen — FP8 weight-only dequant to BF16

## MOSS SoundEffect-v2 — Audio Generation

**Status:** ✅ WORKING on `forge-reg.local:30500/tech-noir/moss:latest`

### Configuration
| Setting | Value |
|---------|-------|
| Image | `forge-reg.local:30500/tech-noir/moss:latest` |
| Port | 8050 |
| Model | `moss-soundeffect-v2` (Wan Audio based) |
| Framework | Custom Python HTTP server |
| VRAM usage | ~13 GB (53%) |
| Dependencies | Required `build-essential` (for Triton JIT) + `diffusers` (both installed at runtime) |

### Benchmarks (3s audio, 50 steps, cfg=4.0)

| Run | Latency | Notes |
|-----|---------|-------|
| Cold (Triton compile) | ~53.7 s | Includes Triton kernel compilation |
| Warm | ~4.8 s | Pure inference time (RTF ~1.5x) |

### Notes
- Container was missing `diffusers` and `build-essential` — installed at runtime
- Model loads in ~12.7s
- Generated audio: 48kHz WAV, ~288KB for 3s
- API: `POST /generate` with JSON body `{"prompt": "...", "model": "moss-soundeffect-v2", "seconds": 3}`

## CrispASR (Speech Recognition) — whisper base

**Status:** ✅ WORKING on `ghcr.io/crispstrobe/crispasr:main-cuda-12`

### Configuration
| Setting | Value |
|---------|-------|
| Image | `ghcr.io/crispstrobe/crispasr:main-cuda-12` (v0.7.2) |
| Port | 8051 |
| Backend | `whisper` (auto-downloaded ggml-base.bin) |
| VRAM usage | Minimal (~200MB) |
| API | `POST /v1/audio/transcriptions` (OpenAI-compatible) |

### Benchmarks

| Audio | Duration | Latency | Output |
|-------|----------|---------|--------|
| Sine wave (2s) | 2s | 74 ms | Empty (no speech) |
| TTS sample (speech) | ~5s | 71 ms | Accurate transcription |

### Notes
- Vibevoice ASR GGUFs on disk are actually TTS-only models. Need to download proper vibevoice ASR model for diarization features.
- Container started successfully with auto-download of whisper fallback.
- API fully OpenAI-compatible.

## ACE-Step

**Status:** ⏳ NOT STARTED — needs GGUF download and Docker build
