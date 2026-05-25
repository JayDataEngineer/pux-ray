# Forge GPU Model Tracker

Real-time status of every model on the Forge (RTX 4090, 24GB VRAM).

Last updated: 2026-05-25

## RESOLVED: GPU Driver — Secure Boot (2026-05-25)

DKMS 595.71.05 modules were rejected by Secure Boot. Fix: used pre-compiled objects from `linux-objects-nvidia-595` package + Canonical signatures from `linux-signatures-nvidia-595` package. Build script at `/lib/modules/$(uname -r)/kernel/nvidia-595/bits/BUILD` links objects and appends Canonical signature. Result: 595.71.05 module signed by "Canonical Ltd. Kernel Module Signing" — Secure Boot accepts it.

**Note:** DKMS `nvidia-dkms-595` package will overwrite these on kernel updates. May need to re-run the build script or hold the package.

## Status Legend

| Status | Meaning |
|--------|---------|
| PASS | Model loads, infers, produces valid output |
| SKIP | Model weights missing or needs config (HF_TOKEN, etc.) |
| FIX | Known bug, fix in progress |
| BLOCK | Needs Docker image change or external dependency |

## Model Test Results

### 3D Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| TRELLIS.2 | `trellis/trellis` | image_b64 | model/gltf-binary | PASS | image_cond fix (non-Module), always compute cond_1024 |
| AniGen | `anigen/anigen` | image_b64 | rigged-3D | PASS | |

### Motion Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| HY-Motion 1.0 | `hy_motion/hy-motion-1.0` | prompt | video/mp4 | PASS | |
| Kimodo SOMA-RP | `kimodo/kimodo-soma-rp` | prompt | motion NPZ | PASS | autocast + model.float() + TEXT_ENCODER_DEVICE=cpu |
| Kimodo G1-RP | `kimodo/kimodo-g1-rp` | prompt | motion NPZ | PASS | Same fix as SOMA |
| Kimodo SMPLX-RP | `kimodo/kimodo-smplx-rp` | prompt | motion NPZ | PASS | Same fix as SOMA |

### Audio Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| ACE-Step v1.5 | `tts/ace_step_v1_5` | prompt | audio/wav | PASS | |
| MOSS-SoundEffect | `moss/moss-soundeffect` | prompt | audio/wav | PASS | |
| MOSS TTS Nano | `moss/moss-tts-nano` | prompt | audio/wav | PASS | |
| MOSS TTS Local Transformer | `moss/moss-tts-local-transformer` | prompt | audio/wav | PASS | |
| MOSS TTS | `moss/moss-tts` | prompt | audio/wav | PASS | |
| MOSS TTSD | `moss/moss-ttsd` | prompt | audio/wav | PASS | |
| MOSS VoiceGenerator | `moss/moss-voicegenerator` | prompt | audio/wav | PASS | |
| MOSS TTS Realtime | `moss/moss-tts-realtime` | prompt | audio/wav | PASS | |
| IndexTTS v2 | `tts/index_tts2` | prompt + audio_ref | audio/wav | PASS | Voice cloning |

### Video Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| Wan T2V 14B | `wan/t2v` | prompt | video/mp4 | PASS | Quanto INT8 weight fix: skip dtype normalization for quantized models |

### Image Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| See-Through | `see_through/see-through` | anime_rgba_b64 | layers | SKIP | Model needs specific anime with clear layer structure; cv2 resize fails on test images |

### Multimodal Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| Lance 3B AWQ | `lance/lance-image-awq` | prompt | image | SKIP | mRoPE requires transformers>=4.46; Docker image has incompatible version |

### Infrastructure

| Test | Status | Notes |
|------|--------|-------|
| Forge status endpoint | PASS | Returns vram_total_mb, loaded services |
| Forge release idempotent | PASS | Double-release returns "not_loaded" |

## Known Issues & Fixes

### Kimodo BFloat16 (FIXED)

**Problem:** Denoiser checkpoints store weights in bfloat16. Text encoder (LLM2VecEncoder, not nn.Module) runs bfloat16 on CPU. When text features move to GPU via `.to(device)`, bfloat16 hits float32 denoiser → dtype mismatch.

**Fix (3 parts, all in kimodo_handler.py):**
1. `os.environ["TEXT_ENCODER_DEVICE"] = "cpu"` — runs LLM2VecEncoder on CPU, frees ~14GB VRAM
2. `model.float()` after load — converts denoiser from bfloat16 to float32
3. `torch.autocast("cuda", dtype=torch.float32)` during inference — auto-promotes any remaining bfloat16 tensors

### Wan T2V Quantized Weight Dequantization (FIXED)

**Problem:** `_apply_mmgp_profile()` converted all parameters to bfloat16, including quanto INT8 quantized weights. The `p.data.to(torch.bfloat16)` call dequantized INT8 weights (1 byte/param) to bfloat16 (2 bytes/param), doubling model size from 14GB to 28GB. With 28GB transformer + 6.3GB T5 + 0.2GB VAE = 34.5GB total, mmgp couldn't fit anything in 24GB VRAM, causing 20+ minute timeouts and `'NoneType' object is not subscriptable` errors.

**Fix:** Detect quantized weights (INT8 dtype, QTensor attributes) and skip dtype normalization. Use Wan2GP's native profile 4 budgets (transformer: 100MB, text_encoder: 100MB, *: 3000MB) which match Wan2GP's `init_pipe()` defaults.

**File:** `services/wan2gp/deployment.py`

### See-Through Input

**Problem:** Synthetic test images caused cv2.resize errors (zero-dimension arrays from layerdiff). Needs real anime-style image with transparent layers.

**Fix:** conftest.py now downloads real anime image. `real_anime_rgba_b64` fixture provides proper RGBA PNG.

**File:** `tests/conftest.py`

### Lance Vendor Setup

**Problem:** Lance needs two vendor repos (bytedance/Lance + Reza2kn/lance-quant) and AWQ INT4 weights.

**Fix:** Repos cloned to /opt/lance and /opt/lance-quant. Weights exist at /mnt/data/models/lance/. Need to verify forge_lance.py adapter works end-to-end.

**File:** `services/lance/forge_lance.py`

## TRELLIS Fixes (DONE)

- **image_cond:** DinoV3FeatureExtractor is NOT nn.Module. Captured before mmgp filter, passed to _Pipeline separately.
- **BiRefNet:** Also NOT nn.Module. Skipped in eval() loop.
- **cond_1024:** Always compute both 512 and 1024 conditioning (texture sampling needs 1024 regardless of resolution).
- **sampler params:** Pass `rescale_t`, `guidance_rescale`, `guidance_interval` from pipeline.json to flow samplers.

## Running Tests

```bash
# Full suite
FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s

# Quick audio tests only
FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -k "audio"

# 3D/motion models
FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -k "trellis or anigen or kimodo"

# Skip slow models
FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -m "not slow"
```
