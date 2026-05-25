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
| Kimodo SOMA-RP | `kimodo/kimodo-soma-rp` | prompt | motion NPZ | FIX | BFloat16 fix: TEXT_ENCODER_DEVICE=cpu, removed .half() |
| Kimodo G1-RP | `kimodo/kimodo-g1-rp` | prompt | motion NPZ | FIX | Same fix as SOMA |
| Kimodo SMPLX-RP | `kimodo/kimodo-smplx-rp` | prompt | motion NPZ | FIX | Same fix as SOMA |

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
| Wan T2V 14B | `wan/t2v` | prompt | video/mp4 | FIX | Forge timeout increased to 1200s, need to verify ~6s clip inference |

### Image Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| See-Through | `see_through/see-through` | anime_rgba_b64 | layers | FIX | Real anime image fixture added to conftest |

### Multimodal Models

| Model | Service Key | Input | Output | Status | Notes |
|-------|------------|-------|--------|--------|-------|
| Lance 3B AWQ | `lance/lance-image-awq` | prompt | image | FIX | Vendor repos cloned, weights on PVC at /mnt/data/models/lance/ |

### Infrastructure

| Test | Status | Notes |
|------|--------|-------|
| Forge status endpoint | PASS | Returns vram_total_mb, loaded services |
| Forge release idempotent | PASS | Double-release returns "not_loaded" |

## Known Issues & Fixes

### Kimodo BFloat16 (FIX IN PROGRESS)

**Problem:** LLM2VecEncoder (wrapping Llama-3-8B) is NOT an nn.Module. `model.half()` only converts nn.Module submodules, so the text encoder stays in bfloat16 while the denoiser becomes float16. This causes "unsupported ScalarType BFloat16" and dtype mismatch errors.

**Fix:** Set `TEXT_ENCODER_DEVICE=cpu` before loading. Runs text encoder on CPU (~2-3GB RAM, no bfloat16 GPU issues), freeing ~14GB VRAM. Removed all `.half()` calls.

**File:** `opt/wan2gp/models/kimodo/kimodo_handler.py`

### Wan T2V Timeout

**Problem:** Wan T2V 14B is a massive model. Forge's internal `_load_with_cleanup` timeout was 600s, which killed the request before inference completed.

**Fix:** Increased timeout to 1200s. Also need to verify the request uses appropriate parameters for a ~6 second clip (not full quality).

**File:** `services/forge.py`

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
