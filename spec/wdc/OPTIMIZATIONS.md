# WhatDreamsCost — Optimizations

## Integration Level: CONDITIONING STRATEGY (no new model handler needed)

WDC is defined entirely in terms of conditioning parameters on LTX Video,
which is already a Wan2GP built-in (`models.ltx2.ltx2_handler`).

## What's Available

| Component | Wan2GP Model | Integration | Status |
|-----------|-------------|-------------|--------|
| LTX Video 2.3 | `ltx2` (vendor handler) | Built-in | Available now |
| LTX Video VAE | Part of LTX2 pipe dict | Built-in | Available now |
| Audio VAE | Part of LTX2 pipe dict | Built-in | Available now |
| Spatial upscaler | `loras_selected` or built-in | Built-in | Available now |
| Dual CLIP | Part of LTX2 pipe dict | Built-in | Available now |
| Timeline segmentation | **Not in Wan2GP** | Missing | Needs implementation |
| Frame keyframe interpolation | **Not in Wan2GP** | Missing | Needs implementation |

## What We Gain vs ComfyUI

| Aspect | ComfyUI (current) | Wan2GP Workflow (new) |
|--------|------------------|----------------------|
| Pipeline | 97-node JSON graph | Single `svc.infer()` call |
| Model loading | Subprocess boot | mmgp streaming |
| Audio sync | Custom node wiring | `audio_b64` parameter |

## Optimization Gaps

### 1. Timeline Segmentation (MEDIUM)
WDC's LTXDirector creates multi-shot videos by defining segment boundaries
with per-segment prompts and camera guides. In Wan2GP, this would be a
pre-processing step that generates a single combined prompt/conditioning
tensor — not a separate model call. Can be implemented as a utility
function in `services/workflows/wdc.py`.

### 2. Frame Keyframe Interpolation (LOW)
FFLF conditioning creates a smooth interpolation between first and last
frames. Wan2GP's LTX handler may already support this via `image_start`
and `image_end` parameters — needs verification. If not, it's a small
pre-processing step (blend the two images into a conditioning tensor).

### 3. Audio Conditioning (LOW)
Wan2GPService already passes `audio_b64` through to the LTX pipeline
(via `_SAFE_PASSTHROUGH`). WDC's audio sync is simply wiring this
parameter — no custom code needed.

## Priority Upgrade Path

1. **Publish Wan2GPService-based WDC functions** — done (services/workflows/wdc.py)
2. **Verify `image_start`/`image_end` support in LTX handler** — test on hardware
3. **Port timeline segmentation as pre-processing** — utility function
4. **Benchmark vs ComfyUI WDC workflows**
