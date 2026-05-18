# AniGen — Optimizations

## Integration Level: PARTIAL

Same situation as TRELLIS — multi-stage pipeline with custom VRAM management. Wan2GP provides mmgp weight swapping only.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| ss_flow_model | INT8 (flow matching) | No | No | No |
| ss_decoder | Minimal gain | No | No | No |
| slat_flow_model | INT8 (flow matching) | No | No | No |
| slat_decoder | Minimal (spconv) | No | No | No |
| image_cond (DiNOv2) | FP16 | **Yes — shared with TRELLIS, Pixal3D** | No | No |
| dsine | No (FP32, custom) | No | No | No |
| rembg (BiRefNet) | No (FP32) | **Yes — shared with TRELLIS** | **BiRefNet-lite** | No |

## Available Optimizations

### 1. INT8 Quantization for Flow Models (HIGH IMPACT)
- ss_flow_model: ~2GB → ~1GB
- slat_flow_model: ~4GB → ~2GB
- **Total savings: ~3GB**

### 2. Shared DINOv2 (HIGH IMPACT)
AniGen uses the same DiNOv2 image conditioning as TRELLIS. If TRELLIS is loaded, AniGen skips loading image_cond entirely.

### 3. Shared BiRefNet/rembg (MEDIUM IMPACT)
Same as TRELLIS. One copy, shared. Or swap for BiRefNet-lite.

### 4. Batch Flow Stages (MEDIUM IMPACT)
Flow matching stages can batch_size=2 like TRELLIS.

### 5. Stage Prefetch (MEDIUM IMPACT)
Sequential stages, known ahead of time. Preload next stage while current computes.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| INT8 quantization | No | Yes — Wan2GP qtypes |
| Shared DINOv2 | No | Yes — with TRELLIS |
| Shared rembg | No | Yes — with TRELLIS |
| Batch flow stages | No | Yes |
| Stage prefetch | No | Custom |
| nanovllm / CUDA graphs | No | No — not autoregressive |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (all BF16, no optimizations) | ~18GB |
| INT8 flow models | ~15GB |
| INT8 + shared DINOv2 | ~14GB |
| INT8 + shared + batch 2 | ~17GB |

## Cross-Model Synergy with TRELLIS

AniGen and TRELLIS share enough components that switching between them could be near-instant:
- DINOv2 stays in RAM (shared)
- BiRefNet stays in RAM (shared)
- Only flow models + decoders swap
- With INT8 flow models, the swap set is ~5GB instead of ~10GB
