# Pixal3D — Optimizations

## Integration Level: PARTIAL

Same as TRELLIS — multi-stage pipeline with custom VRAM management. 13 components, even more complex than TRELLIS.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| ss_flow_model | INT8 | **Yes — shared with TRELLIS (same weights)** | No | No |
| ss_decoder | Minimal | **Yes — shared with TRELLIS** | No | No |
| slat_flow_512 | INT8 | **Yes — shared with TRELLIS** | No | No |
| slat_flow_1024 | INT8 | **Yes — shared with TRELLIS** | No | No |
| shape_decoder | Minimal | **Yes — shared with TRELLIS** | No | No |
| tex_slat_flow_512 | INT8 | No (Pixal3D-specific) | No | No |
| tex_slat_flow_1024 | INT8 | No (Pixal3D-specific) | No | No |
| tex_decoder | Minimal | No | No | No |
| image_cond_ss | FP16 | No (projection-specific) | No | No |
| image_cond_shape_512 | FP16 | No | No | No |
| image_cond_shape_1024 | FP16 | No | No | No |
| image_cond_tex_1024 | FP16 | No | No | No |
| rembg | No | **Yes — shared with TRELLIS, AniGen** | BiRefNet-lite | No |

## Available Optimizations

### 1. Shared TRELLIS Components (HIGHEST IMPACT)
Pixal3D shares 5 components with TRELLIS (same weights):
- ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024, shape_decoder

If TRELLIS is loaded, Pixal3D skips loading ~10GB of weights. Only the 4 DiNOv3 conditioners + 2 texture flow models + tex_decoder need loading.

This is the strongest cross-model sharing case in the entire stack.

### 2. INT8 for Flow Models (HIGH IMPACT)
Same as TRELLIS — flow models quantize well to INT8.
- TRELLIS-shared flows: already covered by TRELLIS optimization
- tex_slat_flow_512: ~3GB → ~1.5GB
- tex_slat_flow_1024: ~4GB → ~2GB

### 3. Shared rembg/DiNOv (MEDIUM IMPACT)
BiRefNet shared across TRELLIS, AniGen, Pixal3D.
DiNOv3 conditioners are Pixal3D-specific (projection-mode, different from TRELLIS's DiNOv2).

### 4. Batch Flow Stages (MEDIUM IMPACT)
Same as TRELLIS — flow matching stages can batch_size=2.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| Shared TRELLIS components | No | **Yes — 5 components, ~10GB** |
| INT8 flow models | No | Yes |
| Shared rembg | No | Yes |
| Batch flow stages | No | Yes |
| Stage prefetch | No | Custom |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (all BF16, no sharing) | ~18GB |
| Shared TRELLIS components | ~10GB (only Pixal3D-specific loads) |
| Shared + INT8 Pixal3D flows | ~7GB |
| Shared + INT8 + batch 2 | ~9GB |

## Cross-Model Synergy with TRELLIS

Pixal3D is the strongest sharing candidate. If TRELLIS is already loaded, switching to Pixal3D requires loading only ~8GB of Pixal3D-specific components (4 DiNOv3 conditioners + 2 texture flows + tex_decoder). The ~10GB of shared components stay resident.
