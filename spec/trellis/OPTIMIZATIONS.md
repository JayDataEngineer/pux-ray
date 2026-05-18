# TRELLIS.2 — Optimizations

## Integration Level: PARTIAL

TRELLIS will likely always be Partial. Its 8-stage pipeline with custom VRAM management (spatial cache clearing, intermediate CPU offloading, per-component precision) doesn't map to Wan2GP's single-generate pattern.

Wan2GP provides mmgp weight swapping. Everything else is custom handler code.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| ss_flow_model | INT8 (flow matching transformer) | No | No | No |
| ss_decoder | Minimal gain | No | No | No |
| slat_flow_512 | INT8 (flow matching transformer) | No | No | No |
| slat_flow_1024 | INT8 (flow matching transformer) | No | No | No |
| tex_slat_flow_1024 | INT8 (flow matching transformer) | No | No | No |
| shape_decoder | Minimal (spconv, custom) | No | No | No |
| tex_decoder | Minimal (spconv, custom) | No | No | No |
| image_cond (DiNOv2) | FP16 | **Yes — shared with AniGen, Pixal3D** | No | No |
| rembg (BiRefNet) | No (FP32 required) | **Yes — shared with AniGen** | **BiRefNet-lite** | No |

## Available Optimizations

### 1. INT8 Quantization for Flow Models (HIGH IMPACT)
The 3 flow models (~10GB combined at BF16) are the heaviest components. Wan2GP's `qtypes/int8` via optimum.quanto can quantize these during loading.

- slat_flow_512: ~3GB → ~1.5GB
- slat_flow_1024: ~4GB → ~2GB
- tex_slat_flow_1024: ~4GB → ~2GB
- ss_flow_model: ~2GB → ~1GB
- **Total savings: ~6.5GB**

Pre-made INT8 quants from unsloth or similar could skip JIT quantization overhead.

### 2. Shared DINOv2 (MEDIUM IMPACT)
TRELLIS, AniGen, and Pixal3D all load DiNOv2 for image conditioning. One copy in RAM shared across all three saves ~1GB per model switch.

### 3. Shared BiRefNet/rembg (LOW IMPACT)
TRELLIS and AniGen both use BiRefNet for background removal. One copy, shared. Could also swap for BiRefNet-lite (lighter, faster, slightly lower quality).

### 4. Batch Flow Stages (MEDIUM IMPACT)
The flow matching stages (ss_flow, slat_flow, tex_slat_flow) are repeated transformer calls — N denoising steps through the same model. These could batch_size=2, processing two images through the same forward pass.

Weight loading amortized across 2 samples. Intermediate tensors double but weights don't.

### 5. Stage Prefetch (MEDIUM IMPACT)
Since stages are strictly sequential and known ahead of time, stage N+1's weights could be loaded into pinned RAM while stage N computes. Currently not done — each stage loads on demand.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| INT8 quantization | No | Yes — Wan2GP qtypes |
| Shared DINOv2 | No | Yes — cross-model |
| Shared rembg | No | Yes — cross-model |
| Batch flow stages | No | Yes — Wan2GP native |
| Stage prefetch | No | Custom — needs orchestration |
| nanovllm | No | No — not autoregressive |
| CUDA graphs | No | No — variable sparse tensor shapes |
| MagCache | No | Maybe — flow matching may support step skipping |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (all BF16, no optimizations) | ~14GB |
| INT8 flow models | ~7.5GB |
| INT8 + shared DINOv2 | ~6.5GB |
| INT8 + shared DINOv2 + batch 2 | ~9GB |
| INT8 + shared + batch 2 + prefetch | ~9GB (faster) |
