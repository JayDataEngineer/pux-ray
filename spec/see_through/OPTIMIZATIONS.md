# See-Through — Optimizations

## Integration Level: PARTIAL → POTENTIAL NATIVE

See-Through uses standard SDXL diffusion components. These are Wan2GP's bread and butter — Wan2GP natively supports SDXL-family models. The question is whether the dual-pipeline (LayerDiff + Marigold) can be wired through Wan2GP's native path.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| ld_unet (SDXL) | INT8 | No | No | **YES — Wan2GP natively handles SDXL UNets** |
| ld_vae (SDXL) | FP16 | **Yes — standard SDXL VAE** | No | Yes |
| ld_trans_vae | Minimal (custom) | No | No | No |
| ld_text_encoder | FP16 | **Yes — SDXL text encoder** | No | **Yes** |
| ld_text_encoder_2 | FP16 / INT8 | **Yes — SDXL text encoder 2** | No | **Yes** |
| mg_unet (SD) | INT8 | No | No | **YES — Wan2GP natively handles SD UNets** |
| mg_vae (SD) | FP16 | **Yes — standard SD VAE** | No | Yes |
| mg_text_encoder (SD) | FP16 | **Yes — SD text encoder** | No | Yes |

## Path to Native

See-Through's components are STANDARD SDXL and SD models. Wan2GP already has native handlers for these architectures. The only custom part is:
- `ld_trans_vae` (TransparentVAE) — custom alpha decoder
- The dual-pipeline orchestration (run LayerDiff, then Marigold per-layer)

If Wan2GP can load and run the SDXL UNet natively, it gets ALL optimizations automatically: quantization, attention backends, MagCache step skipping, mmgp weight swapping.

### Steps to Native:
1. Register ld_unet and mg_unet as Wan2GP model types (they're standard SDXL/SD UNets)
2. Wan2GP handles: mmgp, quantization, attention, MagCache, scheduling
3. Keep custom: trans_vae decode, dual-pipeline orchestration
4. Result: Partial → mostly Native, small custom orchestration layer

## Available Optimizations

### 1. Wan2GP Native SDXL Path (HIGHEST IMPACT)
If the UNets are registered as Wan2GP model types, they get:
- Automatic mmgp weight swapping (already have)
- INT8 quantization (currently don't have)
- SageAttention 2 (currently don't have)
- MagCache step skipping (currently don't have)
- Wan2GP's flow matching schedulers

### 2. Shared SDXL Components (MEDIUM IMPACT)
SDXL text encoders and VAEs are used across many models (ComfyUI workflows, etc.). One copy shared.

### 3. INT8 Quantization (MEDIUM IMPACT)
- ld_unet: ~3GB → ~1.5GB
- mg_unet: ~3GB → ~1.5GB
- ld_text_encoder_2: ~2GB → ~1GB

### 4. MagCache for UNet Steps (MEDIUM IMPACT)
Both LayerDiff and Marigold run repeated denoising steps through their UNets. MagCache can skip steps when output change is below threshold.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| Wan2GP native SDXL path | No | **Yes — components are standard SDXL/SD** |
| INT8 quantization | No | Yes — Wan2GP qtypes |
| Shared SDXL components | No | Yes — cross-model |
| MagCache step skipping | No | Yes — standard diffusion UNets |
| SageAttention | No | Yes — Wan2GP attention backends |
| nanovllm | No | No — not autoregressive |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (FP32/FP16, mmgp only) | ~8-10GB |
| Wan2GP native + INT8 UNets | ~5-6GB |
| Native + INT8 + MagCache | ~5-6GB (faster, fewer steps) |

## Upgrade Priority: MEDIUM-HIGH

The components are standard SDXL/SD. Wiring them through Wan2GP's native path gives the most optimization gain for the least effort compared to other models.
