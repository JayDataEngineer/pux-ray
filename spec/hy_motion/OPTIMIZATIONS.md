# HY-Motion — Optimizations

## Integration Level: PARTIAL

Multi-stage pipeline with ODE sampling and custom motion decoding. Wan2GP provides mmgp weight swapping only.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| motion_transformer | INT8 (repeated ODE steps) | No | No | No (ODE, not standard diffusion) |
| text_encoder (Qwen3-8B) | GGUF Q4/Q8, INT8 | **Maybe — same family as MOSS backbone** | No | Partial (encoding only, not generation) |
| text_encoder (CLIP) | FP16 | **Yes — CLIP used across many models** | No | No |
| Small components (MLP, etc.) | No | No | No | No |

## Available Optimizations

### 1. INT8 Quantization for Motion Transformer (MEDIUM IMPACT)
The motion_transformer is called 50 times during ODE sampling. INT8 would reduce its VRAM footprint and potentially speed up each step.

### 2. Quantized Text Encoder (HIGH IMPACT)
Qwen3-8B text encoder at ~16GB is the heaviest component. Options:
- GGUF Q4: ~4-5GB
- INT8 via optimum.quanto: ~8GB
- Since it's only used for encoding (single forward pass), quality loss from quantization is less impactful than for generation models

### 3. Shared CLIP (LOW IMPACT)
CLIP text encoder is used across many models. One shared copy saves marginal VRAM.

### 4. MagCache / Step Skipping (UNCERTAIN)
ODE sampling is 50 sequential steps through the same transformer. If magnitude changes between steps follow a pattern, step skipping may apply. Needs testing — ODE dynamics differ from standard diffusion.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| INT8 motion_transformer | No | Yes — Wan2GP qtypes |
| Quantized text encoder | No | Yes — GGUF or INT8 |
| Shared CLIP | No | Yes — cross-model |
| Step skipping | No | Maybe — needs research |
| nanovllm | No | No — not autoregressive |
| CUDA graphs | No | No — ODE variable step sizes |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current | ~20GB+ (Qwen3-8B alone is 16GB) |
| INT8 text encoder | ~12GB |
| INT8 text encoder + INT8 motion transformer | ~10GB |
| GGUF Q4 text encoder + INT8 motion | ~8GB |

## Key Constraint

The Qwen3-8B text encoder dominates VRAM. At BF16 it's ~16GB — leaving almost no room for the motion transformer. Quantizing the text encoder is not optional, it's required for this model to run comfortably on 24GB.
