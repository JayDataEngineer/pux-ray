# VibeVoice TTS — Optimizations

## Integration Level: PARTIAL

VibeVoice TTS is more complex than ASR — it has TWO generation modes (autoregressive LLM + DDPM diffusion). The LLM backbone could be Native, but the DDPM diffusion head and dual tokenizer system add complexity.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| language_model (Qwen2) | GGUF Q4/Q8, INT8 | No | No | **YES — nanovllm, CUDA graphs (autoregressive part)** |
| acoustic_tokenizer | No (VAE, small) | **Yes — shared with VibeVoice ASR** | No | No |
| acoustic_connector | No (tiny) | **Yes — shared with VibeVoice ASR** | No | No |
| semantic_tokenizer | No (small) | No | No | No |
| semantic_connector | No (tiny) | No | No | No |
| prediction_head (DDPM) | INT8 (diffusion MLP) | No | No | No (custom diffusion) |
| lm_head | INT8 | No | No | Yes |

## Available Optimizations

### 1. nanovllm + CUDA Graphs for LLM Backbone (HIGH IMPACT)
The Qwen2 backbone handles the autoregressive text + speech token generation (Stage 2). This is standard LLM inference — nanovllm + CUDA graphs apply directly.

Does NOT apply to the DDPM diffusion head (Stage 3) which is a different generation pattern.

### 2. INT8 for DDPM Prediction Head (LOW-MEDIUM IMPACT)
The diffusion head is called N times during speech generation. INT8 quantization reduces its footprint. But it's only ~1GB — marginal gain.

### 3. GGUF Quantization for Backbone (HIGH IMPACT)
VibeVoice-7B at GGUF Q4: ~16GB → ~4-5GB
Significant VRAM savings for the largest component.

### 4. Shared Components with ASR (MEDIUM IMPACT)
acoustic_tokenizer + acoustic_connector shared between ASR and TTS. One copy if both services are available.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| nanovllm (LLM backbone only) | No | **Yes — for autoregressive stage** |
| CUDA graphs (LLM backbone only) | No | **Yes — for autoregressive stage** |
| GGUF quantization | No | Yes — Qwen2 backbone |
| INT8 prediction head | No | Yes — marginal gain |
| Shared acoustic components | No | Yes — with VibeVoice ASR |

## The Dual-Mode Challenge

VibeVoice TTS has two distinct generation phases:
1. **Autoregressive** (Qwen2 LLM) → nanovllm + CUDA graphs apply
2. **Diffusion** (DDPM head) → standard diffusion, no nanovllm/CUDA graphs

This means VibeVoice TTS can only be PARTIALLY Native. The LLM backbone gets Native optimizations. The DDPM head stays custom. This is the first model where the integration level is split across components.

## VRAM Budget Impact (7B variant)

| Scenario | Peak VRAM |
|----------|-----------|
| Current (BF16, mmgp only) | ~18GB |
| INT8 backbone | ~10GB |
| GGUF Q4 backbone | ~7GB |
| GGUF Q4 + nanovllm + CUDA graphs | ~7GB (faster AR stage) |

## Upgrade Priority: MEDIUM

Similar to MOSS and VibeVoice ASR — the Qwen2 backbone can be upgraded to Native. But the DDPM diffusion head and dual-tokenizer system add complexity not present in those models.
