# VibeVoice ASR — Optimizations

## Integration Level: PARTIAL → NATIVE (upgradeable)

VibeVoice ASR is a Qwen2-based autoregressive model — same family as LLMs. Strong candidate for Native integration via nanovllm + CUDA graphs.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| language_model (Qwen2) | GGUF Q4/Q8, INT8 | No | No | **YES — nanovllm, CUDA graphs** |
| acoustic_tokenizer | No (VAE, small) | **Yes — shared with VibeVoice TTS** | No | No |
| acoustic_connector | No (tiny) | **Yes — shared with VibeVoice TTS** | No | No |
| lm_head | INT8 | No | No | Yes |

## Path to Native

Same as MOSS — autoregressive Qwen2 backbone is exactly what nanovllm + CUDA graphs are built for.

### Steps to Native:
1. Wire Qwen2 backbone into nanovllm's ModelRunner
2. Capture CUDA graphs for the decode step (fixed token shape)
3. Use StaticCache for KV management
4. Optional: GGUF Q4/Q8 weights for reduced VRAM

## Available Optimizations

### 1. nanovllm + CUDA Graphs (HIGH IMPACT)
Qwen2 autoregressive transcription → same pattern as LLM inference.
Expected speedup similar to faster_qwen3_tts (6-10x).

### 2. GGUF Quantization (HIGH IMPACT)
VibeVoice-7B at GGUF Q4: ~16GB → ~4-5GB
VibeVoice-1.5B at GGUF Q4: already small
Pre-made GGUF quants may be available for Qwen2-family models.

### 3. Shared Components with VibeVoice TTS (MEDIUM IMPACT)
ASR and TTS share: acoustic_tokenizer, acoustic_connector, same Qwen2 family backbone.
If both are loaded, shared components stay resident.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| nanovllm paged attention | No | **Yes — Qwen2 autoregressive** |
| CUDA graphs | No | **Yes — fixed-shape decode** |
| GGUF quantization | No | Yes |
| Shared acoustic components | No | Yes — with VibeVoice TTS |

## VRAM Budget Impact (7B variant)

| Scenario | Peak VRAM |
|----------|-----------|
| Current (BF16, mmgp only) | ~16GB |
| INT8 backbone | ~8GB |
| GGUF Q4 backbone | ~5GB |
| INT8 + nanovllm + CUDA graphs | ~8GB (much faster) |

## Upgrade Priority: MEDIUM

Same pattern as MOSS upgrade. After MOSS proves the nanovllm + CUDA graph path, VibeVoice ASR follows the same blueprint.
