# MOSS-SoundEffect — Optimizations

## Integration Level: PARTIAL → NATIVE (upgradeable)

MOSS is currently Partial (mmgp only). It is a strong candidate for Native integration because it's a standard autoregressive transformer — the same pattern Wan2GP optimizes for LLMs.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| Qwen3-8B backbone | GGUF Q4/Q8, INT8 | No (unique model) | No | **YES — nanovllm, CUDA graphs** |
| Audio Tokenizer | No (FP32, small) | No | No | No (too small to matter) |

## Path to Native

MOSS is a Qwen3-8B autoregressive model. Wan2GP's shared layer has:
- **nanovllm** — paged attention, KV cache block manager, scheduling
- **CUDA graphs** — capture fixed-shape decode steps
- **GGUF quantization** — load pre-quantized Q4/Q8 weights
- **mmgp** — weight swapping (already used)

The autoregressive generate loop (Stage 2) is identical to LLM decode: same token, same transformer, repeated N times. This is EXACTLY what nanovllm + CUDA graphs are built for.

### Steps to Native:
1. Wire Qwen3-8B backbone into nanovllm's ModelRunner
2. Capture CUDA graphs for the decode step (fixed token shape per step)
3. Use StaticCache for KV management
4. Optional: load GGUF Q4/Q8 weights instead of BF16

## Available Optimizations

### 1. nanovllm Paged Attention (HIGH IMPACT)
The backbone's autoregressive loop is identical to LLM inference. nanovllm's paged KV cache would:
- Reduce KV cache memory fragmentation
- Enable efficient KV cache management
- This is what gives faster_qwen3_tts its 6-10x speedup

### 2. CUDA Graphs (HIGH IMPACT)
After prefill, every decode step has the same shape. Capture as CUDA graph:
- Eliminates CPU overhead between decode steps
- Combined with nanovllm, this is the faster_qwen3_tts pattern

### 3. GGUF Quantization (HIGH IMPACT)
Qwen3-8B backbone at GGUF Q4: ~16GB → ~4-5GB
Qwen3-8B backbone at GGUF Q8: ~16GB → ~8GB
Pre-made GGUF quants available from unsloth and similar sources.

Or INT8 via optimum.quanto: ~16GB → ~8GB (JIT during loading)

### 4. Batch Generation (MEDIUM IMPACT)
Two sound effects generated simultaneously through same backbone. Same weights, double the latents.

## Current vs Available

| Optimization | Currently Used | Available |
|-------------|---------------|-----------|
| mmgp weight swapping | Yes | Yes |
| nanovllm paged attention | No | **Yes — same pattern as faster_qwen3_tts** |
| CUDA graphs | No | **Yes — fixed-shape decode** |
| GGUF quantization | No | Yes — Wan2GP qtypes/gguf |
| INT8 quantization | No | Yes — Wan2GP qtypes + optimum.quanto |
| Batch generation | No | Yes — autoregressive batch |

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (BF16, mmgp only) | ~18-19GB |
| INT8 backbone | ~10GB |
| GGUF Q4 backbone | ~6GB |
| INT8 + nanovllm + CUDA graphs | ~10GB (much faster) |
| GGUF Q4 + nanovllm + CUDA graphs | ~6GB (fast + tiny) |

## Upgrade Priority: HIGH

MOSS is the best candidate for Native upgrade. It's a standard autoregressive transformer — the exact pattern Wan2GP's shared layer is optimized for. The path is proven by faster_qwen3_tts which already does this.
