# faster_qwen3_tts — Optimizations

## Integration Level: NATIVE

faster_qwen3_tts is already Native Custom. It's the reference implementation for what Native looks like.

It uses: mmgp weight swapping, CUDA graphs (custom implementation), StaticCache for KV management, Wan2GP family_handler pattern.

## Component Triage

| Component | Quantizable | Shared | Replaceable | Wan2GP Native |
|-----------|------------|--------|-------------|---------------|
| talker (28-layer transformer) | INT8, GGUF | No | No | Yes — CUDA graphs + mmgp |
| code_predictor (5-layer) | INT8 | No | No | Yes — CUDA graphed |
| speech_tokenizer | No (FP32, small) | No | No | No (too small) |
| text_projection | No (tiny) | No | No | No (too small) |

## Already Applied

| Optimization | Status |
|-------------|--------|
| mmgp weight swapping | Applied |
| CUDA graphs (custom) | Applied — 6-10x speedup |
| StaticCache KV management | Applied |
| Wan2GP family_handler pattern | Applied |

## Available But Not Applied

### 1. nanovllm Paged Attention (LOW IMPACT)
Could replace StaticCache with nanovllm's paged attention for more efficient KV management. Marginal gain since CUDA graphs already provide the speedup.

### 2. INT8 Quantization (MEDIUM IMPACT)
talker at INT8 would reduce VRAM footprint. Currently runs at full precision.
- talker: ~X GB → ~X/2 GB
- code_predictor: ~Y GB → ~Y/2 GB

### 3. GGUF Quantization (MEDIUM IMPACT)
Pre-made GGUF Q4/Q8 weights for Qwen3-TTS backbone would reduce loading time and VRAM.

### 4. Wan2GP cudagraph_kit (LOW IMPACT)
Currently uses custom CUDA graph implementation. Could migrate to Wan2GP's shared cudagraph_kit for consistency, but no performance gain.

## VRAM Budget Impact

| Scenario | Peak VRAM |
|----------|-----------|
| Current (Native, CUDA graphs) | moderate |
| + INT8 talker + predictor | lower |
| + GGUF Q4 backbone | much lower |

## Status: REFERENCE IMPLEMENTATION

This model IS the target state for other autoregressive models (MOSS). When upgrading MOSS to Native, faster_qwen3_tts is the pattern to follow.
