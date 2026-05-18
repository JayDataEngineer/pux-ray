# Qwen-TTS (Standard) — Optimizations

## Integration Level: NATIVE (Wan2GP Built-in)

Standard Qwen-TTS is a Wan2GP built-in TTS model. Handler at `opt/wan2gp/models/TTS/qwen3_handler.py`. Wan2GP manages the full inference flow.

This is NOT a custom handler. It ships with Wan2GP.

## Wan2GP Optimizations Applied

| Optimization | Status | Notes |
|-------------|--------|-------|
| mmgp weight swapping | Applied | Standard Wan2GP pipe dict |
| Wan2GP family_handler pattern | Applied | Standard handler interface |
| mmgp profile selection | Applied | Profile 1-5 for RAM/VRAM tradeoff |

## Wan2GP Optimizations Available

| Optimization | Available | Notes |
|-------------|-----------|-------|
| INT8 quantization | Yes | Via Wan2GP qtypes |
| SageAttention | Yes | Wan2GP attention backend |
| mmgp profiles | Yes | Adjustable at load time |

## The faster_qwen3_tts Upgrade Path

Standard Qwen-TTS → faster_qwen3_tts is the Native upgrade path for this model.

faster_qwen3_tts takes the same underlying Qwen3-TTS model and:
1. Decomposes it into separate components (talker, code_predictor, speech_tokenizer)
2. Replaces dynamic KV cache with StaticCache (CUDA graph compatible)
3. Captures CUDA graphs for predictor loop (15-step) and decode loop
4. Uses direct `forward()` calls instead of HuggingFace `model.generate()`
5. Result: 6-10x speedup

This upgrade is already done. `faster_qwen3_tts` IS the optimized version of Qwen-TTS. Both are documented in the spec:
- `spec/qwen_tts/` — the built-in baseline
- `spec/faster_qwen3_tts/` — the Native Custom optimized version

## Component Triage

As a built-in model, Wan2GP handles everything. The only action item is: use `faster_qwen3_tts` instead of standard Qwen-TTS when speed matters (which is always).

## No Action Required

This model is fully managed by Wan2GP. It's documented for completeness — it's the baseline that faster_qwen3_tts optimizes.
