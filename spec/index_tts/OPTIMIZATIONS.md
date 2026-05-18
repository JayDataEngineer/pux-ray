# IndexTTS v2 — Optimizations

## Integration Level: NATIVE (Wan2GP Built-in)

IndexTTS v2 is a Wan2GP built-in TTS model. Handler at `opt/wan2gp/models/TTS/index_tts2_handler.py`. Wan2GP manages the full inference flow — pipe dict, mmgp, the works.

This is NOT a custom handler we wrote. It ships with Wan2GP.

## Wan2GP Optimizations Applied

| Optimization | Status | Notes |
|-------------|--------|-------|
| mmgp weight swapping | Applied | Standard Wan2GP pipe dict |
| Wan2GP family_handler pattern | Applied | Standard handler interface |
| CUDA Graph acceleration | Applied | Wan2GP handles graph capture |

## Wan2GP Optimizations Available

| Optimization | Available | Notes |
|-------------|-----------|-------|
| INT8 quantization | Yes | Via Wan2GP qtypes |
| GGUF quantization | Maybe | If backbone has GGUF-compatible architecture |
| SageAttention | Yes | Wan2GP attention backend selection |
| mmgp profiles | Yes | Profile selection (1-5) for RAM/VRAM tradeoff |

## Component Triage

As a built-in model, Wan2GP handles the component decomposition internally. The handler manages:
- UnifiedVoice GPT model
- S2Mel acoustic model with CFM
- BigVGAN vocoder
- Auxiliary models (w2v-bert, CAMPPlus, QwenEmotion)

All managed through Wan2GP's standard pipe dict + coTenantsMap mechanism.

## No Action Required

This model is fully managed by Wan2GP. Any optimization improvements come from Wan2GP upstream updates, not from our orchestration layer.

The spec documents it for completeness — it's a reference point for what Native Built-in looks like in our system.
