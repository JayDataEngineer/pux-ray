# MOSS-SoundEffect — Original Workflow

**Source**: https://github.com/OpenMOSS/MOSS-TTS
**Paper**: https://arxiv.org/abs/2603.18090
**Type**: Text-to-sound-effect
**Architecture**: MossTTSDelay — single 8B-parameter Qwen3 transformer with multi-head parallel RVQ prediction

## Inference Pipeline

### Stage 1: Tokenization
```
text prompt → MossTTSDelayProcessor → input_ids [B, S, 33]
  ├─ text token in channel 0
  └─ padding tokens in channels 1-32
```

### Stage 2: Autoregressive Generation
```
For each timestep (up to max_new_tokens):
  ├─ Forward pass: Qwen3-8B backbone → 33 LM head logits
  ├─ Head 0: sample text/audio sequence token
  ├─ Heads 1-32: sample 32 RVQ codebook layers in parallel
  ├─ Delay pattern: Head K at step T predicts layer K of frame T-K+1
  ├─ Append to sequence with KV cache
  └─ Stop on audio_end_token
```

### Stage 3: Decode
```
Generated tokens → Processor.decode() → extract audio_codes
audio_codes → MOSS Audio Tokenizer → numpy array → WAV
```

## Components

| Component | What it is | Precision | Size |
|-----------|-----------|-----------|------|
| MossTTSDelayModel | Qwen3-8B backbone + 33 LM heads + 32 VQ embeddings | BF16 | ~16GB |
| Audio Tokenizer | RVQ tokens → waveform decoder | FP32 | ~200MB |
| MossTTSDelayProcessor | Tokenizer + audio tokenizer wrapper | N/A | N/A |

## Key Characteristics

- **Single autoregressive transformer** — no diffusion, no VAE, no multi-stage pipeline
- **Multi-head parallel prediction** — all 32 RVQ layers predicted in one forward pass via delay pattern
- **KV cache grows with sequence length** — ~1-2GB for typical generations
- **Audio tokenizer is lightweight** — runs once at the end, FP32, separate from main model
- **Peak VRAM ~18-19GB** — fits in 24GB with mmgp

## Core Mechanism: Delay Pattern

At each timestep, Head K predicts Layer K of frame T-K+1. This means:
- Step 1: Head 1 predicts Layer 1 of frame 1
- Step 2: Head 1 predicts Layer 1 of frame 2, Head 2 predicts Layer 2 of frame 1
- Entire frame's worth of multi-layer audio tokens generated in single forward pass
- No nested loops, no hierarchical transformers
