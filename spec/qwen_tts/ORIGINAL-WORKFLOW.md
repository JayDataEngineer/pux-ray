# Qwen-TTS (Standard) — Original Workflow

**Source**: Alibaba Qwen-TTS (via Wan2GP at `opt/wan2gp/models/TTS/qwen3/` and `qwen3_handler.py`)
**Type**: Text-to-speech with voice cloning and voice design
**Architecture**: Qwen3TTSModel — single unified model with talker decoder and speech tokenizer

## Inference Pipeline

### Stage 1: Text Processing
```
text prompt → Qwen2Tokenizer → tokenize → segment splitting
→ parse speaker labels for multi-speaker dialogue
```

### Stage 2: Speaker Encoding (for voice cloning)
```
reference audio → Qwen3TTSSpeakerEncoder → speaker embeddings
→ conditions the generation with voice characteristics
```

### Stage 3: Audio Generation (Autoregressive)
```
text tokens + speaker embeddings → Qwen3TTSModel.generate()
  → talker decoder generates audio codes autoregressively
  → code_predictor generates codebook entries
  → sequential token generation with KV cache
```

### Stage 4: Decode
```
audio codes → speech tokenizer (12Hz codec) → waveform
```

## Components

| Module | Role | Notes |
|--------|------|-------|
| Qwen3TTSModel | Main unified model | Contains talker, code_predictor, speech_tokenizer |
| Qwen3TTSTalkerModel | Decoder with attention | Multi-head attention + talker text MLP |
| Qwen3TTSTalkerCodePredictor | Code generation | Codebook prediction |
| Qwen3TTSSpeakerEncoder | Voice cloning encoder | Reference audio → speaker embedding |
| Qwen3TTSProcessor | Text preprocessing | Tokenization + audio tokenizer wrapper |
| Speech Tokenizer | 12Hz neural codec | Audio code → waveform decode |

## Key Characteristics

- **Three modes**: CustomVoice (9 predefined speakers), VoiceDesign (zero-shot), Base (reference cloning)
- **HuggingFace transformers** — uses standard `model.generate()` API
- **Autoregressive generation** — sequential token generation with KV cache
- **12Hz speech tokenizer** — lower frame rate than some models
- **Multi-speaker dialogue** — parse speaker labels for conversational TTS

## Relationship to faster_qwen3_tts

| Feature | Qwen-TTS (standard) | faster_qwen3_tts |
|---------|-------------------|-----------------|
| Architecture | Single `Qwen3TTSModel` | Decomposed: talker + code_predictor + speech_tokenizer |
| Inference | `model.generate()` (HuggingFace) | Direct `forward()` calls (custom) |
| Speed | Baseline | 6-10x faster with CUDA graphs |
| Memory | Wan2GP manages | Granular per-module control |
| KV cache | Dynamic | StaticCache (CUDA graph compatible) |

faster_qwen3_tts is an optimized decomposition of the same underlying Qwen3-TTS model. It breaks the monolithic model into components, uses StaticCache instead of dynamic KV cache, and captures CUDA graphs for fixed-shape decode steps.
