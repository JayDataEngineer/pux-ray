# VibeVoice ASR — Original Workflow

**Source**: Microsoft VibeVoice (community fork: https://github.com/vibevoice-community/VibeVoice)
**Technical Report**: https://arxiv.org/pdf/2508.19205
**Type**: Speech-to-text (ASR) with diarization
**Architecture**: Qwen2-based LLM with acoustic tokenizer

## Inference Pipeline

### Stage 1: Audio Encoding
```
audio waveform → acoustic_tokenizer → acoustic latents
acoustic latents → acoustic_connector → audio embeddings (LM space)
```

### Stage 2: Transcription (Autoregressive)
```
audio embeddings + text prompt (language specification)
→ language_model (Qwen2 LLM) → autoregressive text generation
→ transcribed text with timestamps
```

## Components

| Module | Role | Size | Notes |
|--------|------|------|-------|
| language_model | Qwen2-based LLM | ~7GB (7B) / ~16GB (7B full) | Autoregressive text generation |
| acoustic_tokenizer | Audio → acoustic latents (VAE) | ~1GB | 6-stage encoder/decoder |
| acoustic_connector | Acoustic → LM space projection | ~tiny | Linear projection |
| lm_head | Vocabulary projection | ~small | Token prediction |

## Key Characteristics

- **Qwen2 backbone** — standard autoregressive LLM for transcription
- **Acoustic tokenizer** — VAE with ConvNeXt V2 blocks, 24kHz sample rate
- **Single-pass transcription** — audio encoded once, then LLM generates text
- **Supports diarization** — speaker identification in transcription output
- **Ultra-low frame rate** — 7.5 Hz continuous speech tokenization

## Model Variants

| Variant | Size | Notes |
|---------|------|-------|
| VibeVoice-Streaming-0.5B | 0.5B | Real-time streaming |
| VibeVoice-1.5B | 1.5B | Standard |
| VibeVoice-7B | 7B | Highest quality, ~16GB |
