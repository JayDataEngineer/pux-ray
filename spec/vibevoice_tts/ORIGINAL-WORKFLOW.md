# VibeVoice TTS — Original Workflow

**Source**: Microsoft VibeVoice (community fork: https://github.com/vibevoice-community/VibeVoice)
**Technical Report**: https://arxiv.org/pdf/2508.19205
**Type**: Text-to-speech — expressive, long-form, multi-speaker conversational audio
**Architecture**: Qwen2 LLM + DDPM diffusion head + dual tokenizer system

## Inference Pipeline

### Stage 1: Text Tokenization
```
text prompt → chat template tokenization
optional: reference audio for voice cloning → acoustic encoding
```

### Stage 2: Autoregressive Generation
```
text tokens → language_model (Qwen2 LLM) → autoregressive generation
→ generates text + speech tokens interleaved
→ detects <|speech_diffusion|> token to trigger speech generation
```

### Stage 3: Speech Diffusion
```
speech tokens → prediction_head (DDPM) → samples audio latents
→ N diffusion denoising steps through VibeVoiceDiffusionHead
```

### Stage 4: Audio Decode
```
audio latents → acoustic_tokenizer.decode() → waveform
→ output: audio at 24kHz
```

## Components

| Module | Role | Size | Notes |
|--------|------|------|-------|
| language_model | Qwen2 LLM — text + speech token generation | ~16GB (7B) | Autoregressive backbone |
| acoustic_tokenizer | Speech ↔ acoustic latents (VAE) | ~1GB | 6-stage encoder/decoder |
| acoustic_connector | Acoustic → LM space | ~tiny | Linear projection |
| semantic_tokenizer | Speech → semantic features | ~500MB | TTS-specific |
| semantic_connector | Semantic → LM space | ~tiny | TTS-specific |
| prediction_head | DDPM diffusion head for speech | ~1GB | Timestep embedding + MLP |
| lm_head | Vocabulary projection | ~small | Token prediction |

## Key Characteristics

- **Dual tokenizer system** — acoustic (VAE) + semantic (feature extraction)
- **DDPM diffusion head** — generates speech latents from text tokens
- **Ultra-long form** — up to 90 minutes, 4 speakers
- **Voice cloning** — reference audio conditions the generation
- **Interleaved text + speech tokens** — LLM generates both simultaneously
- **Two generation modes**: autoregressive (LLM) + diffusion (DDPM head)
- **7.5 Hz frame rate** — continuous speech tokenization

## Model Variants

| Variant | Size | Notes |
|---------|------|-------|
| VibeVoice-7B | 7B, ~16GB | Full quality, long-form |
| Community fork | 7B | Same weights, community maintained |
