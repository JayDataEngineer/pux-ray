# IndexTTS v2 — Original Workflow

**Source**: https://github.com/IndexTeam/IndexTTS (via Wan2GP fork DeepBeepMeep/TTS)
**Type**: Text-to-speech with voice cloning and emotion control
**Architecture**: Multi-stage GPT + flow matching + BigVGAN vocoder

## Inference Pipeline (5 Stages)

### Stage 1: Speaker & Emotion Feature Extraction
```
reference audio → w2v-bert-2.0 (semantic model) → speaker embeddings
reference audio → CAMPPlus → speaker style features
text/optional audio → QwenEmotion → emotion conditioning
```

### Stage 2: Text Processing
```
text → BPE tokenizer → normalize → split into segments (auto or manual)
→ parse emotion tags and speaker labels
```

### Stage 3: Latent Generation (GPT)
```
text tokens + speaker embeddings + emotion conditioning
→ UnifiedVoice (GPT transformer) → acoustic tokens
→ cross-attention between speaker, emotion, text conditions
→ speech conditioning latent representations
```

### Stage 4: Acoustic Modeling (S2Mel + CFM)
```
acoustic tokens → S2Mel model
  → GPT layer for latent processing
  → length regulator
  → Conditional Flow Matching (CFM) diffusion
  → mel-spectrogram features
```

### Stage 5: Vocoder
```
mel-spectrogram → BigVGAN → waveform at 22kHz
```

## Components

| Module | Role | Size | Notes |
|--------|------|------|-------|
| UnifiedVoice | Main GPT transformer for token generation | ~large | Core generation model |
| S2Mel | Acoustic model with CFM diffusion | ~medium | Tokens → mel-spectrogram |
| BigVGAN | Vocoder (mel → waveform) | ~medium | 22kHz output |
| w2v-bert-2.0 | Semantic audio encoder | ~large | Speaker embedding extraction |
| Semantic Codec | Vector quantization for discrete tokens | ~small | Audio tokenization |
| CAMPPlus | Speaker style extraction | ~small | Voice characteristics |
| QwenEmotion | Emotion detection from text | ~medium | Emotion conditioning |

## Key Characteristics

- **Voice cloning** from single reference audio
- **Emotion control** — via text tags or reference audio
- **Multi-speaker dialogue** — generate speech for multiple speakers
- **Conditional Flow Matching** — diffusion-based mel generation (not standard diffusion)
- **22kHz output** — lower than some models (24kHz) but good quality
- **Multi-stage pipeline** — GPT generation, then CFM diffusion, then vocoder

## Capabilities

- Voice cloning from 1 reference audio
- Voice + emotion from 2 reference audios
- Dialogue generation between speakers
- Emotion control via text prompts or audio conditioning
