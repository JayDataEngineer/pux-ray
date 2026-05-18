# faster_qwen3_tts — Original Workflow

**Source**: Wan2GP custom handler (based on Qwen3-TTS from HuggingFace)
**Type**: GPU-accelerated text-to-speech with CUDA graphs
**Architecture**: Qwen3-TTS with custom CUDA graph capture for 6-10x speedup

## Inference Pipeline (3 Stages)

### Stage 1: Prefill
```
text prompt → tokenize → standard HuggingFace forward pass
→ processes text embeddings, builds initial KV cache (dynamic)
→ outputs: past_key_values, past_hidden, generation_step, initial logits
→ NOT CUDA-graphed (variable-length input)
```

### Stage 2: Predictor Loop (CUDA-graphed)
```
past_hidden + first_codebook_embedding → 15-step autoregressive code prediction
→ each step: embed previous token → transformer forward → sample next token
→ StaticCache for KV management (fixed-size, pre-allocated)
→ captured as single CUDA graph for deterministic shapes
```

### Stage 3: Decode Loop (CUDA-graphed)
```
codebook embeddings → single-token decode through talker's 28 transformer layers
→ StaticCache for KV management
→ captured as one CUDA graph per position
→ hidden states → codec_head → audio tokens
```

## Components

| Module | Role | Size | Notes |
|--------|------|------|-------|
| talker | Main TTS transformer (28 layers) | ~large | Backbone |
| code_predictor | 5-layer transformer for codebook prediction | ~medium | 15-step AR loop |
| speech_tokenizer | VAE decoder for audio reconstruction | ~small | Tokens → waveform |
| text_projection | Text embedding projection | ~tiny | Bridges text to model space |

## Key Characteristics

- **Three separate execution modes**: dynamic prefill, graphed predictor, graphed decoder
- **StaticCache** for CUDA graph compatibility — pre-allocated fixed-size KV cache
- **PredictorGraph**: captures full 15-step predictor loop as one CUDA graph
- **TalkerGraph**: captures single-token decode step as CUDA graph
- **6-10x speedup** over baseline Qwen3-TTS without graphs
- **Three voice modes**: custom_voice, voice_clone, voice_design

## How It Uses Wan2GP

faster_qwen3_tts is the **reference implementation** for Native Custom integration:
- Uses PyTorch native `torch.cuda.CUDAGraph` (not Wan2GP's cudagraph_kit directly)
- Uses StaticCache (same concept as nanovllm's paged KV cache)
- Uses mmgp for weight management via standard pipe dict
- Handler conforms to Wan2GP family_handler pattern
- Gets Wan2GP's mmgp + quantization + attention backends

## Relationship to Wan2GP Shared Layer

- **cudagraph_kit.py** provides reusable CUDA graph utilities — faster_qwen3_tts implements its own variant
- **nanovllm** provides paged attention + scheduling — faster_qwen3_tts uses StaticCache instead (simpler)
- Both achieve similar results through different implementations
- The handler is specialized for TTS decode patterns (predictor + talker dual-loop)
