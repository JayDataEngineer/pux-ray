# MOSS TTS — GGUF Migration Guide

## Background

The MOSS-TTS project (OpenMOSS-Team) has deprecated the original PyTorch
(safetensors) model format in favor of **GGUF** for the TTS backbone, with
**ONNX Runtime** (or TensorRT) for the audio tokenizer. This is a deliberate
architecture choice: the Qwen3 backbone runs through llama.cpp (GPU/CPU),
while the audio codec uses ONNX for maximum portability.

**Our old model files at `/mnt/data/models/audio/moss-tts/` are obsoleted**
by this migration. They are safetensors format PyTorch checkpoints that do
not match the production C++ inference pipeline. The only production-safe
path is the GGUF+ONNX route documented below.

## Why GGUF?

| Factor | Safetensors (old) | GGUF (new) |
|--------|-------------------|------------|
| Inference engine | Python + PyTorch | llama.cpp C++ |
| VRAM efficiency | Full model in VRAM | Quantized (Q4_K_M = ~4.5GB for 8B) |
| Startup time | ~12s model load | ~2s (mmap'd weights) |
| Torch dependency | Required | Optional (torch-free path available) |
| Production readiness | Prototype | Mature (llama.cpp ecosystem) |
| Streaming | Limited | First-class via llama.cpp |

## Model Inventory

### Current Disk State (`/mnt/data/models/audio/`)

| Directory | Format | Status | Action |
|-----------|--------|--------|--------|
| `moss-tts/` | safetensors (4 shards) | ❌ Obsolete | Replace with GGUF |
| `moss-tts-local-transformer/` | safetensors (2 shards) | ❌ Obsolete | Replace with GGUF |
| `moss-tts-nano/` | pytorch_model.bin | ❌ Obsolete | Replace with GGUF |
| `moss-tts-realtime/` | safetensors | ❌ Obsolete | Replace with GGUF |
| `moss-ttsd/` | safetensors (4 shards) | ❌ Obsolete | Replace with GGUF |
| `moss-voicegenerator/` | safetensors | ❌ Obsolete | Replace with GGUF |
| `moss-soundeffect-v2/` | diffusers pipeline | ✅ Working | Keep (not affected) |
| `moss-soundeffect/` | safetensors (broken) | ❌ Obsolete | Remove |

## Migration Steps

### Step 1: Download GGUF Backbone

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF \
  --local-dir /mnt/data/models/audio/moss-gguf/MOSS-TTS-GGUF
```

Expected contents:
```
MOSS-TTS-GGUF/
├── MOSS_TTS_Q4_K_M.gguf        # Q4_K_M quantized backbone (~4.5 GB)
├── MOSS_TTS_Q8_0.gguf          # Q8_0 quantized backbone (~8 GB, optional)
├── embeddings/                   # 33 embedding .npy files
│   ├── 0.npy, 1.npy, ...
│   └── ...
├── lm_heads/                     # 33 LM head .npy files
│   ├── 0.npy, 1.npy, ...
│   └── ...
└── tokenizer/                    # BPE tokenizer files
    ├── merges.txt
    └── vocab.json
```

### Step 2: Download ONNX Audio Tokenizer

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX \
  --local-dir /mnt/data/models/audio/moss-gguf/MOSS-Audio-Tokenizer-ONNX
```

Expected contents:
```
MOSS-Audio-Tokenizer-ONNX/
├── encoder.onnx                  # Audio encoder (~200 MB)
├── decoder.onnx                  # Audio decoder (~200 MB)
└── config.json
```

### Step 3: Build llama.cpp + C Bridge

The vendor code at `vendor/moss-tts-delay/llama_cpp/` contains the bridge.
See `vendor/moss-tts-delay/llama_cpp/README.md` for build instructions:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp && cmake -B build && cmake --build build --config Release -j

# Build the C bridge shared library
cd /path/to/vendor/moss-tts-delay/llama_cpp
bash build_bridge.sh /tmp/llama.cpp
```

### Step 4: Update MOSS Docker Image

The production MOSS Docker (`forge-reg.local:30500/tech-noir/moss:latest`)
needs to be rebuilt to include:

1. llama.cpp shared library
2. ONNX Runtime Python package
3. The `moss_tts_delay/llama_cpp/` module
4. Volume mounts for GGUF weights + ONNX models

See `infra/docker/Dockerfile.moss` (needs update).

### Step 5: Update Launcher

The MOSS server at `services/audio/moss_server.py` needs a new model
backend that uses the llama.cpp pipeline instead of the diffusers pipeline
for TTS models. SoundEffect-v2 (diffusers pipeline) remains unchanged.

## Current Status

| Model | Backend | Status | Owner |
|-------|---------|--------|-------|
| moss-soundeffect-v2 | Diffusers | ✅ Production | MOSS Pool |
| moss-tts (GGUF) | llama.cpp | ⏳ Needs build | MOSS Pool |
| moss-tts-realtime | PyTorch | ❌ Obsolete | — |
| moss-tts-local-transformer | PyTorch | ❌ Obsolete | — |
| moss-voicegenerator | PyTorch | ❌ Obsolete | — |

## Fallback Strategy

Until the GGUF path is production-ready:

1. **SoundEffect-v2** continues to work via the existing diffusers pipeline
   (port 8050, `inference-moss` container)
2. **TTS** requests should be routed to the Kokoro TTS service (diffusers,
   Tier D, port 8095) as a temporary fallback
3. **MOSS TTS GGUF** goes live once the Docker image is rebuilt

## Reference

- GitHub: https://github.com/OpenMOSS/MOSS-TTS
- GGUF weights: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF
- ONNX tokenizer: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX
- llama.cpp backend docs: `vendor/moss-tts-delay/llama_cpp/README.md`
- First-class e2e guide: https://github.com/OpenMOSS/llama.cpp/blob/moss-tts-firstclass/docs/moss-tts-firstclass-e2e.md
