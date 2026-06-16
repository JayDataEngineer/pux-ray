# WAVE — Wan VACE Optimization Notes

## Working Configurations (RTX 4090 24GB)

### Stack 1: stable-diffusion.cpp + Q4_K_M GGUF + VAE ✅ PURE GPU (lower res)
```
Engine:   sd-server (C++/GGML/CUDA)
Model:    Wan2.1_14B_VACE-Q4_K_M.gguf  (11.1GB diffusion + 5.7GB T5 + 0.2GB VAE = 17.1GB)
Flags:    --diffusion-fa --diffusion-model ... --vae ... --t5xxl ...
Port:     1234 (sd-server HTTP API)
VRAM:     20.4GB at 640×368 (fits in 24GB ✅)
          32.8GB at 832×480 (15.7GB VAE buffer overflows ❌)
Speed:    ~2.6s/step → 18 steps × 2.6s ≈ 47s denoise
```

### Stack 2: DiffSynth-Studio + FP8 CPU Offload (Python) ✅ WORKS
```
Engine:   DiffSynth-Studio via WanVideoPipeline
Model:    Wan2.2-VACE-Fun-A14B (dual-expert MoE, FP8 pre-quantized)
Speed:    242s for 81 frames, 18 steps (PCIe bandwidth bottleneck)
VRAM:     ~5.6GB (weights in CPU RAM, blocks streamed to GPU)
```

### Stack 3: vLLM-Omni (planned, dependency issues)
```
Engine:   vLLM-Omni 0.22.0 + vLLM 0.23.0 (C extension ABI mismatch)
Model:    Wan-AI/Wan2.1-VACE-14B-diffusers (70GB, downloaded)
Status:   🔴 vLLM C extension broken (wrong PyTorch version)
           🔴 NGC container vLLM dev version incompatible with Omni
```

## Pure GPU VACE Problem (RTX 4090, 24GB)

### The VAE Bottleneck
- Standard VAE compute buffer for VACE context encode: **15.7GB** at 832×480
- Q4_K_M model (17.1GB) + VAE buffer (15.7GB) = **32.8GB > 24GB** ❌
- Even with `--vae-tiling`, sd.cpp VACE path allocates full 15.7GB buffer
- `--vae-on-cpu` works but slow (146s/tile, ~20min total)

### Lower Resolution = Pure GPU ✅
The VAE compute buffer scales with spatial resolution:
| Resolution | Latents | VAE Buffer | Total VRAM | Fits 24GB? |
|------------|---------|------------|------------|------------|
| 832×480 | 104×60 | 15.7GB | 32.8GB | ❌ |
| 640×368 | 80×46 | ~3.3GB | 20.4GB | ✅ |
| 576×336 | 72×42 | ~2.4GB | 19.5GB | ✅ |

### TAE (Tiny AutoEncoder) — DOES NOT WORK FOR VACE
- `--tae` flag in sd.cpp loads 22MB Tiny AutoEncoder
- TAE reduces buffer to ~1GB but **doesn't support VACE context** (96 channels)
- Standard VAE is required for VACE's multi-channel VCU encoding
- TAE works only for standard T2V/I2V (16-channel latents)

## Key Metrics (81 frames, 18 steps, 832×480)

| Config | Total | Denoise | Per-step | VRAM | Pure GPU? |
|--------|-------|---------|----------|------|-----------|
| DiffSynth FP8 CPU offload | 242.0s | 241.5s | 13.4s | 5.6GB | ❌ |
| sd.cpp Q4_K_M GPU + VAE CPU | ~400s | 46s | 2.6s | 17.1GB | ❌ (VAE CPU) |
| sd.cpp Q4_K_M GPU at 640×368 | ~90s | ~47s | 2.6s | 20.4GB | ✅ |

## What sd.cpp Flags Work
| Flag | Effect | Status |
|---|---|---|
| `--diffusion-fa` | Flash attention | ✅ Required for speed |
| `--vae-tiling` | VAE spatial tiling | 🔴 Broken for Wan 3D VAE |
| `--vae-on-cpu` | VAE on CPU (slow) | 🟡 146s/tile |
| `--tae` | Tiny AutoEncoder (22MB) | 🔴 No VACE support |
| `--backend vae=cpu,diffusion=cuda0` | Per-backend | 🔴 VAE tensor name mismatch |
| `--offload-to-cpu` | Full model on CPU | 🟠 Too slow |

## Director Capabilities (DiffSynth-Studio)
All WhatDreamsCost LTX Director node features implemented:
- ✅ Multi-keyframe injection (replace/guide/fade modes)
- ✅ Per-segment prompts (Prompt Relay)
- ✅ Motion amplitude control
- ✅ Continuity handoff between segments
- ✅ Global + local prompts
- ✅ Video stitching

## Remaining Issues
- sd.cpp: job queue intermittent (jobs qued but not processed)
- Wan2.2 VACE-Fun: no diffusers format for vLLM-Omni
- vLLM-Omni: version mismatch between vLLM, Omni, PyTorch, and CUDA
- Optimal pure GPU speed (35-45s) requires: either 32GB+ VRAM, or fix VAE tiling for VACE context
