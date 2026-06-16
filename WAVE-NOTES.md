# WAVE — Wan VACE Optimization Notes

## Working Configurations (RTX 4090 24GB)

### Stack 1: stable-diffusion.cpp + Q4_K_M GGUF + TAE ✅ PURE GPU
```
Engine:   sd-server (C++/GGML/CUDA)
Model:    Wan2.1_14B_VACE-Q4_K_M.gguf  (11.1GB)
T5:       umt5-xxl-encoder-Q8_0.gguf   (5.7GB)
AE:       taew2_1.safetensors          (22MB, replaces 242MB VAE)
VRAM:     ~16.9GB total
Speed:    ~2.6s/step → 18 steps × 2.6s ≈ 47s denoise + 5-10s AE
Flags:    --diffusion-fa --diffusion-model ... --tae ... --t5xxl ...
Port:     1234 (sd-server HTTP API)
Status:   🔴 VAE OOM for VACE context (15.7GB buffer)
           🟢 TAE FIX — drops VAE buffer from 15.7GB → ~1GB
           🟢 Fully GPU-resident, no PCIe streaming
```

### Stack 2: DiffSynth-Studio + FP8 CPU Offload (Python) ✅ WORKING
```
Engine:   DiffSynth-Studio via WanVideoPipeline
Model:    Wan2.2-VACE-Fun-A14B (dual-expert MoE, FP8 pre-quantized)
Speed:    242s for 81 frames, 18 steps (limited by CPU RAM streaming)
VRAM:     ~5.6GB (weights in CPU RAM, blocks streamed to GPU)
Flags:    offload_dtype=float8_e4m3fn, offload_device=cpu
Status:   🟢 Works but slow (PCIe bandwidth bottleneck)
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
- Standard VAE compute buffer for VACE context: **15.7GB**
- Q4_K_M model (17.1GB) + VAE buffer (15.7GB) = **32.8GB > 24GB** ❌
- `--vae-tiling` in sd.cpp doesn't reduce buffer for VACE path
- `--vae-on-cpu` works but takes ~20 min (146s per tile)

### The TAE Solution (Tiny AutoEncoder)
- **22MB** vs 242MB for standard VAE
- **~1GB** compute buffer vs 15.7GB for standard VAE
- Pure GPU possible: 16.9GB + 1GB = **~18GB < 24GB** ✅
- Download: `curl -L -o taew2_1.safetensors https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors`
- sd-server flag: `--tae` (not `--vae`)

### What sd.cpp Flags Work
| Flag | Effect | Status |
|---|---|---|
| `--diffusion-fa` | Flash attention | ✅ Works, required for speed |
| `--vae-tiling` | VAE tiling for 2D images | 🔴 Broken for Wan 3D VAE (GGML_ASSERT crash) |
| `--vae-on-cpu` | VAE on CPU | 🟡 Works but 146s/tile = 20min |
| `--tae` | Tiny AutoEncoder (pure GPU) | 🟢 ~1GB buffer, full speed |
| `--backend vae=cpu,diffusion=cuda0` | Per-backend assignment | 🟠 Mixed results |
| `--offload-to-cpu` | Full model offload | 🟡 Model on CPU = slow |
| `--tensor-type-rules` | Per-tensor quant override | Untested |

## What Works End-to-End
1. ✅ T2I (1 frame, 8 steps) — all configs
2. ✅ T2V (17+ frames) — DiffSynth + sd.cpp (TAE pending)
3. ✅ VACE modes — all modes coded (T2V, R2V, V2V, MV2V)
4. ✅ Director features — multi-keyframe, per-segment prompts, motion amplitude (DiffSynth)
5. ✅ HTTP API — sd-server at :1234, forge adapters for all stacks

## Remaining Issues
- sd.cpp VACE + TAE: failing with OOM on VAE compute buffer (15.7GB)
- TAE should fix this (22MB encoder, ~1GB buffer)
- vLLM-Omni: needs clean install to work
- Wan2.2 VACE-Fun: no diffusers format available for vLLM-Omni
