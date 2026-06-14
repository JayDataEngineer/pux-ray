# LTX-Video Hyper-Optimization Research

> **Source:** Deep research report on high-performance LTX-Video inference
> **Date:** 2026-06-14
> **Status:** REFERENCE — implementation guide for Phase 3 LTX sequencer

---

## Three Frameworks Analyzed

### 1. NVIDIA TensorRT-LLM VisualGen
- Production DiT inference engine (separate from autoregressive LLMs)
- `BasePipeline` class manages unified denoising loop
- Quantization: FP8/FP4 via NVIDIA ModelOpt (dynamic or static)
- Attention backends: QK16PV8 (BF16 Q/K + FP8 V), SAGE (SageAttention)
- TeaCache accelerator: L1 distance threshold monitoring for step skipping
- Multi-GPU via DiffusionExecutor + ZeroMQ IPC

### 2. HAO AI Lab FastVideo / Dreamverse
- UC Berkeley optimized training + inference framework
- Sequence-parallel distributed execution
- Video Sparse Attention (VSA) + SageAttention backends
- DMD2 step-distillation: reduces to 4-8 step regime
- Dreamverse: real-time streaming via fMP4 over WebSocket
- Custom FFmpeg binary compiled from source

### 3. Lightricks Native ltx-pipelines
- Official production library for LTX-2/LTX-2.3
- Modular components from ltx-core (schedulers, guiders, noisers)

**Pipeline variants:**
| Pipeline | Description |
|----------|-------------|
| TI2VidTwoStagesPipeline | Standard high-quality (half-res Stage 1 → 2× upscale Stage 2) |
| TI2VidTwoStagesHQPipeline | Second-order res_2s solver (fewer steps, same quality) |
| TI2VidOneStagePipeline | Single-stage, full-res (rapid prototyping) |
| DistilledPipeline | 8-step fixed path (8 Stage 1 + 4 Stage 2), no CFG |
| ICLoraPipeline | Video-to-video with IC-LoRA control (depth, pose, edge) |

**Native FP8:** `--quantization fp8-scaled-mm` via Tensor Cores
**torch.compile:** Direct execution wrappers included

---

## Latent Injection Paradigms

### Replacing Latents (Direct Overwrite)
```
Frame:   [0]      [1]      [2]      [3]      [4]
Latent:  [Enc(I)] [Noise]  [Noise]  [Noise]  [Enc(I)]
```
- Physically overwrites latent at frame index with VAE-encoded image
- Guarantees strict pixel-level alignment
- **Breaks ODE solver** — causes flickering, scene jumps
- Use for: deliberate hard scene cuts

### Guiding Latents (Continuous Additive)
```
Frame:   [0]      [1]      [2]      [3]      [4]
Latent:  [Noise]  [Noise]  [Noise]  [Noise]  [Noise]
Signal:  [+G_0]   [+G_1]   [+G_2]   [+G_3]   [+G_4]
```
- Additive spatial-temporal conditioning on noise latents
- Gaussian decay around keyframe: `attenuation = exp(-0.5 * distance²) * strength`
- Smooth transitions, preserves solver continuity
- Use for: in-betweening, smooth transitions

### IC-LoRA Masking
- Parse `reference_downscale_factor` from LoRA metadata
- Load spatial mask `(B, 1, F, H, W)` → grayscale → normalize [0,1]
- Downsample to latent space with **causal temporal alignment** (first frame special)
- Multiply by `conditioning_attention_strength` (γ ∈ [0,1])
- Inject into self-attention query-key projections

---

## Compiler Recompilation Storm Solution

### The Problem
torch.compile traces the DiT graph. Any shape change (frame count, keyframe
index, mask dimensions) invalidates the compiled graph → recompilation.
Large DiT compilation takes minutes. Dynamic timelines = constant recompilation.

### Solution: Three-Part Architecture

#### Part 1: Static Latent Shape Padding
- Fix temporal dimension to profiles: {9, 17, 33, 65, 97} frames (LTX VAE: 8n+1)
- Fix spatial dimensions to standard aspect (e.g., 768×512)
- Pad shorter sequences with zeros to nearest profile
- 3D attention mask: valid frames attend normally, padded = -∞

```python
# Attention mask construction
M_attn(i, j) = 0      if i, j < F_logical (valid)
M_attn(i, j) = -inf   otherwise (padded)
```

#### Part 2: Piecewise CUDA Graphs (PCG)
- Divide DiT into individual transformer block subgraphs
- Each block wrapped in `CUDAPiecewiseBackend`
- Dynamic operations (masking, slicing) stay eager between compiled blocks
- Eliminates launch overhead for heavy layers while supporting runtime flexibility

#### Part 3: Discrete Shape Profiling + Memory Pooling
- Pre-capture CUDA graphs for profiles: P = {9, 17, 33, 65, 97}
- Binary search maps logical frames → smallest compatible profile
- Capture in reverse order (largest → smallest) for memory pool reuse
- Global memory pool shared across all profiles
- Keeps total VRAM within 24GB consumer limit

---

## HyperOptimizedLTXSequencer Blueprint

The target implementation. Four components:

### 1. StaticLatentPadder
- Maps logical frames → nearest profile via binary search
- Pads latents to profile size
- Generates attention mask for valid vs padded frames

### 2. AdvancedLatentInjector
- `apply_replacements()`: Direct latent overwrite at keyframe indices
- `generate_guiding_signals()`: Gaussian-decay additive signals
- Both operate BEFORE entering compiled DiT layers

### 3. PiecewiseDiTExecutor
- `warmup_and_capture()`: Pre-capture CUDA graphs for all profiles
- Captures largest profile first (memory pool allocation)
- Replays pre-captured graph at runtime (zero recompilation)

### 4. Main Sequencer Flow
```
1. Inject replacements (hard keyframes)
2. Generate guiding signals (soft keyframes)
3. Map to nearest temporal profile
4. Pad latents + generate attention mask
5. Apply IC-LoRA mask if present
6. Copy to static graph buffers
7. Replay pre-captured CUDA graph
8. Slice output to logical frame count
```

---

## Deployment Strategy for Consumer GPUs (RTX 4090/5090)

1. **Dual-Stage Execution:** Stage 1 at half-res for motion composition,
   Stage 2 at full-res with distilled LoRA for detail recovery
2. **Discrete Profiles:** Lock to {9, 17, 33, 65, 97} frame counts
3. **Guiding Latents:** Use soft additive guidance for transitions
   (replacing latents only for hard scene cuts)
4. **FP8 Quantization:** `fp8-scaled-mm` for Tensor Core acceleration
5. **Memory Pooling:** Single global pool, reverse-order graph capture

---

## Comparison Table

| Feature | Replacing | Guiding | IC-LoRA Mask | Static Padder |
|---------|-----------|---------|-------------|---------------|
| Concept | Direct overwrite | Additive signal | Cross-attn weighting | 3D mask over padding |
| Solver impact | Breaks trajectory | Preserves | Preserves | Zero (masked) |
| Use case | Hard cuts | Smooth transitions | Region control | Shape stability |
| Memory | Low | Moderate | High (5D mask) | Optimized via reuse |

---

**Implementation target:** `services/native/ltx_sequencer.py`
**Dependencies:** ltx-pipelines (Lightricks), torch.compile, CUDA Graphs
**Test model:** LTX-Video 2B (cached at /models/ltx-video/)
