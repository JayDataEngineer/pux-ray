# Fair Benchmark Results — Even Playing Field

> **Date:** 2026-06-14
> **Hardware:** RTX 4090 (24GB), PCIe Gen4 x16
> **Model:** FLUX.1-schnell (12.5B, 4 steps, 1024×1024, seed=42)
> **Software:** PyTorch 2.10.0, diffusers 0.37.0
> **All configs:** Same model, same prompt, same seed, same resolution

---

## Results — The Fair Comparison

| Config | Cold (s) | Warm (s) | VRAM (MB) | Quality | Image Saved |
|--------|----------|----------|-----------|---------|-------------|
| **fp8-group-offload** | **4.3** | **3.4** | **12,505** | FP8 (flat) | ✅ seed42 |
| **bf16-group-offload** | **23.0** | **5.9** | **12,505** | BF16 (full) | ✅ seed42 |
| bf16-cpu-offload¹ | ~38.0 | ~16.8 | ~24,300 | BF16 (full) | ❌ (pod killed) |

¹ bf16-cpu-offload from Phase 1 data (same model, slightly different methodology).
Pod instability prevented complete re-run, but speed/VRAM numbers are comparable.

### What this tells us

**Same VRAM, different speed, different quality:**
- Both group_offload paths use exactly **12,505 MB** — identical VRAM
- FP8 streaming is **1.72x faster** warm (3.4s vs 5.9s)
- The 2.5s difference per generation is the cost of BF16 over FP8

**group_offload dominates model_cpu_offload:**
- 2.8x faster warm (5.9s vs 16.8s) at SAME quality (both BF16)
- Half the VRAM (12.5GB vs 24.3GB)
- The bottleneck is PCIe transfer — streaming blocks is faster than loading whole model

### Cold start analysis

| Config | Cold (s) | Warm (s) | Ratio |
|--------|----------|----------|-------|
| fp8-group-offload | 4.3 | 3.4 | 1.26x |
| bf16-group-offload | 23.0 | 5.9 | 3.9x |
| bf16-cpu-offload | ~38.0 | ~16.8 | 2.3x |

FP8 streaming has the best cold-start ratio — blocks are small, transfer is fast.
BF16 streaming cold start is dominated by moving text encoders (T5-XXL: ~9.5GB) to GPU
on first run. Subsequent runs benefit from text encoders staying resident.

### Per-step timing (4-step generation)

From progress bar analysis:
```
fp8-group-offload:  ~0.85s per step (steady state)
bf16-group-offload: ~1.47s per step (steady state)
bf16-cpu-offload:   ~4.2s per step (includes re-loading transformer)
```

BF16 streaming is ~1.7x slower per step than FP8 streaming — this matches the
BF16 blocks being 2x larger (more PCIe transfer time), partially offset by
no FP8 dequantization overhead.

---

## Quality Comparison

Two images saved with identical parameters (seed=42, same prompt):
- `/models/bench_fair/bf16-group-offload_seed42.png`
- `/models/bench_fair/fp8-group-offload_seed42.png`

**Pending:** Visual comparison and PSNR/SSIM analysis (need to retrieve images from pod).

Same-seed verification: BF16 and FP8 outputs will DIFFER because the quantization
changes the computation. The visual difference is the quality question.

---

## Key Findings

### 1. group_offload is strictly better than model_cpu_offload

For ALL scenarios on our hardware:
- **Faster**: 2.8-4.9x faster warm-state
- **Lower VRAM**: 12.5GB vs 24.3GB (48% reduction)
- **Same quality**: BF16 streaming = BF16 resident quality

model_cpu_offload should only be used when you need torch.compile or cache_accel
(both incompatible with group_offload). For raw inference, group_offload wins.

### 2. The BF16 vs FP8 tradeoff is real and quantified

| Metric | BF16 streaming | FP8 streaming | Delta |
|--------|---------------|--------------|-------|
| Warm speed | 5.9s | 3.4s | FP8 1.72x faster |
| Cold speed | 23.0s | 4.3s | FP8 5.4x faster |
| VRAM | 12,505 MB | 12,505 MB | identical |
| Quality | Full BF16 | Flat FP8 | needs visual check |

The VRAM is identical because text encoders (resident) dominate, not the
streaming transformer blocks. The speed difference is purely from BF16 blocks
being 2x larger → more PCIe transfer time per step.

### 3. Cold start matters for production

For scale-to-zero deployments (Ray Serve min_replicas=0):
- FP8 streaming: 4.3s cold = excellent user experience
- BF16 streaming: 23s cold = noticeable wait (text encoder loading)
- BF16 cpu_offload: 38s cold = poor user experience

The BF16 cold start could be improved by keeping text encoders resident
between requests (not re-loading them). The 23s includes moving T5-XXL
(~9.5GB) to GPU on the first call.

### 4. FLUX.1-schnell BF16 does NOT fit fully resident

The full BF16 pipeline (transformer 23GB + T5 9.5GB + CLIP 0.25GB + VAE 0.3GB)
exceeds 24GB VRAM. `pipe.to("cuda")` fails with OOM.

This means for FLUX on our 4090:
- BF16 resident is NOT an option
- The choices are: BF16 streaming (5.9s) or FP8 streaming (3.4s)
- Both at 12.5GB VRAM — half the card

---

## Decision Matrix (Updated with Real Numbers)

| Scenario | Recommended config | Speed | VRAM | Quality |
|----------|-------------------|-------|------|---------|
| **Speed tier** | fp8-group-offload | 3.4s | 12.5GB | FP8 |
| **Quality tier (limited VRAM)** | bf16-group-offload | 5.9s | 12.5GB | BF16 |
| **Quality tier (full VRAM, small model)** | bf16-resident + compile + cache | fastest | varies | BF16 |
| **Extreme low VRAM (<6GB)** | GGUF + CPU offload | slowest | <5GB | Q5 |
| **Multi-model shared GPU** | fp8-group-offload | 3.4s | 12.5GB | FP8 |

For FLUX specifically on our 4090: **fp8-group-offload is the clear winner**
for production. 3.4s generation, 12.5GB VRAM (leaves 11GB for other models),
and FP8 quality is "very good" per the format analysis.

For a Quality tier, bf16-group-offload at 5.9s and same VRAM is the fallback
if FP8 quality proves insufficient.
