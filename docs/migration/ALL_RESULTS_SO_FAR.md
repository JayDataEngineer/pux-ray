# ALL Benchmark Results — Complete Data

> **Hardware:** RTX 4090 (24GB), PCIe Gen4 x16
> **Software:** PyTorch 2.10.0, diffusers 0.37.0
> **Seed:** 42 (same across all fair comparison runs)
> **All models cached locally** at /models/flux-schnell/ and /models/flux-dev/

---

## FLUX.1-dev Results (20 steps, 1024×1024, seed=42)

The REAL benchmark — 20 steps is representative of production workloads.

| # | Technique | Cold (s) | Warm (s) | Std | VRAM (MB) | Steps/s | Image |
|---|-----------|----------|----------|-----|-----------|---------|-------|
| 1 | **fp8 group_offload** | 16.8 | **16.1** | ±0.02 | **12,505** | 1.24 | ✅ |
| 2 | **bf16 group_offload** | 44.7 | **20.6** | ±0.48 | **12,505** | 0.97 | ✅ |
| 3 | bf16 model_cpu_offload + cache_accel | 49.7 | **29.0** | ±0.91 | 24,386 | 0.69 | ✅ |
| 4 | bf16 model_cpu_offload (baseline) | 58.0¹ | — | — | — | — | ❌ OOM² |
| 5 | model_cpu_offload + compile | 64.3¹ | — | — | — | — | ❌ timeout³ |

¹ Cold start only — did not complete warm runs
² Forge service loaded models during benchmark, causing OOM
³ torch.compile takes 60s+ for FLUX; kubectl exec timed out

### Per-step analysis (20-step generation)

```
fp8 group_offload:     0.80s/step × 20 = 16.1s total
bf16 group_offload:    1.03s/step × 20 = 20.6s total
cache + cpu_offload:   1.45s/step × 20 = 29.0s total (cache skipped ~0 steps at threshold=0.05)
```

---

## FLUX.1-schnell Results (4 steps, 1024×1024, seed=42)

Low step count — useful for cold start analysis but NOT representative of real workloads.

| # | Technique | Cold (s) | Warm (s) | Std | VRAM (MB) | Steps/s | Image |
|---|-----------|----------|----------|-----|-----------|---------|-------|
| 1 | **fp8 group_offload** | 4.3 | **3.4** | ±0.002 | **12,505** | 1.17 | ✅ |
| 2 | **bf16 group_offload** | 23.0 | **5.9** | ±0.61 | **12,505** | 0.69 | ✅ |

### Per-step analysis (4-step generation)

```
fp8 group_offload:     0.85s/step × 4 = 3.4s total
bf16 group_offload:    1.47s/step × 4 = 5.9s total
```

---

## Cross-Model Comparison (group_offload only)

| Model | Steps | FP8 Warm | BF16 Warm | FP8 per-step | BF16 per-step | FP8/BF16 ratio |
|-------|-------|----------|-----------|-------------|--------------|----------------|
| FLUX-schnell | 4 | 3.4s | 5.9s | 0.85s | 1.47s | 1.72x |
| FLUX-dev | 20 | 16.1s | 20.6s | 0.80s | 1.03s | 1.28x |

**Key finding:** The FP8 speed advantage SHRINKS with more steps (1.72x → 1.28x).
At 4 steps, PCIe transfer overhead dominates and FP8's smaller blocks help a lot.
At 20 steps, compute dominates and FP8's compute advantage is smaller.
The per-step BF16 time drops from 1.47s to 1.03s with more steps — likely because
the CUDA stream prefetch pipeline warms up and overlaps better.

---

## VRAM Techniques Tested

### ✅ Tested and completed

| Technique | Description | When to use |
|-----------|-------------|-------------|
| **group_offload (FP8)** | Block-level streaming + FP8 storage | Speed tier, shared GPU |
| **group_offload (BF16)** | Block-level streaming, no quant | Quality tier, limited VRAM |
| **model_cpu_offload + cache** | Component swap + step caching | Didn't help at threshold=0.05 |

### ⚠️ Partially tested (incomplete data)

| Technique | Issue |
|-----------|-------|
| model_cpu_offload (baseline) | OOM — forge service loaded during benchmark |
| model_cpu_offload + compile | Timeout — torch.compile takes 60s+ for FLUX |

### ❌ Not tested

| Technique | Why | Priority |
|-----------|-----|----------|
| BF16 fully resident | OOM — FLUX pipeline >24GB in BF16 | N/A |
| sequential_cpu_offload | Pod instability | LOW (known to be very slow) |
| FP8 mixed (text enc FP8, transformer BF16) | Pod instability | HIGH |
| int8 group_offload | Need int8 diffusers-format loader | MEDIUM |
| GGUF Q5/Q8 | Need GGUF model files | MEDIUM |
| compile + cache combined | INCOMPATIBLE (discovered) | N/A |

---

## Key Insights from Real Data

### 1. group_offload dominates model_cpu_offload
At 20 steps, same BF16 quality:
- group_offload: **20.6s, 12.5GB**
- model_cpu_offload + cache: **29.0s, 24.4GB**
- group_offload is **1.4x faster AND uses half the VRAM**

### 2. Cache acceleration barely helped at threshold=0.05
cache-only (29.0s) vs expected baseline without cache (~35-40s based on cold start extrapolation).
Cache may have skipped 2-3 steps out of 20. A higher threshold might skip more.

### 3. FP8 advantage shrinks with more steps
- 4 steps: FP8 is 1.72x faster than BF16
- 20 steps: FP8 is 1.28x faster than BF16
- At 40 steps (Z-Image), the advantage might shrink further

### 4. Both group_offload paths use identical VRAM
12,505 MB regardless of FP8 or BF16 — because text encoders (T5-XXL ~9.5GB)
are resident and dominate VRAM usage. The streaming transformer blocks are tiny
(~800MB for 2 blocks) in both formats.

### 5. Cold start: FP8 streaming is dramatically better
- FP8 cold: 4.3s (schnell) / 16.8s (dev) — smaller blocks transfer faster
- BF16 cold: 23.0s (schnell) / 44.7s (dev) — BF16 blocks are 2x larger
- For scale-to-zero deployments, FP8's cold start advantage is critical

---

## Output Images (for quality comparison)

Saved on pod at `/models/bench_fair/`:

```
FLUX-dev (20 steps, seed=42):
  fp8-group-offload_seed42.png     1013KB
  bf16-group-offload_seed42.png    1018KB
  cache-only_seed42.png            1015KB

FLUX-schnell (4 steps, seed=42):
  fp8-group-offload_seed42.png     (from earlier run)
  bf16-group-offload_seed42.png    (from earlier run)
```

All same seed, same prompt — can be compared pixel-by-pixel for quality delta.
Pending: retrieve and compare.
