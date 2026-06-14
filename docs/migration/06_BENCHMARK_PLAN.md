# Fair Benchmark Plan — Even Playing Field Methodology

> **Status:** PLANNING — methodology defined, ready to execute
> **Date:** 2026-06-14
> **Principle:** One variable at a time. Same seed, same prompt, same model.

---

## Table of Contents

1. [What Phase 1 Got Wrong](#1-what-phase-1-got-wrong)
2. [The BF16 Streaming Insight](#2-bf16-streaming-insight)
3. [Test Matrix](#3-test-matrix)
4. [Controls & Methodology](#4-controls--methodology)
5. [Metrics to Capture](#5-metrics-to-capture)
6. [Quality Comparison Protocol](#6-quality-comparison-protocol)
7. [mmGP Baseline Comparison](#7-mmgp-baseline-comparison)
8. [Execution Plan](#8-execution-plan)

---

## 1. What Phase 1 Got Wrong

The Phase 1 benchmark (04_PHASE1_RESULTS.md) had these flaws:

| Flaw | Impact |
|------|--------|
| Only group_offload had FP8 | Conflated quantization with streaming — 2 variables changed |
| No output images saved | Can't compare quality |
| No mmGP comparison | Tested native paths against each other, not against the system we're replacing |
| No cold-start vs steady-state separation | Can't tell loading overhead from inference speed |
| No per-phase breakdown | Don't know if time is text encoding, denoising, or VAE decode |
| No BF16 group_offload test | The most important Quality-tier path wasn't measured |
| FLUX-schnell only (4 steps) | Cache acceleration and compile can't show their value at 4 steps |

This document defines the methodology to fix all of these.

---

## 2. BF16 Streaming Insight

**The key discovery from the tier architecture discussion:**

group_offload enables BF16 quality at low VRAM by streaming blocks instead
of quantizing weights. This means the Quality tier doesn't need to sacrifice
precision even on shared or limited GPUs.

```
Traditional approach (quantize to fit):
  20GB BF16 transformer → 5GB Q5 GGUF (permanent quality loss)
  Fits in VRAM because weights are smaller.

group_offload approach (stream to fit):  
  20GB BF16 transformer → stays 20GB BF16 (zero quality loss)
  Fits in VRAM because only 1-2 blocks (~800MB) are resident at a time.
```

### VRAM math (Qwen-Image-Edit 20B on 8GB)

```
VLM (Qwen2.5-VL)          FP8 resident     ~3.5 GB    (quantized, invisible impact)
Transformer (20B DiT)     BF16 streaming   ~1.0 GB    (2 blocks at a time via group_offload)
VAE                       BF16 resident    ~0.25 GB   (always BF16)
Activations (1024×1024)                    ~2.0 GB    (denoising intermediates)
CUDA context                              ~1.0 GB
                                          ────────
Total                                     ~7.75 GB   ← fits in 8GB with BF16 transformer
```

### VRAM estimates per model (BF16 streaming)

| Model | Transformer BF16 | Streaming resident | Text enc FP8 | VAE | Total | Fits 8GB? |
|-------|-----------------|-------------------|-------------|-----|-------|-----------|
| Anima | 3.9GB | ~0.5GB | ~0.6GB | 0.25GB | ~3.4GB | ✅ easily |
| Z-Image | ~12GB | ~1GB | ~2GB | 0.3GB | ~5.3GB | ✅ |
| Qwen-Image-Edit | ~20GB | ~1GB | ~3.5GB | 0.25GB | ~5.8GB | ✅ |
| FLUX.1-dev | ~23GB | ~1.5GB | ~5GB | 0.3GB | ~8.8GB | ⚠️ tight |
| Wan 14B | ~28GB | ~2GB | ~5GB | 0.5GB | ~8.5GB | ⚠️ tight |

### Tradeoff: BF16 streaming vs Q5 GGUF resident

```
Q5 GGUF resident:
  Transfer: 0 (weights already in VRAM at Q5 size, ~5GB for 20B)
  Compute:  software dequant → BF16 → matmul (per block, per step)
  Quality:  ★★★☆☆ (5-bit, block-quantized)

BF16 group_offload:
  Transfer: 50 blocks × 400MB × PCIe Gen4 (per step, overlapped with compute)
  Compute:  direct BF16 matmul (no dequant)
  Quality:  ★★★★★ (full precision)
```

**Which is faster is UNKNOWN — this is the #1 thing to benchmark.**

---

## 3. Test Matrix

### Group A: Isolate optimization path (all BF16, same model)

Tests the effect of the OPTIMIZATION TECHNIQUE at constant quality.
Only works if the model fits BF16 resident — use Anima (3.9GB).

| ID | Configuration | What it tests |
|----|--------------|---------------|
| A1 | BF16 fully resident (`pipe.to("cuda")`) | Gold standard — fastest possible |
| A2 | BF16 model_cpu_offload | Component-level swapping |
| A3 | BF16 group_offload (use_stream=True) | Block-level streaming |
| A4 | BF16 group_offload + compile_repeated_blocks | Compile benefit (if compatible) |
| A5 | BF16 model_cpu_offload + cache_accel | Cache benefit at 30 steps |
| A6 | BF16 model_cpu_offload + compile | Compile benefit at 30 steps |

**Expected outcome:** A1 fastest, A3 slower but same quality, A5/A6 potentially
faster than A1 if cache skips steps or compile fuses kernels.

### Group B: Isolate format (all group_offload, same model)

Tests the effect of QUANTIZATION at constant optimization path.

| ID | Configuration | What it tests |
|----|--------------|---------------|
| B1 | BF16 group_offload | Full quality baseline |
| B2 | FP8 group_offload (layerwise_casting) | Flat FP8 quality + speed |
| B3 | int8 group_offload (if loadable) | int8 quality + speed |

**Expected outcome:** B2 faster than B1 (smaller blocks = faster PCIe),
B1 better quality than B2. B3 TBD.

### Group C: Cross-comparison (the product question)

The configurations we'd actually deploy. Different model if needed.

| ID | Configuration | Product tier |
|----|--------------|-------------|
| C1 | BF16 resident + compile + cache | Quality (full VRAM) |
| C2 | BF16 group_offload | Quality (limited VRAM) |
| C3 | FP8 resident | Speed (full VRAM) |
| C4 | FP8 group_offload | Speed (limited VRAM) |
| C5 | mmGP path (current production) | Baseline to beat |

**Expected outcome:** C1 fastest+best quality. C5 is the number to beat.
C2 should match C1 quality but slower. C3/C4 faster but lower quality.

### Group D: Step count sensitivity

Cache acceleration and compile need many steps to amortize. Test at different step counts.

| Steps | Model | Cache impact expected | Compile impact expected |
|-------|-------|-----------------------|------------------------|
| 4 | FLUX-schnell | Minimal | Minimal |
| 20 | FLUX-dev | Significant | Moderate |
| 30 | Anima | Significant | Moderate |

---

## 4. Controls & Methodology

### Fixed across all tests

```
Seed:         42 (same noise → reproducible)
Prompt:       [defined per model, same across all configs of that model]
Resolution:   1024×1024 (images) or model default
Guidance:     Model default (4.0 for Anima, 0.0 for schnell, 3.5 for dev)
Warmup:       3 runs (discarded) — handles compilation, JIT, cache warming
Timed:        5 runs (reported)
Cooldown:     5 seconds between runs (thermal throttling prevention)
Cleanup:      gc.collect() + torch.cuda.empty_cache() between runs
```

### Separated measurements

For EACH configuration, measure these phases SEPARATELY:

```
1. MODEL_LOAD:    from_pretrained() time (disk → CPU RAM)
2. MODEL_TO_GPU:  pipe.to("cuda") or first group_offload setup time
3. COMPILE_TIME:  torch.compile / compile_repeated_blocks time (if applicable)
4. COLD_START:    First inference call (includes any remaining initialization)
5. WARM_STATE:    Steady-state inference (3rd-7th calls, averaged)
6. PER_STEP:      Average time per denoising step (extracted from progress bar)
```

The number that matters for production:
- **Interactive latency** = COLD_START (first request after scale-from-zero)
- **Throughput** = WARM_STATE (repeated requests)

### VRAM measurement protocol

```python
# Before each timed run:
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()

# After each timed run:
metrics = {
    "peak_allocated": torch.cuda.max_memory_allocated() / 1e6,   # MB
    "peak_reserved": torch.cuda.max_memory_reserved() / 1e6,      # MB  
    "system_vram": nvidia_smi_query(),                             # MB
}
```

Track ALL THREE. Peak allocated shows model+activations. System shows total
including CUDA context and other processes. Reserved shows allocator overhead.

### Thermal monitoring

```python
# Before and after each run:
nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm
# If temp > 80°C or SM clock < 80% of max → insert longer cooldown
```

---

## 5. Metrics to Capture

### Per-configuration output

```json
{
  "config_id": "A3",
  "description": "BF16 group_offload (use_stream=True)",
  "model": "anima",
  "format": "bf16",
  "optimization": "group_offload",
  "phases": {
    "model_load_s": 2.3,
    "model_to_gpu_s": 0.0,
    "compile_time_s": 0.0,
    "cold_start_s": 4.2,
    "warm_state_mean_s": 3.1,
    "warm_state_std_s": 0.2,
    "warm_state_min_s": 2.9,
    "warm_state_max_s": 3.4
  },
  "vram": {
    "peak_allocated_mb": 4200,
    "peak_reserved_mb": 4800,
    "system_vram_mb": 5100
  },
  "quality": {
    "output_path": "/models/bench_results/A3_seed42.png"
  },
  "gpu": {
    "temp_before_c": 45,
    "temp_after_c": 62,
    "sm_clock_mhz": 2520,
    "max_sm_clock_mhz": 2520
  }
}
```

### Comparison table (generated from all configs)

```
Config    | Format | Optimization   | Cold (s) | Warm (s) | VRAM (MB) | Quality
----------|--------|-----------------|----------|----------|-----------|--------
A1        | BF16   | resident        |    ?     |    ?     |    ?      | ★★★★★
A2        | BF16   | model_cpu_off   |    ?     |    ?     |    ?      | ★★★★★
A3        | BF16   | group_offload   |    ?     |    ?     |    ?      | ★★★★★
B1        | BF16   | group_offload   |    ?     |    ?     |    ?      | ★★★★★
B2        | FP8    | group_offload   |    ?     |    ?     |    ?      | ★★★★☆?
C5        | int8   | mmGP            |    ?     |    ?     |    ?      | ★★★★☆?
```

---

## 6. Quality Comparison Protocol

### Step 1: Save output images

Every configuration saves its first timed-run output:
```
/models/bench_results/
  ├── A1_bf16_resident_seed42.png
  ├── A2_bf16_cpu_offload_seed42.png
  ├── A3_bf16_group_offload_seed42.png
  ├── B1_bf16_group_offload_seed42.png      (should be IDENTICAL to A3)
  ├── B2_fp8_group_offload_seed42.png        (may differ slightly)
  ├── C5_mmgp_int8_seed42.png               (current production)
  └── ...
```

### Step 2: Same-format verification

Configurations using the SAME format and seed should produce IDENTICAL output:
- A1, A2, A3 (all BF16) → should be pixel-identical
- If not identical → there's a bug in the offload path

Verify with:
```python
import numpy as np
from PIL import Image

img1 = np.array(Image.open("A1_bf16_resident_seed42.png"))
img3 = np.array(Image.open("A3_bf16_group_offload_seed42.png"))
print(f"Max pixel diff: {np.abs(img1.astype(int) - img3.astype(int)).max()}")
print(f"Mean pixel diff: {np.abs(img1.astype(int) - img3.astype(int)).mean():.4f}")
# Expected: max diff < 5, mean diff < 0.5 (minor floating-point variance)
```

### Step 3: Cross-format visual comparison

Different formats (BF16 vs FP8 vs int8 vs Q5) should be compared visually:
- Side-by-side comparison
- Focus on: skin textures, fabric detail, text rendering, color accuracy
- The AI's concern: flat FP8 "degrades micro-textures" — verify or debunk

### Step 4: Quantitative metrics (optional, if visual isn't conclusive)

```python
# PSNR (Peak Signal-to-Noise Ratio) — higher is better
# SSIM (Structural Similarity) — higher is better
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

reference = np.array(Image.open("A1_bf16_resident_seed42.png"))  # gold standard
candidate = np.array(Image.open("B2_fp8_group_offload_seed42.png"))

psnr = peak_signal_noise_ratio(reference, candidate)
ssim = structural_similarity(reference, candidate, channel_axis=2)
print(f"PSNR: {psnr:.2f} dB (30+ is good, 40+ is excellent)")
print(f"SSIM: {ssim:.4f} (0.95+ is good, 0.99+ is excellent)")
```

---

## 7. mmGP Baseline Comparison

The most important missing test: how does the CURRENT production system perform?

### Method 1: Through the Forge API (realistic)

```bash
# Send a generation request through the existing Ray Serve endpoint
curl -X POST http://forge-service:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux-schnell",
    "input_prompt": "...",
    "sampling_steps": 4,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }'

# Measure: wall-clock time, VRAM via nvidia-smi, save output image
```

### Method 2: Direct handler call (controlled)

```python
# Call the Wan2GP handler directly, bypassing API overhead
from models.flux.flux_main import FluxModel  # or similar

handler = FluxModel(...)
handler.load_model(...)
t0 = time.perf_counter()
output = handler.generate(seed=42, input_prompt="...", sampling_steps=4)
elapsed = time.perf_counter() - t0
```

### What to measure on the mmGP path

- Cold start (first generation after model load)
- Warm steady state (repeated generations)
- Peak VRAM (should match our int8 quantized model size)
- Output image (for quality comparison against native paths)
- Component breakdown if possible (text encode / denoise / VAE decode)

### Fair mmGP comparison notes

The mmGP path uses:
- Pre-quantized int8 files (`_quanto_bf16_int8.safetensors`)
- mmGP's block-level offload with CUDA streams
- mmGP's LoRA injection (if LoRAs are used)

This is MOST comparable to our B3 (int8 group_offload) configuration.
The fair comparison is:
```
C5 (mmGP int8 streaming)  vs  B3 (native int8 group_offload)
```

---

## 8. Execution Plan

### Phase 2a: Anima fair comparison (Groups A + B)

**Why Anima first:**
- Small enough for BF16 resident (3.9GB) — ALL paths testable
- 30 steps — cache_accel and compile can show value
- Custom pipeline — proves the framework works beyond standard pipelines
- Already have the handler code for mmGP comparison

**Steps:**
1. Write Anima native runner (loads CosmosTransformer3DModel + Qwen3 + VAE natively)
2. Test all Group A configs (BF16: resident, model_cpu_offload, group_offload, +compile, +cache)
3. Test all Group B configs (group_offload: BF16, FP8)
4. Run mmGP baseline (current anima_main.py)
5. Save all output images
6. Generate comparison table

**Estimated time:** 1 day

### Phase 2b: FLUX fair comparison (Groups B + C + D)

**Why FLUX second:**
- Standard pipeline — simpler to configure
- schnell (4 steps) and dev (20 steps) — tests step-count sensitivity
- Already downloaded and cached

**Steps:**
1. Test Group B configs (group_offload: BF16, FP8)
2. Test Group C configs (resident vs offload, quality vs speed)
3. Test Group D configs (4 steps vs 20 steps — cache/compile impact)
4. Run mmGP baseline
5. Save all images
6. Generate comparison table

**Estimated time:** 1 day

### Phase 2c: Quality evaluation

1. Download all saved images
2. Same-format verification (A1 vs A3 — should be identical)
3. Cross-format visual comparison (BF16 vs FP8 vs int8)
4. PSNR/SSIM metrics (BF16 reference vs others)
5. Document findings

**Estimated time:** half day

### Phase 2d: Decision document

Based on all data, produce:
- Recommended configuration per tier per model
- Quality acceptance criteria (what PSNR/SSIM is "good enough")
- Speed/VRAM/quality tradeoff curves
- Updated tier architecture with verified numbers

**Estimated time:** half day

---

## Appendix: Models to Download

| Model | Size | Format | Purpose | Status |
|-------|------|--------|---------|--------|
| FLUX.1-schnell | ~23GB | BF16 diffusers | 4-step speed testing | ✅ Cached at /models/hf_cache |
| FLUX.1-dev | ~23GB | BF16 diffusers | 20-step quality testing | ❌ Needs download + HF token |
| Anima | ~5.7GB | Component files | Custom pipeline testing | ✅ On disk at /models/wan2gp/ |
| Z-Image Turbo | ~6.5GB | BF16 diffusers | Alternative speed testing | ❌ Needs download |

Priority: Anima (already on disk) → FLUX-schnell (already cached) → others as needed.
