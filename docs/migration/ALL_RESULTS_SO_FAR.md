# ALL Benchmark Results So Far — Raw Data

> Consolidated from Phase 1 and Fair Comparison runs
> ALL numbers are from FLUX.1-schnell, 4 steps, 1024×1024, RTX 4090
> **WARNING: 4 steps is NOT representative of real workloads (Z-Image=40, FLUX-dev=20)**

---

## Fair Comparison (2026-06-14, seed=42, images saved)

| Config | Cold (s) | Warm (s) | Std | Min (s) | VRAM (MB) | Steps/s | Image |
|--------|----------|----------|-----|---------|-----------|---------|-------|
| fp8-group-offload | 4.27 | **3.41** | ±0.002 | 3.40 | 12,505 | 1.17 | ✅ seed42 |
| bf16-group-offload | 23.01 | **5.86** | ±0.607 | 5.17 | 12,505 | 0.69 | ✅ seed42 |
| bf16-cpu-offload | 38.24¹ | ~16.8² | — | — | ~24,300 | ~0.24 | ❌ |

¹ Cold only — pod killed before warm runs completed
² From Phase 1 data (different methodology, no seed control)

## Phase 1 Results (2026-06-14, earlier run, no seed control, no images)

| Config | Warm (s) | Std | VRAM (MB) | Steps/s | Notes |
|--------|----------|-----|-----------|---------|-------|
| fp8-group-offload | **3.83** | ±0.51 | 12,505 | 1.06 | layerwise_casting + group_offload |
| cache-only | 15.11 | ±0.13 | 24,365 | 0.26 | model_cpu_offload + first_block_cache |
| compile-only | 15.98 | ±1.02 | 24,113 | 0.25 | model_cpu_offload + compile_repeated_blocks |
| baseline (cpu-offload) | 16.75 | ±1.55 | 24,283 | 0.24 | model_cpu_offload only |

## Configurations NOT Tested

| Config | Why not | Priority |
|--------|---------|----------|
| BF16 fully resident | OOM — FLUX pipeline >24GB in BF16 | N/A (impossible) |
| int8 group_offload | Need int8 diffusers-format model | HIGH |
| FP8 scaled (torchao) | torchao not installed | MEDIUM |
| GGUF Q5 | Need GGUF model file | MEDIUM |
| GGUF Q8 | Need GGUF model file | LOW |
| compile + cache combined | INCOMPATIBLE (graph break) | N/A (impossible) |

## Per-step timing (from progress bars)

```
fp8-group-offload (Fair Comparison, warm):
  Step 1: ~0.85s    Step 2: ~0.85s    Step 3: ~0.85s    Step 4: ~0.85s

bf16-group-offload (Fair Comparison, warm):
  Step 1: ~1.47s    Step 2: ~1.47s    Step 3: ~1.47s    Step 4: ~1.47s

bf16-cpu-offload (Phase 1, warm):
  Step 1: ~5-7s     Step 2: ~2.5s     Step 3: ~1.7s     Step 4: ~1.3s
  (first step includes loading entire transformer to GPU)
```

## Incompatibilities Discovered

| Combination | Status | Error |
|-------------|--------|-------|
| torch.compile + group_offload | ❌ INCOMPATIBLE | swap_tensors vs TensorWeakRef |
| cache_accel + group_offload | ❌ INCOMPATIBLE | block-skipping breaks prefetch chain |
| cache_accel + torch.compile | ❌ INCOMPATIBLE | @torch.compiler.disable graph break |
| use_stream=True + blocks>1 | ⚠️ FORCED to 1 | diffusers 0.37.0 silently overrides |

## Models Cached on Persistent Storage

```
/models/flux-schnell/          — FLUX.1-schnell diffusers format (~26GB) ✅
/models/wan2gp/                — Wan2GP format models (int8 quanto files)
  flux1-schnell_quanto_bf16_int8.safetensors  (12GB, transformer only)
  flux_vae.safetensors                         (335MB)
```

## What We Need But Don't Have

1. **A 20-40 step model for realistic benchmarking** (Z-Image, FLUX-dev, or Anima)
2. **int8 quanto diffusers-format model** (to test int8 group_offload)
3. **torchao installed** (to test FP8 scaled vs FP8 flat)
4. **GGUF model files** (to test Q5/Q8 paths)
5. **Quality comparison** (images saved but not yet retrieved from pod)
6. **bf16-cpu-offload with proper methodology** (pod keeps dying)
