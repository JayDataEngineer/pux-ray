# TeaCache — Timestep Embedding Aware Cache

## Overview

TeaCache (Timestep Embedding Aware Cache) is a caching technique for diffusion
transformers that skips DiT block recomputation when consecutive timestep
embeddings are nearly identical. Instead of running all transformer blocks on
every denoising step, it caches the hidden states from a previous "full"
computation and reuses them when the model would produce near-identical output.

## How It Works

### Cache Signal

The cache signal is the **`timestep_proj`** vector — the output of
`self.condition_embedder()` (not raw timestep embedding, but the projected
representation fed to each DiT block). When consecutive timesteps produce
similar `timestep_proj` vectors, their DiT outputs will also be similar, so
recomputation is wasteful.

### Distance Metric

We use **raw L1 distance** (via `F.l1_loss`) between consecutive
`timestep_proj` vectors. No polynomial rescaling.

Threshold → behavior:
- `0` = disabled
- `0.001–0.005` = conservative (cache rarely, high quality)
- `0.01` = **sweet spot** (70% speedup, no visible quality loss)
- `0.015–0.02` = aggressive (max speed, slight quality trade-off)

## Why Raw L1 (Not Polynomial Rescaling)

The official TeaCache implementation uses polynomial coefficients
`[-5784.55, 5449.51, -1811.17, 256.27, -13.02]` to rescale the L1 distance
before comparing against the threshold. **This breaks on 14B+ models.**

For typical consecutive-step L1 distances of 0.001–0.04, the polynomial
produces `abs(poly) ≈ 5–13`. With any reasonable threshold (0.1–0.3), the
accumulated distance **always** exceeds the threshold, meaning no caching
ever triggers.

**Root cause**: the polynomial coefficients were calibrated for smaller models
(1.3B) where `timestep_proj` operates at a different scale. For 14B models,
skipping the polynomial and comparing raw L1 directly is correct.

## Architecture: CFG Handling

Classifier-Free Guidance (CFG) evaluates both conditional and unconditional
branches per step. TeaCache uses a **call counter** (`_tc_call_counter % 2`)
to track branches independently:

- **Even calls** (0, 2, 4, …) = conditional branch
- **Odd calls** (1, 3, 5, …) = unconditional branch

Each branch has its own `prev_timestep_proj` and `cached_hidden_states` in the
state dictionary (`tc_s`). The first 2 CFG pairs (4 calls total) always
compute as **retention steps** to build a reference cache.

## VACE-Specific: vace_blocks Always Run

VACE blocks (`self.vace_blocks`) process conditioning input (depth, pose,
edge, etc.) and are **never cached** — the conditioning changes every step
regardless of timestep similarity. Only the main DiT blocks
(`self.blocks`) benefit from caching.

## Benchmark Results

### 25-step Base Mode (640×480, 33 frames, RTX 4090)

| Threshold | Time   | Speedup | Std   | Quality vs Baseline |
|-----------|--------|---------|-------|---------------------|
| Baseline  | ~165s  | —       | ~84   | —                   |
| 0.005     | 85.9s  | **48%** | 83.9  | Identical (≤0.1σ)   |
| **0.01**  | **49.3s** | **70%** | **82.9** | **Excellent** |
| 0.015     | 44.3s  | 73%     | 80.8  | Slight trade-off    |
| 0.02      | 44.2s  | 73%     | 90.9  | Diminishing returns  |

### 10-step Fast Mode with TeaCache

| Threshold | Time  | Speedup vs 72s Baseline |
|-----------|-------|------------------------|
| Baseline  | ~72s  | —                       |
| 0.01      | 34.5s | **52%**                 |

### Key Insight

**25-step base mode + TeaCache(0.01) at 49s is faster than 10-step fast mode
without TeaCache at 72s**, with better quality. If the server has TeaCache
enabled, always prefer `vace_base` over `vace_fast`.

## Configuration

TeaCache is a **server-side toggle** controlled by the `OMNI_TEACACHE_THRESH`
environment variable:

```bash
# Disabled (default)
OMNI_TEACACHE_THRESH=0 bash scripts/run_omni_14b.sh

# Sweet spot
OMNI_TEACACHE_THRESH=0.01 bash scripts/run_omni_14b.sh

# Conservative
OMNI_TEACACHE_THRESH=0.005 bash scripts/run_omni_14b.sh

# Aggressive
OMNI_TEACACHE_THRESH=0.02 bash scripts/run_omni_14b.sh
```

All requests to the container benefit automatically — no per-request control.

## Implementation Notes

- File: `scripts/pipeline_wan2_2_vace_patch.py` (lines 1169–1410)
- Replaces `WanTransformer3DModel.forward` with `_teacache_forward`
- Activated at import time when `OMNI_TEACACHE_THRESH > 0`
- Uses `inspect.getsource()` on the real model to discover the correct
  `condition_embedder` interface at runtime
- Block call signature: `block(hs, enc_hs, timestep_proj, rotary_emb, mask)`
- Post-processing: `output_scale_shift_prepare(temb)` → `norm_out(hs, scale, shift)` → inline unpatchify
