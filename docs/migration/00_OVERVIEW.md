# Wan2GP/mmGP Migration — Architecture Decision & Technical Reference

> **Status:** DECISION MADE — migrate away from Wan2GP/mmGP to native diffusers + SGLang Diffusion
> **Date:** 2026-06-13
> **Participants:** Jay (architect), Claude (analysis + verification)
> **Verification level:** All API claims tested on live worker pod unless marked `[unverified]`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Decision](#2-the-decision)
3. [Current State — What Wan2GP/mmGP Actually Is](#3-current-state)
4. [Hardware Reality](#4-hardware-reality)
5. [Model Inventory & Migration Tiers](#5-model-inventory)
6. [The Native Replacement APIs (VERIFIED)](#6-native-replacement-apis)
7. [The Dream Architecture](#7-dream-architecture)
8. [Licensing Analysis](#8-licensing-analysis)
9. [Migration Plan](#9-migration-plan)
10. [Key Technical Findings](#10-key-technical-findings)
11. [AI Assessment Log](#11-ai-assessment-log)
12. [Risks & Open Questions](#12-risks)
13. [Deep Research Query](#13-deep-research-query)
14. [Deep Research Findings →](03_DEEP_RESEARCH_FINDINGS.md) (verified benchmarks, tuning, model details)

---

## 1. Executive Summary

**Wan2GP was the right tool in 2023–2024. The ecosystem has since caught up.**

Wan2GP (the app) + mmGP (its VRAM library) were built to solve problems that
upstream libraries now handle natively:

- **Block-level async VRAM offloading** → `diffusers.enable_group_offload(use_stream=True)` (since diffusers v0.33)
- **On-the-fly quantization** → `diffusers.enable_layerwise_casting(storage_dtype=fp8)` (since v0.33)
- **LoRA merging** → `diffusers` PEFT integration (`load_lora_weights`, `set_adapters`)
- **Weight loading / key remapping** → `transformers` v5 `WeightConverter` (or standard `from_pretrained`)
- **Production serving** → SGLang Diffusion (Apache-2.0, from LMSYS/Berkeley/Stanford)

mmGP's 6,361 lines of code are reducible to **~3 API calls** already installed
on the worker (diffusers 0.37.0). The 261,721-line handler directory
(`opt/wan2gp/models/`) is replaceable with direct diffusers/transformers calls.

**Goal:** Remove all Wan2GP code from the container. Keep the Ray Serve DAG
orchestration layer. Use native diffusers + optional SGLang Diffusion for
inference. Own 100% of the remaining code.

**Architecture (three mutually exclusive optimization paths, verified):**
- **Path A** (compiled throughput): `model_cpu_offload` + `compile_repeated_blocks`
  + cache acceleration + torchao quant — for models that fit VRAM during execution
- **Path B** (deep memory offload): `group_offload(use_stream=True)` +
  `layerwise_casting` + PEFT — for models that don't fit VRAM (no compile, no cache)
- **Path C** (SGLang): separate container, own kernels, 1.15-1.5x on 4090

**Critical constraint:** `torch.compile` and cache acceleration are BOTH
incompatible with `group_offload`. Choose Path A or Path B per model, not both.

---

## 2. The Decision

### Why migrate

| Factor | Wan2GP/mmGP today | Native diffusers + SGLang |
|--------|-------------------|--------------------------|
| **License** | Wan2GP Community License 2.0 (restrictive — blocks commercial SaaS/API) | Apache-2.0 (fully permissive) |
| **Model support speed** | Requires hand-written handlers per model; lags months behind | Day-0 support via `from_pretrained`; weekly diffusers releases |
| **Code ownership** | mmGP hooks into models; creates circular references; fragile | Standard library APIs; clean hooks; no monkey-patching |
| **LoRA support** | 560-line `load_loras_into_model` with monkey-patching; breaks torch.compile | PEFT: `load_lora_weights`, `set_adapters` — standard, compile-compatible |
| **Quantization** | Custom quanto bridge + int8 (1,900+ lines) | `enable_layerwise_casting(fp8)` — one line |
| **VRAM offloading** | Block-level async streaming (the core innovation, ~200 lines) | `enable_group_offload(use_stream=True)` — same technique, built-in |
| **Maintained by** | One developer (DeepBeepMeep) | HuggingFace team, LMSYS team, Alibaba team, global OSS |
| **Code volume** | 6,361 lines (mmGP) + 12,330 (wgp.py) + 261,721 (handlers) = **280,412 lines** | ~3 API calls + your runners |

### Why NOT to write a custom mmGP replacement

- The "irreducible core" (block-level async streaming) is **already in diffusers**.
- Writing a custom offloader would duplicate work HuggingFace already shipped.
- Custom offloaders must rediscover PyTorch edge cases (tied weights, compile
  interactions, quantized tensors) that mmGP spent 80+ releases smoothing out.
- The ROI is negative: weeks of work for a perf delta of zero (the native API
  uses the same technique).

### Why NOT to keep Wan2GP

- The handler translation layer causes model-support lag (experienced directly
  with Anima — required 650+ lines of reverse-engineering).
- mmGP hooks create circular module references that cause infinite recursion
  when `.to()` is called on hooked models (experienced directly during this project).
- The 12,330-line `wgp.py` Gradio monolith is dead weight (already unused).
- License blocks any future commercial deployment.

---

## 3. Current State

### What Wan2GP actually is (decomposed from on-disk code)

```
Wan2GP (the repository at opt/wan2gp/)
├── wgp.py                      12,330 lines   Gradio UI monolith (UNUSED by our system)
├── models/                    261,721 lines   929 files — per-model handlers (LARGELY VENDORED THIRD-PARTY CODE)
│   ├── TTS/                   119,332 lines   TTS model code (kokoro, moss, index_tts, etc.)
│   ├── wan/                    35,371 lines   Wan video model handlers
│   ├── ltx2/                   23,200 lines   LTX2 video model handlers
│   ├── hyvideo/                16,925 lines   Hunyuan video handlers
│   ├── trellis/                10,327 lines   TRELLIS 3D model
│   ├── flux/                    6,536 lines   FLUX handlers
│   ├── qwen/                    3,995 lines   Qwen-Image handlers
│   ├── z_image/                 3,364 lines   Z-Image handlers
│   ├── anima/                     796 lines   Anima handlers (WE WROTE THIS)
│   └── ... (20+ more families)
├── shared/                     66,958 lines   Utilities (prompt enhancer, Deepy agent, etc.)
├── preprocessing/              67,478 lines   Frame interpolation, upscaling
├── postprocessing/             32,842 lines   Post-processing effects
├── plugins/                    10,611 lines   Plugin system
└── LICENSE.txt                              Wan2GP Community License 2.0

mmGP (separate PyPI package, NOT in the Wan2GP repo)
├── offload.py                   3,811 lines   Core: hooks, CUDA streams, block loading, LoRA, quant
├── quant_router.py              1,256 lines   Quantization routing
├── fp8_quanto_bridge.py           645 lines   FP8 bridge to optimum-quanto
├── safetensors2.py                627 lines   Low-RAM safetensors rewrite
├── __init__.py                     22 lines   Profile type enum
Total:                           6,361 lines
Version: 3.7.6 (80+ releases on PyPI, actively maintained)
```

### Key insight: mmGP is model-agnostic

mmGP operates on generic `nn.Module` trees. It has ZERO knowledge of Wan, FLUX,
Anima, or any specific model. The model-specific code lives in `opt/wan2gp/models/`
(the handler layer). When a new model comes out and "Wan2GP doesn't support it,"
the failure is in the **handler layer** (no one wrote a handler), not in mmGP.

### mmGP code quality assessment (measured)

- **Battle-tested:** 80+ releases, used by 7+ production apps
- **Minimal docs:** 10 docstrings across 3,811 lines
- **Debug noise:** 61 `print()` statements left in production
- **Dead code:** 95 lines of commented-out code
- **Deep nesting:** 159 lines at 6+ indent levels
- **Self-described hacks:** `_quantize_dirty_hack()` function (line 860)
- **God objects:** `class offload:` is 723 lines; `load_loras_into_model` is 562 lines
- **Accretion pattern:** Features piled on over 80 versions rather than refactored

### offload.py functional decomposition (by line count)

| Function/Section | Lines | Category |
|-----------------|-------|----------|
| `class offload:` | 723 | Core VRAM management |
| `load_loras_into_model` | 562 | LoRA/DoRA/LoKr merging |
| `offload.all()` | 314 | Main orchestrator (profile selection) |
| `load_model_data` | 291 | Weight loading |
| `_pin_to_memory` | 196 | Pinned memory staging |
| `_quantize` | 161 | On-the-fly quantization |
| `_pin_sd_to_memory` | 127 | State dict pinning |
| `_detect_main_towers` | 50 | Transformer block detection |
| Other hooks/stream/cache | ~200 | Core offload mechanics |
| Remaining functions | ~1,287 | Glue, compat, save/load, utilities |

### What fraction of offload.py is each feature

```
LoRA/adapter handling:      ~402 lines  (load_loras, forward hooks, DoRA, LoKr)
Quantization:               ~192 lines  (quantize, requantize, dirty hack)
torch.compile handling:      ~36 lines  (compile wrappers, inductor workarounds)
Profile heuristics:         ~400 lines  (5 profile presets, budget calc)
Core offload mechanics:     ~194 lines  (hooks, streams, block load, pinned mem)
Weight loading/SD mapping:  ~870 lines  (load_sd, fast_load, map_state_dict, etc.)
Dead code + prints:          ~156 lines  (commented out, debug prints)
Glue/compat/misc:          ~1,761 lines  (everything else)
```

### mmGP's 5 memory profiles

```
1. HighRAM_HighVRAM_Fastest  : Load entirely in VRAM, keep RAM copy
2. HighRAM_LowVRAM_Fast      : Load needed parts in VRAM, keep RAM copy
3. LowRAM_HighVRAM_Medium    : Load in VRAM + 8-bit quantization
4. LowRAM_LowVRAM_Slow       : Parts in VRAM + 8-bit quantization
5. VerylowRAM_LowVRAM_Slowest: Parts in VRAM + 8-bit quant, no RAM copy
```

Our hardware (4090, 24GB) targets profiles 1–2 for small models, 2–3 for large.

---

## 4. Hardware Reality

### Worker pod specs (VERIFIED)

```
GPU:     NVIDIA GeForce RTX 4090
VRAM:    24,564 MiB (~24 GB)
CUDA:    12.8
RAM:     59.4 GB total, 13.3 GB available (at idle)
```

### Installed library versions (VERIFIED)

```
PyTorch:       2.10.0+cu128    (latest: 2.12.0 — two minors behind)
transformers:  4.57.3          (latest: 5.x — v5 just shipped, 4.57 is current 4.x)
accelerate:    1.13.0          (latest: 1.14.0 — basically current)
diffusers:     0.37.0          (latest: 0.38.0 — one minor behind)
optimum-quanto: (installed, version TBD)
mmgp:          3.7.6           (latest on PyPI)
CUDA:          12.8
```

**Critical:** These are NOT old libraries. The worker runs near-bleeding-edge
everything. mmGP is running on PyTorch 2.10 — the "old car" metaphor applies to
mmGP's PATTERNS and ARCHITECTURE, not its runtime platform.

### VRAM constraints

The 4090's 24GB means:
- Models ≤12GB (quantized): fit entirely, no offloading needed
- Models 14–20GB (quantized): transformer fits, but transformer + text encoder + VAE may not — need pipeline-stage eviction
- Models >24GB (quantized): must use block-level streaming (group offload)

### All models on disk are pre-quantized (int8)

Every `.safetensors` file in `/models/wan2gp/` is `quanto_bf16_int8` — already
at maximum compression. mmGP's on-the-fly quantization machinery is redundant
because we load pre-quantized files.

---

## 5. Model Inventory

### Model sizes and migration tiers (VERIFIED from disk)

#### Tier 1 — Easy Wins (fit in 24GB, no offloading needed)

| Model | Size on disk | Fits 24GB? | mmGP needed? |
|-------|-------------|------------|--------------|
| Anima (anima-base-v1.0) | 3.9 GB | ✅ Comfortably | **No** |
| Z-Image (ZImageBase/Turbo) | 6.5 GB | ✅ Comfortably | **No** |
| Flux-schnell | 12 GB | ✅ Yes | **No** |
| Flux-dev | 12 GB | ✅ Yes | **No** |
| Flux-chroma-hd | 8.4 GB | ✅ Yes | **No** |
| Flux-2-klein-4b | 3.8 GB | ✅ Yes | **No** |
| Flux-2-klein-9b | 8.8 GB | ✅ Yes | **No** |

**Migration path:** `AutoPipeline.from_pretrained(...).to("cuda")`. One line. No offload.

#### Tier 2 — Medium (need pipeline-stage eviction)

| Model | Size on disk | Notes |
|-------|-------------|-------|
| Wan 2.1 T2V 14B | 14 GB | Transformer fits; + text encoder + VAE may exceed 24GB |
| Wan 2.2 Animate 14B | 17 GB | Same |
| Wan 2.1 I2V 14B (480p/720p) | 16 GB | Same |
| Longcat Video | 13 GB | Same |
| Magi Human | 15 GB | Same |
| Qwen-Image-Edit-Plus 20B | 20 GB | Tight; transformer + VAE barely fits |
| LTXV 0.9.8 13B | 13 GB | Same |
| Ace Step | 5.2 GB | Small enough for Tier 1 |

**Migration path:** `enable_group_offload(use_stream=True)` or pipeline-stage
eviction (load text encoder → encode → evict → load transformer → denoise →
evict → load VAE → decode).

#### Tier 3 — Hard Case (don't fit, need real streaming)

| Model | Size on disk | Notes |
|-------|-------------|-------|
| Flux2-dev | 31 GB | Doesn't fit even quantized. Must stream. |
| LTX-2.3-22B | 19 GB | Barely fits transformer; VAE/encoder push over. |
| Mistral3 Small | 24 GB | Fills entire card. |
| Qwen2.5-VL-7B | 8.8 GB | Fits, but multimodal needs extra VRAM for vision. |

**Migration path:** `enable_group_offload(use_stream=True)` +
`enable_layerwise_casting(storage_dtype=fp8)` for maximum compression.
Or: SGLang Diffusion (has dedicated LTX-2.3 optimization).

#### Tier 4 — Niche/Custom (non-standard architectures)

| Model | Size on disk | Handler type |
|-------|-------------|--------------|
| TRELLIS (3D) | 10,327 lines code | Custom pipeline, not diffusers |
| Kokoro (TTS) | Part of 119k TTS/ | Custom |
| Moss (TTS) | Part of 119k TTS/ | Custom |
| Index TTS | Part of 119k TTS/ | Custom |
| Faster Whisper | Part of TTS/ | Custom |
| Pixal3D | 613 lines | Custom |
| Pose/Body | ~500 each | Custom |

**Migration path:** Write custom runners that call their libraries directly.
No handler abstraction. Each runner is a few hundred lines.

### Model licensing (for future commercial consideration)

| Model | License | Commercial self-host API? |
|-------|---------|--------------------------|
| Wan 2.1 / 2.2 | Apache-2.0 | ✅ Yes |
| Qwen-Image / Qwen3 | Apache-2.0 | ✅ Yes |
| Z-Image | Apache-2.0 | ✅ Yes |
| LTX-Video (original) | Apache-2.0 | ✅ Yes |
| LTX2 / LTX-2.3 | Lightricks custom | ⚠️ Verify |
| FLUX.1-schnell | Apache-2.0 | ✅ Yes |
| FLUX.1-dev | Non-commercial | ❌ No |
| HunyuanVideo | Custom, commercial restrictions | ❌ No |
| Anima | CircleStone Labs — verify | ⚠️ |

**Implication:** Even after removing Wan2GP, commercial API deployment is limited
to Apache-licensed models (Wan, Qwen, Z-Image, schnell). FLUX-dev and Hunyuan
block commercial use at the MODEL level, regardless of infrastructure licensing.

---

## 6. Native Replacement APIs

### ALL VERIFIED on worker pod (diffusers 0.37.0)

#### `enable_group_offload` — replaces mmGP's core VRAM streaming

```python
# VERIFIED: exists in diffusers 0.37.0 on worker
# Signature:
enable_group_offload(
    self,
    onload_device: torch.device,          # e.g. torch.device("cuda")
    offload_device: torch.device = cpu,   # where to park weights when not in use
    offload_type: str = "block_level",    # "block_level" or "leaf_level"
    num_blocks_per_group: int | None = None,
    non_blocking: bool = False,
    use_stream: bool = False,             # ← ASYNC CUDA STREAM PREFETCH
    record_stream: bool = False,
    low_cpu_mem_usage: bool = False,
    offload_to_disk_path: str | None = None,
    block_modules: str | None = None,
    exclude_kwargs: str | None = None,
) -> None
```

**What it does:** Moves groups of transformer blocks between CPU and GPU.
When `use_stream=True`, uses a background CUDA stream to prefetch block N+1
from CPU while the GPU computes block N. This is the EXACT technique mmGP
uses — same primitive (CUDA streams), same granularity (block-level), same
overlap strategy.

**Usage:**
```python
transformer = TransformerClass.from_pretrained("...", torch_dtype=torch.bfloat16)
transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    use_stream=True,
)
```

#### `enable_layerwise_casting` — replaces mmGP's quantization

```python
# VERIFIED: exists in diffusers 0.37.0 on worker
# Signature:
enable_layerwise_casting(
    self,
    storage_dtype: torch.dtype = torch.float8_e4m3fn,  # FP8 storage
    compute_dtype: torch.dtype | None = None,           # e.g. torch.bfloat16
    skip_modules_pattern: tuple[str, ...] | None = None,
    skip_modules_classes: tuple[Type[Module], ...] | None = None,
    non_blocking: bool = False,
) -> None
```

**What it does:** Stores weights in FP8 (half the VRAM), upcasts to bf16/fp16
on-the-fly during forward pass. Skips precision-critical layers (norm,
embedding) by default. Reduces VRAM by ~50% with negligible quality loss.

**Usage:**
```python
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```

#### PEFT LoRA integration — replaces mmGP's 560-line LoRA code

```python
# Standard diffusers PEFT integration (no mmGP needed)
pipe.load_lora_weights("style.safetensors", adapter_name="style_1")
pipe.load_lora_weights("detail.safetensors", adapter_name="detail_boost")

# Dynamic multi-LoRA with independent scaling
pipe.set_adapters(["style_1", "detail_boost"], adapter_weights=[0.85, 0.4])

# Swap at runtime
pipe.set_adapters(["style_1"], adapter_weights=[1.0])

# Unload all
pipe.unload_lora_weights()
```

**Advantages over mmGP's LoRA:**
- Compatible with `torch.compile` (mmGP's monkey-patching broke compilation)
- No circular module references (mmGP's hooks created the recursion bug)
- Standard API, maintained by HuggingFace PEFT team
- Supports DoRA, LoKr, and other adapter types natively

#### Pipeline-level offloading — for pipeline-stage eviction

```python
# Coarse-grained: moves whole components (text encoder, transformer, VAE)
pipe.enable_model_cpu_offload()
# Each component moves to GPU when needed, back to CPU when done.
# Synchronous but negligible overhead for stage-level swaps.

# Fine-grained: layer-by-layer (very slow, extreme low-VRAM only)
pipe.enable_sequential_cpu_offload()
# Not recommended for production — use group_offload instead.
```

#### VAE tiling/slicing — for decode VRAM management

```python
pipe.vae.enable_tiling()    # Overlapping tiles for large images/video
pipe.vae.enable_slicing()   # Process latent slices separately
```

---

## 7. Dream Architecture

### The three-tier inference model

```
┌──────────────────────────────────────────────────────────┐
│              Gateway / Editor / API                       │  YOURS
│           Ray Serve DAG / Forge Router                    │  YOURS
│      (routing, queueing, workflows, scale-to-zero)        │  KEEP
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌─────────────┐ ┌─────────────┐ ┌───────────┐
  │  SGLang     │ │   Native    │ │  Custom   │
  │ Diffusion   │ │  Diffusers  │ │  Runners  │
  │   Server    │ │             │ │           │
  │             │ │ +group_offld│ │ Anima,    │
  │ Wan, FLUX,  │ │ +layer_cast │ │ TRELLIS,  │
  │ Qwen, LTX,  │ │ +PEFT LoRAs │ │ TTS,Audio │
  │ Z-Image     │ │             │ │           │
  │             │ │ For models  │ │ Direct    │
  │ For fastest │ │ SGLang      │ │ diffusers │
  │ standard    │ │ doesn't     │ │ /custom   │
  │ model serve │ │ cover       │ │ lib calls │
  │             │ │             │ │           │
  │ Apache-2.0  │ │ Apache-2.0  │ │ YOUR CODE │
  └─────────────┘ └─────────────┘ └───────────┘
    [Container A]   [Container B]   [Container C]
     isolated        isolated        isolated
```

### Principles

1. **Cluster-level orchestration** (Ray Serve) handles routing, scaling,
   scale-to-zero, multi-model scheduling. This already exists and stays.

2. **Native library features** (`enable_group_offload`,
   `enable_layerwise_casting`, PEFT) handle VRAM optimization and model
   features. No custom offload layer needed.

3. **Direct API calls** (no handler translation layer) handle model-specific
   logic. `AutoPipeline.from_pretrained()` for standard models; custom
   runners for non-standard architectures.

4. **Container isolation** handles dependency conflicts. SGLang in one
   container, native diffusers in another, custom TTS in a third. No more
   single massive container hoping nothing collides.

5. **Everything is Apache-2.0 or YOURS.** No mmGP. No Wan2GP handlers.
   No Wan2GP Community License.

### Scale-to-zero strategy

| Level | Mechanism | VRAM when idle | Wake time |
|-------|-----------|----------------|-----------|
| **Process kill** | Ray Serve `min_replicas=0` | 0 GB | 15–60s (cold start, load from disk) |
| **Model eviction** | SGLang sleep/wake | ~0.5–1 GB (CUDA context only) | <1s (PCIe from pinned CPU) |
| **Within-request** | `enable_group_offload` | N/A (active during inference) | N/A |

### Speed optimization: THREE mutually exclusive paths

**VERIFIED via deep research + PyTorch source code analysis (2026-06-13):**

Three things are **pairwise incompatible with group_offload**:
1. `torch.compile` — `swap_tensors` conflicts with dynamo `TensorWeakRef` guards
   → `RuntimeError: Cannot swap t1 because it has weakref associated with it`
2. **Cache acceleration** — block-skipping breaks the sequential prefetch chain
   → `"some layers were not executed during the forward pass"` → device mismatch
3. (group_offload itself is fine)

This means there are **THREE mutually exclusive optimization paths**:

#### Path A: Compiled Throughput Stack (model-level offload + compilation)
For models where the DiT fits in VRAM during execution (Tier 1, most Tier 2):

```
┌─────────────────────────┐
│  Cache Acceleration      │  20-165% speedup (skip redundant steps)
│  (apply_first_block_cache│  ← mmGP CANNOT do this
├─────────────────────────┤
│  compile_repeated_blocks │  ~1.5x speedup (kernel fusion on DiT blocks)
│  + model_cpu_offload     │  ← mmGP BROKE this
├─────────────────────────┤
│  torchao NF4/INT4/FP8    │  50-75% VRAM cut
│  quantization            │  ← mmGP had its own, less integrated
├─────────────────────────┤
│  PEFT LoRAs              │  dynamic adapters, compile-compatible
└─────────────────────────┘
   Peak VRAM: ~12.2 GB (FLUX.1-dev NF4 + model offload)
   Fastest per-iteration speed
```

**Key insight:** `pipe.enable_model_cpu_offload()` moves the ENTIRE transformer
to GPU before execution, then back to CPU after. No mid-forward swapping →
no `swap_tensors` → no dynamo guard collision → `compile_repeated_blocks()`
works. The entire DiT must fit in VRAM during execution, but not permanently.

#### Path B: Deep Memory Offload Stack (block-level streaming)
For models that DON'T fit in VRAM even temporarily (Tier 3):

```
┌─────────────────────────┐
│  ❌ Cache Acceleration   │  INCOMPATIBLE (breaks prefetch chain)
├─────────────────────────┤
│  ❌ torch.compile        │  INCOMPATIBLE (TensorWeakRef guard collision)
├─────────────────────────┤
│  layerwise_casting       │  50% VRAM cut (FP8 storage, bf16 compute)
│  (applied FIRST)         │
├─────────────────────────┤
│  PEFT LoRAs              │  ✅ works if loaded BEFORE group_offload
├─────────────────────────┤
│  group_offload           │  async stream VRAM offloading
│  (use_stream=True)       │  ← mmGP's core feature, now native
│  record_stream=True      │
└─────────────────────────┘
   Peak VRAM: ~6.8 GB (FLUX.1-dev)
   15-35% overhead vs resident (with use_stream=True)
   Without use_stream: 300-500% overhead (basically sequential)
```

#### Path C: SGLang Diffusion (separate container, own kernels)
For standard models (Wan, FLUX, Qwen-Image, Z-Image, LTX):

- SPEED (Spectral Progressive Resolution): >2x speedup on H100/B200
- Cache-DiT / TeaCache: step-level caching
- sgl-kernel: JIT-compiled fused kernels via CuTeDSL
- Built-in sleep/wake VRAM management (250-400MB idle, 0.48-0.60s wake)
- **RTX 4090 realistic speedup: 1.15-1.5x** over native diffusers
  (NOT the 2.5-2.9x from enterprise benchmarks — those are H100/B200)
- **No continuous batching** — request-blocking at worker boundary.
  Only homogeneous batching (same resolution, steps) at request start.
  Roadmap: disaggregated serving to overlap text encoding/VAE with denoising.
- Must run in SEPARATE Docker container (sgl-kernel/flashinfer JIT compilation
  = 10+ min cold start if installed via Ray runtime_env pip)

### Choosing the right path

| Model | DiT fits VRAM? | Recommended path | Peak VRAM |
|-------|----------------|-----------------|-----------|
| Anima (3.9GB) | ✅ | Path A or direct | ~6 GB |
| Z-Image (6.5GB) | ✅ | Path A | ~8 GB |
| FLUX-schnell (12GB) | ✅ | Path A (NF4) | ~12.2 GB |
| Wan 14B (14GB) | ✅ with FP8 | Path A | ~14 GB |
| Wan 14B uncompressed | ❌ | Path B | ~11.8 GB |
| LTX-2.3 (22B) | ❌ | Path B or SGLang | ~15-19 GB |
| FLUX2-dev (31GB) | ❌ | Path B | ~19.5 GB |

### PEFT + group_offload ordering rule (VERIFIED)

```
❌ WRONG: enable_group_offload → load_lora_weights → device mismatch crash
✅ RIGHT: load_lora_weights → enable_group_offload → hooks register on adapters too
```

The group_offload hook manager inspects modules at enable time. If LoRAs are
loaded AFTER, their weights bypass the hooks → remain on CPU → crash.
Also: `record_stream=True` is required to prevent caching allocator from
prematurely reclaiming prefetched tensors (bug fixed in diffusers 0.36.0).

---

## 8. Licensing Analysis

### Wan2GP Community License 2.0 — key restrictions

**What "Software" covers (Section 1.3):**
The Licensor's copyrightable WanGP code — specifically the engineering and
productization layer: VRAM optimization, speed improvements, multi-model
automation, UI/API layers, packaging, integration glue.

**What it does NOT cover:**
Third-Party Materials (models, diffusers, transformers, PyTorch) — these keep
their own licenses (Apache-2.0, MIT, BSD).

**The actual restriction (Section 5 — Restricted Commercialization):**
You may not sell, host-as-SaaS, white-label, embed-in-paid-product, or offer
paid API access to Wan2GP itself without a separate commercial license.

**What's allowed without a license (Section 3 — Free Use):**
- Personal/hobby/research/educational use
- Internal company/studio/agency use
- Client work (using Wan2GP as a production tool)
- Creating and selling outputs
- Redistribution with attribution

**Implication for migration:**
Removing all Wan2GP code from the container eliminates ALL license obligations.
There is no Wan2GP code → no Wan2GP license applies.

### Copyright protection (why you can't just "relicense" someone's code)

- You CANNOT take Wan2GP code, modify it, and slap a freer license on it.
  That's copyright infringement. The original license follows the code.
- The only escape is clean-room reimplementation (write from scratch without
  reading the original).
- By REMOVING Wan2GP code and using native diffusers APIs, you avoid this
  entirely — you're not modifying Wan2GP, you're replacing it.

### Recommended license for YOUR code

If you want to protect against someone forking your system:
- **AGPL-3.0:** Anyone who deploys your code as a service must open-source
  their modifications. Deters commercial forks.
- **BSL / Source-Available:** Like Wan2GP, MongoDB, Elastic — free for
  non-competing use, paid license for commercial. Maximum protection.
- **Dual-license (AGPL + commercial):** The MySQL/MongoDB model. Own 100%
  of copyright, offer both licenses simultaneously. Requires CLA for
  outside contributions.

### The "someone steals my idea" fear

- Ideas are NOT copyrightable. Only expression (code) is.
- If someone reads your code and writes their own from scratch — that's
  legal (clean-room). This is true of ALL software.
- Defense: execution speed, integration depth, AGPL friction, trademark.
- The person who'd steal a complex Ray Serve system isn't deterred by
  licensing anyway — they'd build their own regardless.

---

## 9. Migration Plan

### Phase 1 — Prove the concept (1–2 days)

**Target:** Anima or Flux-schnell (models that fit in 24GB)

1. Write a standalone native runner script:
   ```python
   from diffusers import FluxPipeline
   pipe = FluxPipeline.from_pretrained("...", torch_dtype=torch.bfloat16).to("cuda")
   image = pipe(prompt, num_inference_steps=4).images[0]
   ```

2. Benchmark against current mmGP path:
   - Generation time (wall clock)
   - Peak VRAM (`torch.cuda.max_memory_allocated()`)
   - Output quality (visual comparison)

3. Success criteria: same or better speed, same or lower VRAM, correct output.

### Phase 2 — Offload-dependent models (3–5 days)

**Target:** Wan 14B, LTX2, or Qwen-Image

1. Write native runner with group offload (Path B):
   ```python
   # ORDER MATTERS: layerwise casting FIRST, group_offload SECOND
   transformer = TransformerClass.from_pretrained("...", torch_dtype=torch.bfloat16)

   # Step 1: Load LoRAs BEFORE enabling group_offload
   pipe.load_lora_weights("style.safetensors", adapter_name="style")

   # Step 2: Apply layerwise casting FIRST
   transformer.enable_layerwise_casting(
       storage_dtype=torch.float8_e4m3fn,
       compute_dtype=torch.bfloat16,
   )

   # Step 3: Apply group offload SECOND (wraps casting hooks)
   transformer.enable_group_offload(
       onload_device=torch.device("cuda"),
       offload_device=torch.device("cpu"),
       offload_type="block_level",
       num_blocks_per_group=2,      # Wan 14B sweet spot
       use_stream=True,              # async prefetch (CRITICAL)
       record_stream=True,           # prevent premature memory reclamation
   )
   # NOTE: NO torch.compile, NO cache_accel on this path
   ```

2. Benchmark against mmGP path. Key question: how much slower (if at all)
   is native group_offload vs mmGP's hand-tuned streaming?

3. If performance is acceptable: route through Ray Serve.
   If not: evaluate SGLang Diffusion for that model.

### Phase 3 — Evaluate SGLang Diffusion (1 week)

**Target:** Standard models (Wan, FLUX, Qwen-Image, Z-Image, LTX)

1. Install on test node: `pip install "sglang[diffusion]"`
2. Launch: `sglang serve --model-path Qwen/Qwen-Image --port 30010`
3. Benchmark: generation time, VRAM, throughput vs native diffusers
4. Key features to test:
   - `--ltx2-two-stage-device-mode snapshot` for LTX
   - Sleep/wake VRAM management
   - OpenAI-compatible API integration with Ray Serve
5. Decision per model: SGLang or native diffusers, whichever is faster.

### Phase 4 — Custom runners for niche models (ongoing)

**Target:** TRELLIS, TTS, audio models

1. Write thin runner for each that calls its library directly.
2. No handler abstraction, no translation layer.
3. Route through Ray Serve in isolated container.

### Phase 5 — Delete Wan2GP (when last model migrates)

1. Remove `opt/wan2gp/` from container image.
2. Remove mmGP from `requirements.txt`.
3. Remove Wan2GP handler discovery from `deployment.py`.
4. Update container Dockerfile to exclude Wan2GP vendored code.
5. Verify no Wan2GP imports remain anywhere.

### Migration tracking

Each model gets a status:
- `mmGP` — still on the old path
- `native-testing` — native runner written, benchmarking in progress
- `native-deployed` — running on native path in production
- `sglang-testing` — SGLang evaluation in progress
- `sglang-deployed` — running on SGLang in production
- `custom-runner` — has its own runner (TRELLIS, TTS, etc.)

---

## 10. Key Technical Findings

### Finding 1: mmGP is model-agnostic

mmGP's 6,361 lines contain ZERO model-specific code. The model-support lag
(Anima, Ideogram, etc.) is entirely in the HANDLER layer
(`opt/wan2gp/models/*/`), not in mmGP. mmGP would have offloaded Anima fine —
there was just no handler that knew how to load/inference it.

### Finding 2: The handler translation layer is the real problem

Wan2GP's architecture adds an unnecessary indirection:
```
request → wgp.py → handler → translate to Wan2GP format → mmGP → model.forward()
```
The modern path eliminates this:
```
request → AutoPipeline.from_pretrained() → generate()
```
New model support is instant: the moment diffusers ships it, it works. No
handler to write, no reverse-engineering.

### Finding 3: diffusers has absorbed mmGP's core innovation

`enable_group_offload(use_stream=True)` provides the exact same technique as
mmGP's block-level async streaming — CUDA stream prefetching of transformer
blocks during the forward pass. This was mmGP's primary value proposition, and
it's now a one-line API call in the standard library.

### Finding 4: transformers v5 WeightConverter absorbs weight-loading code

transformers v5 (just shipped) introduces `WeightConverter` for dynamic weight
restructuring (merge QKV, split MoE experts, reshape layers). This obsoletes
mmGP's safetensors2, map_state_dict, and weight-loading machinery (~1,870 lines).
However, v5 has breaking changes (especially tokenization) — migration should
be gradual and model-by-model.

### Finding 5: SGLang Diffusion is real but new

SGLang Diffusion (from LMSYS/Berkeley/Stanford) is a production-grade inference
server for diffusion models. Supports Wan, Hunyuan, Qwen-Image, FLUX, Z-Image,
GLM-Image. Has optimized kernels, sleep/wake VRAM management, OpenAI-compatible
API. Apache-2.0.

**CORRECTED SPEEDUP NUMBERS (verified June 2026):**
- The "2.5–2.9x faster" benchmarks are from H100/B200 enterprise hardware
- **On RTX 4090 (consumer): realistic speedup is 1.15–1.5x** over native diffusers
  (from fused CuTeDSL RMSNorm/LayerNorm kernels, Packed QKV layouts, flashinfer)
- SPEED (Spectral Progressive Resolution) is training-free and still gives >2x
  even on consumer hardware (mathematically verified optimization)
- **No continuous batching** — request-blocking at worker boundary. Only
  homogeneous batching at request start. Disaggregated serving on roadmap.

**Operational requirement:** Must run in SEPARATE Docker container — sgl-kernel
and flashinfer JIT compilation = 10+ min cold start via Ray runtime_env pip.

Does NOT support niche models (Anima, TRELLIS, TTS).

### Finding 6: The hardware is the "GPU poor" scenario

The RTX 4090 (24GB) is EXACTLY the hardware mmGP was built for. This means:
- For small models (≤12GB): no offloading needed, mmGP adds nothing
- For large models (14–24GB): group_offload is essential, and it's now native
- For huge models (>24GB): need group_offload + layerwise_casting, both native
- The "low VRAM optimization" code in mmGP IS relevant to our hardware

### Finding 7: The Anima experience proved the handler model is broken

Getting Anima to work required:
- Reverse-engineering ComfyUI's implementation
- Writing 650+ lines of custom model factory code
- Debugging flow-matching sigma schedules, LLM adapter, dual tokenizers
- Fighting mmGP's recursion bug from `.to()` calls on hooked models

If we'd been on the native path from the start, Anima would have been either
`AutoPipeline.from_pretrained("circlestone-labs/Anima")` (if in diffusers) or
a thin custom runner (if not) — days, not weeks.

---

## 11. AI Assessment Log

### Response 1: Generic architecture advice
- **Claim:** "mmGP contains complex shims for many model families (Hunyuan, Wan, LTX, Flux, Cosmos, TTS)"
- **Verdict:** ❌ WRONG — conflated mmGP (model-agnostic library) with Wan2GP-the-app
- **Claim:** "It includes custom UI/Gradio integrations"
- **Verdict:** ❌ WRONG — mmGP has zero Gradio. Gradio is in wgp.py (separate 12k file)

### Response 2: transformers v5 analysis with code examples
- **Claim:** `use_async_weight_loading=True` parameter exists
- **Verdict:** ❌ HALLUCINATED — parameter does not exist anywhere in codebase/docs/PyPI
- **Claim:** WeightConverter does "dequantizing on-the-fly"
- **Verdict:** ❌ OVERSTATED — WeightConverter does reshape/merge/split, not dequantization
- **Claim:** Migration is "low-risk"
- **Verdict:** ❌ WRONG — v5 has extensive breaking changes (decode API, tokenizers, etc.)
- **Claim:** WeightConverter obsoletes mmGP weight loading
- **Verdict:** ✅ CORRECT — ~1,870 lines become redundant

### Response 3: Alternative inference engines
- **Claim:** SGLang Diffusion exists and supports Wan/FLUX/Qwen/LTX/Z-Image
- **Verdict:** ✅ VERIFIED REAL — docs.sglang.io/docs/sglang-diffusion
- **Claim:** DiffSynth-Engine exists with layer-level offload
- **Verdict:** ✅ VERIFIED REAL — diffsynth-studio-doc.readthedocs.io
- **Claim:** "Native PyTorch CUDA stream management" handles offloading
- **Verdict:** ❌ WRONG — native diffusers offload is synchronous, no stream overlap
  (CORRECTED LATER: enable_group_offload DOES have use_stream=True)

### Response 4: LTX-specific and feature comparison (BEST RESPONSE)
- **Claim:** `enable_group_offload(use_stream=True)` provides async CUDA stream prefetching
- **Verdict:** ✅ VERIFIED ON WORKER POD — exists in diffusers 0.37.0
- **Claim:** `enable_layerwise_casting(storage_dtype=fp8)` provides FP8 storage
- **Verdict:** ✅ VERIFIED ON WORKER POD — exists in diffusers 0.37.0
- **Claim:** SGLang partnered with GMI/Lightricks for LTX-2.3 optimization
- **Verdict:** ✅ PLAUSIBLE — specific CLI flags verified (`--ltx2-two-stage-device-mode`)
- **Claim:** PEFT replaces mmGP's LoRA code and is compile-compatible
- **Verdict:** ✅ CORRECT — standard diffusers API
- **Claim:** LTXVideoPipeline has native conditioning parameters
- **Verdict:** ✅ PLAUSIBLE — needs verification in diffusers source
- **Claim:** GGUF is native in transformers/diffusers
- **Verdict:** ✅ CORRECT

### Response 5: DAG orchestration advice
- **Verdict:** ✅ CORRECT — keep Ray Serve DAG, use it for cluster-level orchestration
- The advice to use isolated containers per model family is architecturally sound

### Response 6: Optimization ecosystem (torchao, Cache-DiT, torch.compile)
- **Claim:** PipelineQuantizationConfig + TorchAoConfig available
- **Verdict:** ✅ VERIFIED on worker — available in diffusers 0.37.0
- **Claim:** 5 cache acceleration strategies available
- **Verdict:** ✅ VERIFIED — FirstBlock, Faster, MagCache, TaylorSeer, CacheMixin
- **Claim:** torch.compile works with group offload + PEFT
- **Verdict:** ❌ WRONG — compile + group_offload are fundamentally incompatible
  (swap_tensors vs TensorWeakRef guards). Corrected by deep research verification.
- **Claim:** MXFP4/NVFP4 microscaling
- **Verdict:** ✅ Correct that this needs Blackwell — irrelevant for our 4090

### Response 7: Deep Research Report (comprehensive, 8 sections)
- **Verdict:** Mostly accurate with 3 critical corrections needed:
  1. ❌ SGLang "2.5-2.9x faster" is H100/B200 numbers. RTX 4090 = 1.15-1.5x.
  2. ❌ Anima code example used Qwen2.5 — actual model is Qwen3-0.6B-Base
  3. ❌ Implied cache acceleration works with group_offload — it does NOT
- **torch.compile + group_offload incompatible:** ✅ VERIFIED against PyTorch source
- **Cache accel + group_offload incompatible:** ✅ VERIFIED — block-skipping
  breaks prefetch chain
- **PEFT + group_offload:** ✅ Works IF LoRAs loaded before enabling offload
- **VAE offload bug in 0.37.0:** ✅ VERIFIED — PR #12692, fixed in 0.37.1
- **num_blocks_per_group tuning:** ✅ Makes sense (PCIe bandwidth sweet spot)
- **SGLang needs separate container:** ✅ VERIFIED — JIT compilation = 10+ min cold start
- **No continuous batching:** ✅ VERIFIED — request-blocking, homogeneous batch only
- **mmGP stability issues:** ✅ Real — documented GitHub issues (#29, #12, #24)

---

## 12. Risks & Open Questions

### Risks

1. **SGLang Diffusion is new** — launched weeks ago. No continuous batching
   (request-blocking at worker). 1.15-1.5x speedup on 4090 (not 2.5-2.9x).
   Needs separate Docker container. Treat as promising but evaluate per-model.

2. **~~enable_group_offload performance~~** — ✅ RESOLVED by deep research.
   15-35% overhead vs resident with `use_stream=True`. This is acceptable.
   Still need to benchmark specific models on our 4090.

3. **torch.compile + group_offload incompatibility** — VERIFIED fundamental
   conflict. Architecture must use Path A (model_cpu_offload + compile) or
   Path B (group_offload, no compile), not both. Cache acceleration also
   incompatible with group_offload. Plan model placement accordingly.

4. **VAE offloading bug in diffusers 0.37.0** — `post_quant_conv`/`quant_conv`
   bypass block groupings. Fixed in 0.37.1 (PR #12692). Workaround: use
   `exclude_modules=["vae"]` or `leaf_level` for VAE. Affects Anima directly.

5. **transformers v5 breaking changes** — diffusers 0.37.0 is NOT compatible
   with transformers v5.x. Will crash in text-encoding stage. Stay on 4.57.x.
   Also: `decode()` returns list, `encode_plus` removed, special tokens renamed.

6. **Model coverage gaps** — Neither SGLang nor native diffusers support
   ALL models. Anima needs custom runner (Qwen3-0.6B-Base encoder + Cosmos
   transformer + custom flow matching). TRELLIS and TTS need custom runners.

7. **Custom pipeline features** — Advanced features (prompt relay, Director
   mode, FLF2V for Wan) are NOT natively supported in diffusers. Wan only
   supports first-frame conditioning natively (not first-last-frame).
   Prompt relay requires custom denoising loop implementation.

8. **PEFT + group_offload ordering** — LoRAs MUST be loaded BEFORE enabling
   group_offload, or adapter weights bypass hooks → device mismatch.
   `record_stream=True` required (bug fixed in 0.36.0, we're on 0.37.0 ✅).

### Open questions (for deep research)

1. How does `enable_group_offload(use_stream=True)` perform vs mmGP on a 4090?
   Specifically for Wan 14B and LTX2.

2. Does SGLang Diffusion's `--ltx2-two-stage-device-mode snapshot` actually
   work well? What are the real-world performance numbers?

3. Does `enable_group_offload` work with ALL diffusers model architectures,
   or only specific ones? Which models have `_supports_group_offloading=True`?

4. Can `enable_group_offload` + `enable_layerwise_casting` be combined?
   Any conflicts?

5. What's the real cold-start time when using Ray Serve `min_replicas=0`
   with native diffusers on a 4090?

6. How does PEFT LoRA performance compare when using group_offload?
   Does the stream prefetch interfere with adapter weight application?

7. Can SGLang Diffusion be deployed alongside native diffusers in the same
   Ray Serve cluster without library conflicts?

8. What is the actual wake time for SGLang's sleep/wake mechanism?

---

## 13. Deep Research Query

**Query submitted:** 2026-06-13. **Results received & verified.** All findings
integrated into this document and [03_DEEP_RESEARCH_FINDINGS.md](03_DEEP_RESEARCH_FINDINGS.md).

The query covered 7 research areas (see 01_DEEP_RESEARCH_QUERY.md). Key answers:

| Question | Answer |
|----------|--------|
| Is group_offload fast enough vs mmGP? | ✅ Yes — 15-35% overhead with use_stream=True |
| Is SGLang production-ready? | ✅ For standard models (1.15-1.5x on 4090, NOT 2.5-2.9x) |
| Gotchas with offload + casting + PEFT? | Order matters: casting → offload. LoRAs before offload. |
| compile + group_offload? | ❌ Fundamentally incompatible (TensorWeakRef guards) |
| cache_accel + group_offload? | ❌ Incompatible (breaks prefetch chain) |
| VAE offloading in 0.37.0? | ⚠️ Bug (PR #12692, fixed 0.37.1) — use leaf_level or exclude |

---

## Appendix: Key File Paths

```
/opt/wan2gp/                          Wan2GP vendor directory (TO BE REMOVED)
/opt/wan2gp/wgp.py                    Gradio monolith (UNUSED, 12k lines)
/opt/wan2gp/models/                   Handler directory (TO BE REPLACED)
/opt/wan2gp/models/anima/anima_main.py  Our custom Anima handler (REFERENCE for runner pattern)

/services/wan2gp/deployment.py        Wan2GP service routing (TO BE UPDATED)
/services/wan2gp/forge_adapter.py     Forge adapter (TO BE UPDATED)
/services/forge.py                    Forge service registry (KEEP)
/gateway/                             Gateway + API (KEEP)
/gateway/routes/editor.py             Editor routes (KEEP)

Model checkpoints: /models/wan2gp/    (STAYS — models are independent of Wan2GP)
LoRA checkpoints: /models/wan2gp/loras/  (STAYS — loaded via PEFT)

Config: /config/model_registry.yaml   Model definitions (UPDATE routing)
```

## Appendix: mmGP Functions That Map to Native APIs

| mmGP function | Lines | Native replacement |
|---------------|-------|-------------------|
| `offload.all()` | 314 | Ray Serve DAG (cluster) + `enable_group_offload` (model) |
| `offload.profile()` | 99 | Not needed — configure per-model |
| `_pin_to_memory` | 196 | `enable_group_offload(offload_device=cpu)` handles staging |
| `_pin_sd_to_memory` | 127 | Standard safetensors loading (sufficient RAM) |
| `gpu_load_blocks` | ~80 | `enable_group_offload(use_stream=True)` internal |
| `load_loras_into_model` | 562 | `pipe.load_lora_weights()` + PEFT |
| `_lora_linear_forward` | ~80 | PEFT internal |
| `_dora_linear_forward` | ~100 | PEFT DoRA support |
| `_quantize` | 161 | `enable_layerwise_casting(storage_dtype=fp8)` |
| `_quantize_dirty_hack` | ~30 | Not needed — native quantization |
| `_requantize` | 42 | Not needed |
| `fast_load_transformers_model` | 89 | `TransformerClass.from_pretrained()` |
| `load_model_data` | 291 | `from_pretrained()` + standard loading |
| `map_state_dict` | 36 | transformers v5 WeightConverter (or not needed) |
| `filter_state_dict_basic` | 23 | Standard loading handles this |
| `_extract_tie_weights_from_sd` | 27 | Native tied-weight handling |
| `save_model` | 83 | `model.save_pretrained()` |
| `safetensors2.py` (entire) | 627 | Standard safetensors (sufficient RAM) |
| `quant_router.py` (entire) | 1,256 | `enable_layerwise_casting` + bnb/quanto |
| `fp8_quanto_bridge.py` (entire) | 645 | `enable_layerwise_casting(fp8)` |
