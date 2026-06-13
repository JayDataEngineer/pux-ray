# Native API Reference — mmGP Replacement Cheat Sheet

> Quick lookup for the diffusers/PEFT APIs that replace mmGP functionality.
> All APIs verified present in diffusers 0.37.0 on the production worker.

---

## VRAM Offloading

### Block-level async stream offloading (replaces mmGP core)

```python
# For models that don't fit entirely in VRAM
# Streams transformer blocks between CPU↔GPU with async CUDA prefetch
# Overhead: 15-35% vs resident with use_stream=True
# Overhead: 300-500% WITHOUT use_stream (basically as slow as sequential)

transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",       # "block_level" or "leaf_level"
    use_stream=True,                   # async CUDA stream prefetch (CRITICAL)
    record_stream=True,                # prevent premature memory reclamation
    num_blocks_per_group=None,         # tune: more blocks = fewer transfers, more VRAM
)
```

- `block_level`: groups consecutive transformer blocks together
- `leaf_level`: offloads at the finest granularity (individual leaf modules)
- `use_stream=True`: prefetches next group on background CUDA stream while computing current
- `record_stream=True`: prevents caching allocator from reclaiming prefetched tensors

**⚠️ ORDER MATTERS when combining with layerwise_casting:**
```python
# 1. Apply layerwise casting FIRST (sets up casting hooks on CPU weights)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn)

# 2. Apply group offload SECOND (wraps casting hooks with PCIe transfer hooks)
transformer.enable_group_offload(onload_device=torch.device("cuda"), use_stream=True)
```

**Tuning num_blocks_per_group (from deep research benchmarks):**

| Model | Recommended | Target VRAM |
|-------|------------|-------------|
| Wan 14B | `block_level`, num_blocks_per_group=2 | ~11.8 GB |
| FLUX.1-dev | `block_level`, num_blocks_per_group=3 | ~9.6 GB |
| LTX-Video (2B) | `leaf_level`, None (auto) | ~4.1 GB |
| Cosmos 32B | `block_level`, num_blocks_per_group=1 | ~19.5 GB |

**⚠️ VAE bug in diffusers 0.37.0:** `post_quant_conv` and `quant_conv`
bypass block groupings, causing device mismatch. Fixed in 0.37.1+.
Workaround: use `leaf_level` for VAE, or don't offload the VAE (use tiling instead).

**⚠️ Prefetch limit:** Native diffusers restricts prefetch to ONE group ahead
(hardcoded to prevent OOM during peak activations).

**⚠️ INCOMPATIBLE with cache acceleration:** Block-skipping cache strategies
(`apply_first_block_cache`, etc.) break the sequential prefetch chain. If
cache skips Blocks 2-N, those blocks' offload hooks never fire → prefetch
desync → device mismatch. Do NOT combine group_offload with cache acceleration.

**⚠️ INCOMPATIBLE with torch.compile:** `swap_tensors` conflicts with dynamo
`TensorWeakRef` guards → `RuntimeError: Cannot swap t1 because it has weakref`.
Use Path A (model_cpu_offload + compile_repeated_blocks) instead if you need
compilation. See 00_OVERVIEW.md §7.

**⚠️ VAE offloading bug in diffusers 0.37.0:** `post_quant_conv` and `quant_conv`
bypass block groupings (they're not in ModuleList/Sequential). `vae.decode()`
calls `_decode()` directly, bypassing `forward()` → hooks never fire → weights
stay on CPU → crash: `RuntimeError: Input type (CUDABFloat16Type) and weight
type (CPUBFloat16Type) should be the same`. Fixed in 0.37.1 (PR #12692).
**Workarounds for 0.37.0:**
```python
# Option 1: exclude VAE from group offload
pipe.enable_group_offload(onload_device="cuda", use_stream=True, exclude_modules=["vae"])

# Option 2: leaf_level for VAE (hooks individual convs correctly)
from diffusers.hooks import apply_group_offloading
apply_group_offloading(pipe.vae, onload_device="cuda", offload_type="leaf_level")
pipe.vae.enable_tiling()
```

### Pipeline-stage offloading (coarse, for multi-component pipelines)

```python
# Moves whole components (text_encoder → transformer → VAE) between CPU/GPU
# Each component moves to GPU only when needed, back to CPU when done
pipe.enable_model_cpu_offload()
```

- Simpler than group_offload
- Good when transformer fits in VRAM but full pipeline doesn't
- Synchronous (no stream overlap) but negligible overhead for stage-level swaps

### VAE memory management

```python
pipe.vae.enable_tiling()    # Overlapping tiles for large images/video
pipe.vae.enable_slicing()   # Process latent slices sequentially
```

---

## Quantization

### Layerwise weight casting (replaces mmGP int8/fp8 quantization)

```python
# Stores weights in FP8, computes in bf16
# ~50% VRAM reduction with minimal quality loss
# Automatically skips precision-critical layers (norm, embedding)

transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,    # FP8 storage format
    compute_dtype=torch.bfloat16,          # computation dtype
)
```

**Can be combined with `enable_group_offload`** — cast to FP8 for storage,
stream blocks with group offload, upcast to bf16 during compute.

**⚠️ Tensor subclass gotcha:** When using TorchAoConfig with FP8 weight-only
format, parameters become custom tensor subclasses. The `.data` setter only
replaces the outer wrapper, leaving internal quantization params on CPU →
device mismatch. Fixed by using `torch.utils.swap_tensors()` (diffusers 0.36+).

**⚠️ ORDER:** Apply `enable_layerwise_casting` FIRST, then
`enable_group_offload` SECOND. Reversing causes the casting hooks to wrap
already-transferred weights incorrectly.

---

## LoRA Management (PEFT integration)

### Loading LoRAs

```python
# Load a single LoRA
pipe.load_lora_weights("path/to/lora.safetensors", adapter_name="style_1")

# Load multiple LoRAs
pipe.load_lora_weights("style.safetensors", adapter_name="style")
pipe.load_lora_weights("detail.safetensors", adapter_name="detail")
```

### Dynamic adapter control

```python
# Activate specific adapters with independent weights
pipe.set_adapters(["style", "detail"], adapter_weights=[0.85, 0.4])

# Swap to different adapter
pipe.set_adapters(["style"], adapter_weights=[1.0])

# Scale adapter dynamically
pipe.set_adapters(["style"], adapter_weights=[0.5])  # half strength

# Deactivate all adapters (base model only)
pipe.unload_lora_weights()
```

### Fusion and cross-attention control

```python
# Fuse LoRA weights into base model (faster inference, no adapter overhead)
pipe.fuse_lora(adapter_names=["style"], lora_scale=0.85)

# Control which parts of the model get LoRA
pipe.load_lora_weights(
    "lora.safetensors",
    adapter_name="style",
    cross_attention_kwargs={"scale": 0.85},
)
```

**Key advantage over mmGP:** PEFT is compatible with `torch.compile`.
mmGP's monkey-patching (`_lora_linear_forward`) broke compilation graphs.

**⚠️ PEFT + group_offload gotcha:** When using LoRAs with group offloading,
the prefetch stream can bypass adapter weights, causing device mismatch
crashes. The offload manager must be aware of parallel adapter paths
($W_{base} \cdot x + \frac{\alpha}{r}(B \cdot A \cdot x)$). Set
`record_stream=True` to prevent the caching allocator from prematurely
reclaiming prefetched memory. This was buggy in diffusers 0.35.0 and
fixed in 0.36.0+.

---

## Standard Pipeline Loading

### Image generation

```python
from diffusers import FluxPipeline, AutoPipelineForText2Image

# Auto-detects the right pipeline class
pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")  # if it fits; otherwise enable_model_cpu_offload()
```

### Video generation

```python
from diffusers import WanPipeline, LTXVideoPipeline

# Wan 2.1
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B", torch_dtype=torch.bfloat16)

# LTX-Video
pipe = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
```

### Custom transformer loading (for models like Anima)

```python
from diffusers import CosmosTransformer3DModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Anima uses Qwen3-0.6B-Base as text encoder (NOT Qwen2.5!)
# The code block in the deep research report was wrong to use Qwen2.5.
# Loading Qwen2.5 weights → immediate semantic drift / image collapse.
text_encoder = AutoModelForCausalLM.from_pretrained(
    "circlestone-labs/Anima-Base-v1.0-Diffusers",
    subfolder="text_encoder",       # qwen_3_06b_base.safetensors
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="tokenizer")

transformer = CosmosTransformer3DModel.from_pretrained(
    "circlestone-labs/Anima-Base-v1.0-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
# Apply offload + casting (layerwise FIRST, group_offload SECOND)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn)
transformer.enable_group_offload(onload_device=torch.device("cuda"), use_stream=True, record_stream=True)
```

---

## Advanced Pipeline Features

### First/last frame conditioning (LTX-Video)

```python
# Image-to-video with conditioning frame
output = pipe(
    prompt="...",
    image=conditioning_image,         # first frame
    media_frame_number=0,              # [unverified] which frame to condition on
    strength=1.0,                      # [unverified] conditioning strength
    num_prefix_latent_frames=2,        # [unverified] boundary prefix length
    prefix_latents_mode="drop",        # [unverified] "drop" or "soft"
)
```
> Parameters marked [unverified] — need to confirm in diffusers source before using.

### Custom latents

```python
# Pass pre-generated latents directly
output = pipe(
    prompt="...",
    latents=custom_latents,            # [B, C, F, H, W]
    num_inference_steps=30,
)
```

### Latent denormalization (Qwen-Image VAE)

```python
# Qwen-Image VAE expects denormalized latents
latents_denorm = latents * vae.config.latents_std + vae.config.latents_mean
image = vae.decode(latents_denorm).sample
```

---

## torch.compile Integration

```python
# Now works with group_offload + PEFT (unlike mmGP)
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="max-autotune",
    fullgraph=False,
)
```

---

## SGLang Diffusion (alternative serving path)

**RTX 4090 realistic speedup: 1.15–1.5x** over native diffusers.
The "2.5–2.9x" numbers are H100/B200 enterprise benchmarks.
SPEED (Spectral Progressive Resolution) still gives >2x on consumer HW.

**⚠️ MUST run in separate Docker container** — sgl-kernel/flashinfer require
precompiled CUDA extensions. Ray runtime_env pip install = 10+ min cold start.

**⚠️ No continuous batching** — request-blocking at worker boundary.
Only homogeneous batching at request start (same resolution, step count).

```bash
# Install (in SEPARATE container, not via Ray runtime_env)
pip install "sglang[diffusion]"

# Serve a model
sglang serve --model-path Qwen/Qwen-Image --port 30010

# LTX with two-stage mode (snapshot recommended for 24GB VRAM)
sglang serve --model-path Lightricks/LTX-2.3 \
    --ltx2-two-stage-device-mode snapshot

# Generate (CLI)
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A sunset" --save-output

# Sleep/wake VRAM management
curl -X POST http://localhost:30010/release_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{"tags": ["weights", "cache"]}'     # VRAM → 250-400MB
curl -X POST http://localhost:30010/resume_memory_occupation  # wake in 0.48-0.60s
```

```python
# Call via OpenAI-compatible API
import openai
client = openai.Client(base_url="http://localhost:30010/v1", api_key="none")
response = client.images.generate(
    model="qwen-image",
    prompt="A sunset over mountains",
)

# LoRA management
# POST /v1/loras/load   {"lora_name": "...", "path": "..."}
# POST /v1/loras/unload {"lora_name": "..."}
```

### LTX-2 two-stage device modes

| Mode | Peak VRAM | Total Latency | Recommendation |
|------|-----------|---------------|----------------|
| `original` | 9.4 GB | 154.6s | ❌ Slow (sequential swap) |
| `snapshot` | 13.8 GB | 114.0s | ✅ **Sweet spot for 24GB** |
| `resident` | 21.6 GB | 75.7s | ⚠️ Fast but risks OOM |

---

## Quick Decision Matrix

| Situation | Path | API to use |
|-----------|------|-----------|
| Model fits VRAM, want max speed | **A** | `pipe.enable_model_cpu_offload()` + `pipe.transformer.compile_repeated_blocks()` + cache accel |
| Model doesn't fit VRAM | **B** | `layerwise_casting()` THEN `enable_group_offload(use_stream=True, record_stream=True)` |
| Standard model, want SGLang kernels | **C** | Separate container, `sglang serve` |
| Need LoRAs on Path B | **B** | `load_lora_weights()` FIRST, THEN `enable_group_offload()` |
| Need cache accel on Path B | ❌ | **IMPOSSIBLE** — cache breaks prefetch chain. Use Path A instead |
| Need compile on Path B | ❌ | **IMPOSSIBLE** — swap_tensors conflicts with dynamo. Use Path A |
| VAE OOMs on decode | Any | `pipe.vae.enable_tiling()` (don't group-offload VAE in 0.37.0) |
| Niche/custom model | Custom | Runner calling diffusers/library directly |

### Optimization Compatibility Matrix

| | group_offload | torch.compile | cache_accel | model_cpu_offload |
|---|---|---|---|---|
| **group_offload** | — | ❌ | ❌ | redundant |
| **torch.compile** | ❌ | — | ✅ | ✅ |
| **cache_accel** | ❌ | ✅ | — | ✅ |
| **model_cpu_offload** | redundant | ✅ | ✅ | — |
| **layerwise_casting** | ✅ | ✅ | ✅ | ✅ |
| **PEFT LoRAs** | ✅ (load first) | ✅ | ✅ | ✅ |

---

## Optimization Layers BEYOND mmGP (net-new performance gains)

These optimizations are things mmGP never provided — and in some cases mmGP
actively PREVENTED. Stacking them on top of group offload makes native diffusers
potentially FASTER than mmGP.

### Layer 1: Cache Acceleration (20–165% speedup — mmGP CANNOT do this)

**⚠️ INCOMPATIBLE with `enable_group_offload`** — block-skipping breaks the
sequential prefetch chain. Only usable with Path A (model_cpu_offload) or
when model is fully resident in VRAM.

```python
# VERIFIED: available in diffusers 0.37.0
# Five different cache strategies — pick one per model
# DO NOT combine with enable_group_offload(use_stream=True)

from diffusers import (
    apply_first_block_cache,    # FirstBlockCacheConfig
    apply_faster_cache,         # FasterCacheConfig
    apply_mag_cache,            # MagCacheConfig
    apply_taylorseer_cache,     # TaylorSeerCacheConfig
)

# First-Block Cache: monitors first transformer block output across steps.
# If nearly identical between step T and T-1, skips deeper blocks.
apply_first_block_cache(pipe.transformer)

# Faster Cache: adaptive caching with quality preservation
apply_faster_cache(pipe.transformer)
```

**How it works:** Diffusion models perform redundant computation across
denoising steps. Cache strategies cache residuals from previous steps and
skip recomputation on blocks that haven't changed significantly.

**Why it breaks group_offload:** If cache skips Blocks 2-N, those blocks'
offload hooks never fire. The prefetch stream desyncs → next-group
prefetch never triggers → device mismatch crash:
`"some layers were not executed during the forward pass"`.

**Why mmGP can't do this:** mmGP manages MEMORY, not COMPUTATION. It has no
awareness of what the transformer is computing — it only moves weights.
Step caching requires intercepting the forward pass logic, which mmGP's
hook architecture doesn't support.

### Layer 2: torch.compile (~1.5x speedup — mmGP BROKE this)

**⚠️ CRITICAL: torch.compile is INCOMPATIBLE with enable_group_offload.**
You can use ONE or the OTHER, not both.

```python
# VERIFIED: torch.compile available (PyTorch 2.10)
# ONLY usable when model fits in VRAM WITHOUT group_offload
# PEFT is compile-compatible; group_offload is NOT

# ✅ DO THIS (model fits in VRAM):
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="max-autotune",    # or "reduce-overhead" for lower compile time
    fullgraph=False,
)

# ❌ DO NOT DO THIS (will crash):
pipe.transformer.enable_group_offload(onload_device="cuda", use_stream=True)
pipe.transformer = torch.compile(pipe.transformer)  # FAILURE
# torch.dynamo guards conflict with swap_tensors mechanism
```

**Why mmGP broke this:** mmGP's `load_loras_into_model` monkey-patches
`module.forward` with Python wrappers (`_mm_lora_linear_forward`). This
creates dynamic dispatch paths that torch.compile cannot trace → graph breaks
→ no compilation benefit. PEFT (the native path) modifies weights in-place,
so the forward graph stays static → compile works with PEFT.

**Why group_offload breaks this:** `swap_tensors` used by group_offload
refuses to operate on tensors with active dynamo weak references (TensorWeakRef).
Dynamo guard failures occur immediately. This is a fundamental architecture
conflict, not a bug.

**Workaround — model_cpu_offload + compile_repeated_blocks:**
```python
# ✅ DO THIS: coarse offload (whole-model swap, no mid-forward swap)
pipe.enable_model_cpu_offload()  # moves ENTIRE DiT to GPU before execution

# Then compile only the repeated DiT blocks (not the shell, not VAE/text encoder)
pipe.transformer.compile_repeated_blocks(fullgraph=True)
# This works because model_cpu_offload doesn't use swap_tensors mid-forward
```

**Regional compilation tip:** `compile_repeated_blocks()` compiles only
transformer block submodules, not the parent shell. Reduces cold-start
from ~67s to ~10s while keeping the ~1.5x speedup on compute-heavy parts.

### Layer 3: torchao Quantization (50% VRAM cut + speed boost)

```python
# VERIFIED: PipelineQuantizationConfig + TorchAoConfig available in diffusers 0.37.0
# torchao itself: needs `pip install torchao` (not currently installed)

from diffusers import FluxPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig())}
)
pipe = FluxPipeline.from_pretrained("...", quantization_config=quant_config)
```

**vs `enable_layerwise_casting`:** torchao is more integrated with torch.compile
(both are PyTorch-native, designed to compose). `enable_layerwise_casting` is
simpler (no extra dep) but may not compose as cleanly with compilation.

**4090-compatible formats:** FP8 (E4M3), INT8, INT4, NF4 — all supported on
Ada Lovelace. NOT supported: MXFP4/NVFP4 (needs Blackwell B200/B300).

### The Stacked Optimization Pyramid

```
         ┌─────────────────────┐
         │  Cache Acceleration  │  20-165% speedup (skip redundant steps)
         │  (5 strategies)      │  ← mmGP CANNOT do this
         ├─────────────────────┤
         │  torch.compile       │  ~1.5x speedup (kernel fusion)
         │  (regional, DiT)     │  ← mmGP BROKE this
         ├─────────────────────┤
         │  torchao quant       │  50% VRAM cut + speed boost
         │  (FP8/INT8)          │  ← mmGP had its own, less integrated
         ├─────────────────────┤
         │  PEFT LoRAs          │  dynamic adapters, compile-compatible
         │                      │  ← mmGP's monkey-patching broke compile
         ├─────────────────────┤
         │  group_offload       │  async stream VRAM offloading
         │  (use_stream=True)   │  ← mmGP's core feature, now native
         └─────────────────────┘
```

**Each layer is independently valuable. Stacked together, they make native
diffusers FASTER than mmGP ever was — because mmGP only did the bottom
layer and actively prevented the top three.**

---

## What's GONE (mmGP APIs no longer needed)

```python
# ❌ REMOVE — replaced by enable_group_offload
from mmgp import offload
offload.all(pipe, profile_type.LowRAM_LowVRAM)
offload.fast_load_transformers_model(...)

# ❌ REMOVE — replaced by enable_layerwise_casting
offload._quantize(model, weights=qint8)

# ❌ REMOVE — replaced by PEFT
offload.load_loras_into_model(model, lora_path)

# ❌ REMOVE — replaced by standard from_pretrained
offload.load_model_data(model, file_path)
offload.map_state_dict(sd, rules)
```
