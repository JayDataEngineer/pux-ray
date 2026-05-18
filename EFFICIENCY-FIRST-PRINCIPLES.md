# Efficiency-First Principles

Derived from studying Wan2GP + mmgp 3.7.6 architecture. Every decision trades VRAM or time — never both.

**Goal: Maximum output per watt of GPU compute. VRAM + wall-clock time are the only metrics that matter.**

---

## 0. The Fundamental Architecture

**mmgp provides primitives. The model code provides intelligence.**

mmgp is NOT an optimization engine that you "configure." It is a library of memory primitives — swap, pin, quantize, prefetch. Every optimization is implemented at the model level, using these primitives as building blocks.

The relationship is like React and a React application:
- **mmgp** = React (`useState`, `useEffect`, `useMemo`) — generic primitives
- **Model handler** = The full application — where the real complexity lives

Adding a new model is not "pass a pipe dict." It is writing a complete optimization application that understands:
- The model's inference pipeline stages and their VRAM requirements
- Which components can coexist in VRAM (co-tenancy)
- When to cache intermediate results and when to clear them
- Per-component precision requirements
- Model-specific optimizations (CUDA graphs, token budgets, spatial caches)
- Explicit CPU offload of intermediates between pipeline stages
- When to override mmgp's defaults because the model knows better

**The pipe dict is the first line of the handler, not the whole thing.**

---

## 1. Module-Level VRAM Management (mmgp) — The Primitives

mmgp hooks every `nn.Module` in a pipeline dict and intercepts forward passes to swap modules between CPU RAM and GPU VRAM at the individual layer level. This is the primary primitive the model handler builds on.

### How It Works

```python
from mmgp import offload

pipe = {
    "text_encoder": clip,          # ~500MB
    "text_encoder_2": t5,          # ~5GB
    "transformer": wan_model,      # ~14GB
    "vae": vae_decoder,            # ~500MB
}

offload.profile(pipe, profile_no=2)  # Balanced profile
```

After this call, mmgp provides these primitives for the handler to use:
1. Automatic CPU↔GPU swapping via forward hooks (the handler stages the pipeline)
2. On-the-fly quantization to INT8 (handler decides which components to quantize)
3. Pinned RAM for DMA transfers (handler decides what to pin)
4. Async prefetch between sequential layers (automatic for ModuleList)
5. Budget enforcement (handler sets per-component limits)

### Budget System

Controls how much VRAM each model component can occupy at any instant:

```python
offload.profile(
    pipe,
    profile_no=2,
    budgets={"transformer": 250, "text_encoder": 250, "*": 3000},
    vram_safety_coefficient=0.9,  # never use more than 90% of VRAM
)
```

- **`250`** = 250MB. The transformer (14GB) is loaded in 250MB windows — individual layers streamed in and out
- **`"70%"`** = percentage of total VRAM. Adapts to any GPU
- **`"*"`** = default for unnamed models
- **Real usage is 2x budget** — one batch being computed, one being prefetched

### Profiles

| Profile | Name | Budgets | Pinned RAM | Quantize | Use Case |
|---------|------|---------|------------|----------|----------|
| 1 | HighRAM_HighVRAM | Unlimited | Yes | No | 48GB+ GPUs |
| 2 | HighRAM_LowVRAM | 3000MB default | Yes | Yes | 24GB GPUs (ours) |
| 3 | LowRAM_HighVRAM | "70%" | No | Yes | Limited RAM |
| 4 | LowRAM_LowVRAM | 1000MB | No | Yes | 12GB GPUs |
| 5 | VerylowRAM_LowVRAM | 1000MB aggressive | No | Yes | 8GB GPUs |

**Our hardware (RTX 4090 24GB + 64GB RAM): Profile 2 with `pinnedMemory=["transformer"]`.**

### Pinned Memory

```python
offload.profile(pipe, pinnedMemory=["transformer"])
```

- Locks model weights in non-pageable RAM (requires `mlock`-equivalent)
- GPU↔CPU transfers become DMA copies instead of pageable memcpy — ~3-5x faster
- Costs ~50% more RAM (14GB transformer → 21GB pinned RAM)
- **Critical for repeated inference**: first run loads from disk to pinned RAM, subsequent runs skip disk I/O entirely

### Co-Tenancy

```python
coTenantsMap = {
    "text_encoder": ["vae"],
    "vae": ["text_encoder"],
}
```

Models listed as co-tenants are allowed to share VRAM simultaneously. Without this, mmgp unloads model A before loading model B. With co-tenancy, both can be in VRAM if budget allows.

**When to use:** Text encoder + VAE are both small (~500MB each). Keeping both loaded avoids a load/unload cycle between encoding and decoding.

### Async Transfers

```python
asyncTransfers=True  # default
```

While the current transformer layer computes on GPU, mmgp asynchronously copies the next layer from CPU→GPU. Hides transfer latency behind compute.

### Production Configuration (Our Hardware)

```python
offload.profile(
    pipe,
    profile_no=2,                           # Balanced: quantize + budget
    quantizeTransformer=False,              # Pre-quantized weights
    budgets={"transformer": 250, "text_encoder": 250, "*": 3000},
    pinnedMemory=["transformer"],           # Fast reload on repeated runs
    asyncTransfers=True,
    vram_safety_coefficient=0.9,            # 10% VRAM headroom
    perc_reserved_mem_max=0.5,              # Use up to 50% of RAM for pinning
    coTenantsMap={"text_encoder": ["vae"]},  # Keep TE+VAE loaded together
)
```

---

## 2. Step Skipping (MagCache / TeaCache)

**The biggest wall-clock time optimization.** Diffusion models run 20-50 denoising steps. Most steps produce near-identical outputs. Step skipping skips the expensive transformer forward pass when the output change is below a threshold.

### MagCache (Magnitude Cache)

Tracks the magnitude ratio between consecutive transformer outputs. When the ratio is close to 1.0, the output hasn't changed significantly — skip the forward pass and reuse the cached output.

```python
# Per-step decision:
accumulated_ratio *= current_mag_ratio     # How much has output changed since last compute
accumulated_err += abs(1 - accumulated_ratio)  # Cumulative drift

if accumulated_err < threshold and accumulated_steps <= K:
    skip_forward = True    # Reuse cached output
else:
    accumulated_err = 0    # Reset, compute this step
    accumulated_steps = 0
```

**Key parameters:**
- **`threshold`** (default 0.12): Error tolerance. Lower = higher quality, fewer skips
- **`K`** (default 2): Max consecutive skips. Prevents drift accumulation
- **`retention_ratio`** (default 0.2): Fraction of early steps that always compute (most change happens early)
- **`mag_ratios`**: Pre-computed per-model ratios from calibration runs. Interpolated to match requested step count

**Typical speedup: 1.5-3x with <0.5% quality loss.**

### TeaCache (Time-Embedding Cache)

Alternative to MagCache that uses time-step embedding similarity instead of output magnitude:

```python
# Polynomial fit on time embedding difference
coefficients = [-5784.5, 5449.5, -2249.3, ...]
cache_threshold = 0.08  # Lower = more conservative

# If time embedding change < threshold, reuse cached output
if similarity > (1 - cache_threshold):
    skip_forward = True
```

**Choosing between them:**
- **MagCache**: Better for video models (Wan, Hunyuan). More predictable skip patterns
- **TeaCache**: Better for image models (Flux, SD3). Simpler calibration

### Implementation Pattern

```python
class StepSkipCache:
    cache_type: str           # "mag" or "tea"
    mag_ratios: np.ndarray    # Pre-computed magnitude ratios
    accumulated_ratio: list   # Per-stream (conditional/unconditional)
    accumulated_err: list     # Per-stream cumulative error
    accumulated_steps: list   # Per-stream consecutive skip count
    magcache_thresh: float    # Error threshold
    magcache_K: int           # Max consecutive skips
    start_step: int           # Always compute first N steps
    skipped_steps: int        # Counter for logging
```

**In the denoising loop:**
```python
for step_no, t in enumerate(timesteps):
    if skip_cache and step_no > skip_cache.start_step:
        ratio = skip_cache.mag_ratios[step_no * 2 + stream_id]
        skip_cache.accumulated_ratio[stream_id] *= ratio
        skip_cache.accumulated_err[stream_id] += abs(1 - accumulated_ratio)

        if accumulated_err < threshold and consecutive_skips < K:
            x_list[i] = cached_output  # Skip transformer forward
            continue

    # Full forward pass
    x_list[i] = transformer(x, t, context, ...)
    cached_output = x_list[i]
```

---

## 3. On-the-Fly Quantization

mmgp quantizes models from BF16 → INT8 during the `offload.all()` call. Uses HuggingFace `optimum.quanto`.

```python
from optimum.quanto import quantize, qint8, freeze

quantize(model, weights=qint8)  # Replace Linear layers with QLinear
freeze(model)                    # Convert weights to int8 tensors
```

**Impact:**
- **VRAM**: 14GB transformer → ~7GB. 50% reduction
- **RAM**: Same 50% reduction (critical for 64GB systems running multiple models)
- **Quality**: <1% degradation for INT8. FP8 is even less
- **Speed**: Slightly slower per-op (dequantization overhead), but overall faster because more fits in VRAM

**Supported types:**
- `qint8`: Default. Best quality/speed tradeoff
- `qfloat8`: Lower precision, faster on H100/4090 FP8 hardware
- `qint4`: Extreme compression. Use only for text encoders

**Rule:** Always quantize the transformer. Optionally quantize text encoders if RAM is tight. Never quantize the VAE (precision-sensitive).

---

## 4. CUDA Graphs

Captures an entire inference pass as a static CUDA graph. On replay, kernel launch overhead is eliminated.

```python
class CUDAGraphRunner:
    def capture(self, *inputs):
        self.static_inputs = [t.detach().clone() for t in inputs]
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                self.output = self.fn(*self.static_inputs)
        self.graph.replay()  # Warmup

    def replay(self, *inputs):
        for i, t in enumerate(inputs):
            self.static_inputs[i].copy_(t)
        self.graph.replay()
        return self.output
```

**Constraints:**
- All tensor shapes must be identical between capture and replay
- No dynamic control flow, no data-dependent branching
- Works best for autoregressive generation (same shape every token)

**Use cases in our stack:**
- LLM token generation (same KV cache shape per token)
- TTS spectrogram generation
- Any fixed-shape repeated forward pass

**Speedup: 10-30% for small models, 5-15% for large models** (kernel launch overhead is proportionally smaller for large ops).

---

## 5. Text Encoder Caching

Text embeddings are expensive (T5 XXL: ~2s per prompt) and often repeated across requests.

```python
class TextEncoderCache:
    max_size_mb: float = 100  # LRU eviction above this

    def encode(self, encode_fn, prompts, device=None):
        missing = [p for p in prompts if hash(p) not in self._entries]
        if missing:
            results = encode_fn(missing)       # Batch encode
            self._insert_batch(missing, results)  # Move to CPU, store
        return [self._get(p, device) for p in prompts]  # Move to device
```

**Design:**
- Embeddings stored on CPU (not GPU) to avoid VRAM waste
- Moved to GPU only when needed, then released
- LRU eviction when total size exceeds budget
- Hash-based lookup — identical prompts hit cache

**When this matters:** Batch processing, video generation (same prompt, different frames), interactive loops.

---

## 6. The Pipeline Dict Pattern

Every model is decomposed into a dict of `nn.Module` components:

```python
pipe = {
    "transformer": wan_model,        # Main diffusion model
    "text_encoder": clip,            # CLIP text encoder
    "text_encoder_2": t5,            # T5 text encoder
    "vae": vae_decoder,              # VAE decoder
}
```

This is not just organization — it's the fundamental unit of VRAM management. mmgp treats each entry as an independent memory object with its own budget, quantization settings, and pinning policy.

**Decomposition rules:**
1. Always separate the transformer from text encoders and VAE
2. If a model has two text encoders (CLIP + T5), keep them as separate entries
3. Any component >1GB should be its own entry (budget control)
4. Shared components (same CLIP across models) can use the same dict entry

**Our deployment uses this in `_apply_mmgp_profile()`:**
```python
pipe, co_tenants = self._unwrap_pipe(pipe_wrapper)
offload.profile(pipe, profile_no=MMGP_PROFILES["balanced"],
                budgets={"transformer": 250, "text_encoder": 250, "*": 3000})
```

---

## 7. Inference Pipeline Structure

The optimal inference flow for a diffusion model:

```
1. TEXT ENCODING (CPU → GPU → CPU)
   ├─ Tokenize prompt
   ├─ Load text_encoder to GPU
   ├─ Compute embeddings
   ├─ Cache embeddings to CPU
   └─ Unload text_encoder

2. DENOISING LOOP (GPU, transformer streamed)
   ├─ Load first N transformer layers to GPU
   ├─ For each timestep:
   │   ├─ Check MagCache: should we skip?
   │   ├─ If compute: stream layers in/out via mmgp budget
   │   ├─ If skip: reuse cached output
   │   └─ Async prefetch next layer
   └─ Unload transformer

3. VAE DECODING (GPU)
   ├─ Load VAE to GPU
   ├─ Decode latents → pixels
   └─ Unload VAE

4. POST-PROCESSING (CPU)
   └─ Encode to video/image format
```

**Key insight:** Only one pipeline stage is active at a time. Text encoder, transformer, and VAE never need to be in VRAM simultaneously. This is why mmgp's module-level swapping works — the pipeline is naturally sequential.

---

## 8. Mixed Precision Strategy

Not all precision is equal. Different pipeline stages need different precision.

| Component | Precision | Why |
|-----------|-----------|-----|
| Transformer | BF16 or INT8 | Main compute. INT8 saves VRAM with minimal quality loss |
| Text Encoder | BF16 or INT8 | Embeddings are robust to quantization |
| VAE Decoder | FP32 | Latent→pixel conversion is precision-sensitive |
| Time Embedding | FP32 | Sinusoidal embeddings lose structure in lower precision |
| Attention (during graph capture) | BF16 | FP32 attention is unnecessary and 2x slower |

**Implementation:**
```python
handler.load_model(
    ...,
    dtype=torch.bfloat16,       # Transformer
    VAE_dtype=torch.float32,    # VAE
    text_encoder_quantization="int8",  # Text encoder
)
```

---

## 9. Memory Safety

Out-of-memory kills the entire process. Prevention is mandatory.

```python
vram_safety_coefficient=0.9    # Never use more than 90% of VRAM
perc_reserved_mem_max=0.5      # Never pin more than 50% of RAM
```

**Before every inference call:**
```python
torch.cuda.empty_cache()                    # Release cached allocations
gc.collect()                                # Collect Python garbage
offload.flush_torch_caches()                # mmgp's internal cache flush
```

**After model switch:**
```python
offloadobj.release()                        # Release all mmgp hooks
offloadobj.unload_all()                     # Move everything to CPU
del offloadobj                              # Python GC
offload.flush_torch_caches()                # Final cleanup
gc.collect()
torch.cuda.empty_cache()
```

**The 90% rule:** Always keep 10% VRAM free for CUDA internal allocations (workspace memory, kernel launches, cuDNN). mmgp's `vram_safety_coefficient` enforces this automatically.

---

## 10. Model Switching

When switching between models on a single GPU:

1. **Unload current model** (release mmgp hooks, flush caches)
2. **Check if new model fits** (compare VRAM budget vs. free VRAM)
3. **If it doesn't fit, unload everything** (full `gc.collect()` + `torch.cuda.empty_cache()`)
4. **Load new model with mmgp** (quantize, pin, install hooks)
5. **First inference warmup** (mmgp profiles memory usage on first forward pass)

**Time budget for switching:**
- With pinned RAM: ~2-5 seconds (DMA copy from pinned RAM)
- Without pinned RAM: ~10-30 seconds (disk → RAM → GPU)
- With model already in RAM: ~5-10 seconds (RAM → GPU)

---

## 11. The Data Flow — Disk → GPU

mmgp is a 4-stage pipeline from disk to GPU compute. Each stage depends on the previous:

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────┐
│ safetensors2 │────→│   _pin_to_memory │────→│ gpu_load_    │────→│ forward │
│              │     │                  │     │ blocks()     │     │         │
│ mmap + lazy  │     │ torch.empty(     │     │ p.to("cuda") │     │ compute │
│ load tensors │     │   pin_memory=    │     │ non_blocking │     │         │
│ on demand    │     │   True)          │     │              │     │         │
└──────────────┘     └──────────────────┘     └──────────────┘     └─────────┘
     STAGE 1              STAGE 2                 STAGE 3           STAGE 4
    Disk → RAM         RAM → Pinned RAM      Pinned RAM → GPU     GPU compute
```

**Stage 1 — safetensors2**: Memory-maps the weight file. Weights are NOT loaded into RAM. The OS pages them in on demand via virtual memory. This is why mmgp can "load" a 14GB model without allocating 14GB of RAM upfront. Writable tensor mode allows on-the-fly quantization to modify weights in-place without doubling memory.

**Stage 2 — Pin to memory**: Consolidates thousands of small tensors into a few large (~256MB) contiguous pinned buffers. Pinned memory = non-pageable = the GPU can DMA directly from it. Uses `torch.utils.swap_tensors` to redirect module parameters into these big buffers.

**Stage 3 — gpu_load_blocks**: When a forward hook fires, copies weights from pinned RAM to GPU via `p.to("cuda", non_blocking=True)`. Because source is pinned, this is a DMA transfer — no CPU involvement. The `non_blocking` flag on a separate CUDA stream enables async prefetch of the next layer while current layer computes.

**Stage 4 — Forward**: The actual computation. By this point weights are on GPU.

---

## 12. Handler Intelligence — Where the Real Work Lives

Every model handler is a full optimization application. The pipe dict is the first line, not the whole thing. Two examples that prove this:

### IndexTTS2 (Autoregressive TTS)

The handler decomposes into 6 components, but the real intelligence is in the model code:

```python
# Model-specific optimization: CUDA graph engine for autoregressive GPT
self.gpt = offload.fast_load_transformers_model(...)  # mmap loading
self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)  # Pre-allocate KV

# Token budget calculated from audio duration (not generic!)
sound_tokens_per_second = (mel_sr / mel_hop) / _MEL_TOKENS_PER_SOUND_TOKEN
cg_generation_tokens = max(min_cg_sound_tokens, capped_segment_sound_tokens)

# Deferred execution — GPT output → CPU → s2mel → CPU → vocoder
defer_s2mel = bool(generation_kwargs.pop("defer_s2mel", False))
ref_mel_cpu = ref_mel.detach().cpu().contiguous() if defer_s2mel else None

# Override mmgp budget when CUDA graph engine manages the transformer
pipe["transformer"]._budget = 0  # "Don't budget-limit this, I manage it myself"
```

The handler contains: CUDA graph capture, token budget estimation from audio duration, deferred execution between pipeline stages, speaker/emotion vector caching, per-component precision tagging, and explicit override of mmgp's budget system.

### TRELLIS (3D Generation)

Not a diffusion pipeline — 8 separate models run in sequence. The handler does ALL VRAM staging manually:

```python
# Explicit spatial cache clearing between stages (mmgp doesn't know about these)
shape_slat._spatial_cache = {}
torch.cuda.empty_cache()

# Manual intermediate movement between pipeline stages
tex_slat = tex_slat.to('cpu')
torch.cuda.empty_cache()

# Precision conversion at decode time
if shape_dec.dtype != torch.float16:
    shape_dec.convert_to_fp16()

# Component-specific exclusion from mmgp (rembg stays float32, outside pipe)
if dtypes == {torch.float32} and k == "rembg":
    continue  # BiRefNet stays float32, outside mmgp

# Patching mmgp for model-specific needs
_st2._map_to_dtype.setdefault("C64", _torch.complex64)  # complex dtype support
```

The handler contains: spatial cache management, per-stage VRAM clearing, per-component precision, dtype patching, co-tenancy for 8 sequential models, and explicit float32 exclusion for the background removal model.

### What This Means

Adding a new model requires understanding:
1. The model's inference pipeline stages (what runs when, in what order)
2. Each stage's VRAM requirements
3. Which stages can overlap (co-tenancy) vs. must be sequential
4. What hidden state each component caches between calls
5. Per-component precision requirements
6. Whether the model needs CUDA graphs, token budgets, or other specialized optimizations
7. Where to explicitly override mmgp because the model knows better

There is no shortcut. Each model is a new application.

---

## 13. Model Compatibility with mmgp

| Model Type | Compatibility | Why |
|-----------|--------------|-----|
| Standard diffusion (Wan, Hunyuan, Flux) | Graceful | Sequential encode→diffuse→decode, standard nn.Modules |
| Autoregressive LLM (Qwen, GPT) | Graceful | Standard transformer blocks, mmgp detects towers |
| Multi-stage non-diffusion (TRELLIS) | Works but handler must manage | 8 sequential models, spatial caches, sparse CUDA kernels |
| Autoregressive + diffusion hybrid (IndexTTS) | Works with overrides | GPT gets `_budget=0` for CUDA graphs, deferred execution between stages |
| Custom CUDA kernels (spconv, flex_gemm) | Fragile | Internal GPU state that mmgp doesn't know about |
| Models with hidden caches | Fragile | Spatial caches, workspace buffers — not nn.Parameters |

**Rule of thumb:** If a model stores GPU state in anything other than `nn.Parameter` or `nn.Buffer`, the handler must manually manage that state around mmgp's swaps.

---

## 14. Plan — What We're Building

### What mmgp gives us (keep)
- safetensors2 mmap loading (no upfront RAM allocation)
- Forward hook module swapping (CPU↔GPU automatic)
- On-the-fly INT8 quantization via quanto
- Pinned memory + DMA transfers
- Async prefetch between sequential layers

### What mmgp does NOT give us (build ourselves)
- MagCache step skipping (buried in Wan model code)
- CUDA graph kit for autoregressive generation (Wan2GP-specific)
- Text encoder caching (buried in Wan model code)
- Deferred execution helpers (every handler copy/pastes this)
- Per-component precision management (every handler ad-hocs this)
- Handler contract (Wan2GP has 8 methods, we need 2)

### Phase 1: Extract reusable utilities

```
services/efficiency/
├── magcache.py          ← StepSkipCache extracted from Wan model code
├── cudagraph.py          ← AutoRegressiveCudaGraphKit extracted + simplified
├── text_encoder_cache.py ← TextEncoderCache extracted
├── deferred.py           ← Helper for stage→CPU→next stage pattern
└── precision.py          ← Per-component dtype tagging + casting helpers
```

Standalone utilities. No mmgp dependency. Any model handler imports them.

### Phase 2: Handler contract

```python
@dataclass
class ModelSpec:
    name: str
    weights: Path
    components: dict[str, nn.Module]      # The pipe dict
    co_tenants: dict[str, list[str]]       # Who can coexist
    precision: dict[str, torch.dtype]      # Per-component dtype
    quantize: list[str]                    # Which to INT8
    pin: list[str]                         # Which to pinned RAM
    budgets: dict[str, int]                # Per-component MB limits
    step_skip: str | None                  # "magcache" or None

def load(spec: ModelSpec) -> tuple:
    """Load model, return (pipeline_object, spec)"""
    ...

def infer(pipeline, payload: dict) -> dict:
    """Run inference — THIS IS THE HANDLER APPLICATION"""
    ...
```

The `infer()` function IS the handler. It contains all model-specific optimization logic. No shortcuts.

### Phase 3: Migrate handlers

One handler at a time. Start simple, prove the pattern.

---

## Checklist: Applying These Principles

For every new GPU service:

- [ ] Study the model's inference pipeline — how many stages, what runs when
- [ ] Identify hidden state (caches, workspaces, spatial data) beyond nn.Parameters
- [ ] Decompose model into `pipe` dict with named components
- [ ] Define co-tenancy — which components can share VRAM
- [ ] Set per-component precision (transformer BF16/INT8, VAE FP32, time emb FP32)
- [ ] Apply `offload.profile()` with appropriate budget
- [ ] Implement explicit cache clearing between pipeline stages
- [ ] Implement deferred execution — move intermediates to CPU between stages
- [ ] Implement MagCache/TeaCache if diffusion model
- [ ] Implement CUDA graphs if autoregressive with fixed shapes
- [ ] Cache text encoder outputs if repeated prompts expected
- [ ] Set `vram_safety_coefficient=0.9`
- [ ] Add `gc.collect()` + `torch.cuda.empty_cache()` between stages
- [ ] Test with `torch.cuda.memory_allocated()` to verify VRAM stays under budget
- [ ] Profile wall-clock time per stage: encoding, diffusion, decoding, post-processing
- [ ] Override mmgp defaults (`_budget = 0`, `_model_dtype`) where the model knows better
