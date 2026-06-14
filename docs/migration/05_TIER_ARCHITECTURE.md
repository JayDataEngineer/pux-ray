# Tier Architecture & Adaptive Optimization

> **Status:** DESIGN — agreed upon, pending implementation
> **Date:** 2026-06-14
> **Principle:** User picks WHAT they want. System figures out HOW to deliver it.

---

## Table of Contents

1. [Product Tiers](#1-product-tiers)
2. [Component-Type Rules](#2-component-type-rules)
3. [Adaptive Runtime](#3-adaptive-runtime)
4. [Format Selection](#4-format-selection)
5. [Model Registry Metadata](#5-model-registry-metadata)
6. [Model Coverage](#6-model-coverage)
7. [Post-Processing DAG Flows](#7-post-processing-dag-flows)
8. [How This Replaces mmGP](#8-how-this-replaces-mmgp)
9. [Diffusers 0.38.0 Improvements](#9-diffusers-0380)

---

## 1. Product Tiers

Two user-facing choices. Simple. The dropdown is optional — defaults are pre-filled.

### Quality
- Uses the full model variant (not distilled/Turbo)
- More steps (model default, e.g. 30)
- All post-processing available
- System picks the best format VRAM allows — BF16 if possible
- User experience: "best output, willing to wait"

### Speed
- Uses the distilled/Turbo variant if available
- Fewer steps (e.g. 4-8)
- Minimal post-processing
- System uses FP8 for hardware-accelerated compute
- User experience: "good enough, fast"

### Advanced (optional dropdown)
Users who want control can access:
- Steps (slider, with model default)
- CFG scale (slider, with model default)
- Seed
- Sampler/scheduler (if model supports alternatives)

These are **per-request overrides**, not tier definitions. The tier sets the defaults; advanced lets you tweak.

---

## 2. Component-Type Rules

Every generative model — regardless of architecture — has the same component
types. The optimization strategy applies to TYPES, not individual models.

### Rule 1: VAE is always BF16
- VAEs are tiny (250MB - 500MB)
- Quantizing them saves nothing meaningful
- VAE precision errors cause visible decode artifacts
- Don't group_offload VAEs in diffusers 0.37.0 (bug, fixed in 0.37.1)
- Just keep it resident on GPU in BF16

### Rule 2: Text encoder / VLM is quantizable
- Text encoders produce embeddings — minor precision loss is invisible
- This is where you cut VRAM first
- Quantize to FP8 or int8 aggressively
- The LLM/VLM doesn't need pixel-level precision, it needs semantic understanding
- `enable_layerwise_casting` on text encoder only

### Rule 3: Diffusion transformer gets the best format VRAM allows
- This is the precision-critical component
- Errors accumulate across denoising steps (30 steps × small error = visible)
- BF16 if it fits → FP8 if it doesn't → int8 as last resort
- This is the LAST component to quantize, not the first

### Rule 4: group_offload when something doesn't fit
- When even FP8 doesn't fit, stream blocks via `enable_group_offload`
- `use_stream=True` for async prefetch (mandatory for performance)
- `record_stream=True` to prevent memory reclamation bugs
- No torch.compile, no cache_accel on this path (incompatible)

### Degradation chain (applied in order)

```
BF16 resident → BF16 + group_offload → FP8 resident → FP8 + group_offload → GGUF → queue
                                                                                 ↑      ↑
                                                                          last resort  (wait for VRAM)
```

The system degrades gracefully. It sacrifices:
1. First: residency (start streaming) — **same quality, slower**
2. Second: text encoder precision — **invisible quality impact**
3. Third: transformer precision — **visible quality impact**
4. Last resort: GGUF with CPU offload — **quality loss, but runs**
5. Never: VAE precision (keep BF16 always)

### The BF16 streaming insight (KEY ADVANTAGE)

group_offload enables BF16 transformer quality at low VRAM by streaming
blocks instead of quantizing. A 20GB BF16 transformer only needs ~1GB of
VRAM (2 blocks at a time) when streaming. This means:

- **8GB VRAM can run BF16 quality** for models up to ~20B params
- No need for GGUF Q5 unless VRAM < 4GB
- The Quality tier NEVER needs to quantize the transformer (just stream it)
- This is fundamentally better than ComfyUI workflows that jump to Q5 GGUF

```
Quality tier VRAM budget (Qwen-Image-Edit 20B):
  VLM FP8 resident:        ~3.5 GB
  Transformer BF16 stream: ~1.0 GB  (2 blocks, ~800MB)
  VAE BF16 resident:       ~0.25 GB
  Activations:             ~2.0 GB
  CUDA overhead:           ~1.0 GB
  Total:                   ~7.75 GB  ← BF16 quality on 8GB
```

Tradeoff: BF16 streaming is slower than BF16 resident (PCIe transfer per
block per step). But quality stays at BF16 level. Speed is sacrificed,
NOT quality. For the Quality tier, this is the right tradeoff.

---

## 3. Adaptive Runtime

When a user clicks "Generate," the system:

```
1. Detect available VRAM
   vram_free = nvidia-smi or torch.cuda.mem_get_info()
   (account for other loaded models on the same GPU)

2. Look up model component metadata from registry
   components = MODEL_REGISTRY[model_name].components

3. For each component, pick format + optimization:
   for component in components:
       if component.type == "vae":
           → BF16, resident (Rule 1)
       elif component.type == "text_encoder":
           → quantizable (Rule 2), see format matrix
       elif component.type == "transformer":
           → best available (Rule 3), see format matrix

4. Allocate VRAM budget per component:
   vae_size (fixed, ~0.5GB)
   + text_encoder_size (depends on format chosen)
   + transformer_size (depends on format chosen)
   + activation_headroom (~2-4GB depending on resolution)
   ≤ vram_free

5. If budget exceeds VRAM:
   - Degrade text_encoder format first
   - Then add group_offload to transformer
   - Then degrade transformer format
   - Never touch VAE

6. Load model with chosen configuration
7. Execute generation
8. Return result + log which path was used
```

### VRAM detection in practice

```python
def get_available_vram():
    """Get available VRAM, accounting for PyTorch's cache."""
    free, total = torch.cuda.mem_get_info()
    # Don't use ALL free VRAM — leave headroom for activations
    usable = free * 0.85  # 15% headroom
    return usable
```

For multi-model scenarios (multiple Ray Serve replicas on same GPU):
- Track loaded models in a shared registry
- Subtract their resident VRAM from available
- Or: use Ray's GPU resource scheduling (1 GPU = 1 replica)

---

## 4. Format Selection

### Format comparison (RTX 4090)

| Format | Quality | Speed | VRAM | LoRA | Hardware |
|--------|---------|-------|------|------|----------|
| BF16 | ★★★★★ | Fast | 1x | ✅ Native | Any GPU |
| FP8 scaled | ★★★★☆ | **Very fast** | 0.5x | ✅ Native | RTX 40/50, Hopper |
| FP8 flat | ★★★☆☆ | Very fast | 0.5x | ✅ Native | RTX 40/50, Hopper |
| int8 quanto | ★★★★☆ | Fast | 0.5x | ✅ Native | Any GPU |
| GGUF Q8_0 | ★★★★☆ | Moderate | 0.5x | ⚠️ Overhead | Any (CPU offload OK) |
| GGUF Q5_K_M | ★★★☆☆ | Slow if offloaded | 0.3x | ⚠️ Overhead | Any (CPU offload OK) |

### When to use each

**BF16**: Quality tier, when model fits in VRAM. The gold standard.

**FP8**: Speed tier, or Quality tier when VRAM is tight. Hardware-accelerated
on RTX 4090 via dedicated FP8 Tensor Cores. Use FP8 Scaled (with per-tensor
scaling factors) over FP8 Flat when possible — near-BF16 quality.

**int8 quanto**: What production currently uses. Good quality, moderate speed.
Already have all model files in this format on disk (`_quanto_bf16_int8.safetensors`).

**GGUF**: ONLY for the accessibility path — when model must split across
CPU/GPU. Not for speed (software dequant is slower than FP8 hardware dequant).
Q8_0 for quality, Q5_K_M for smaller footprint. LoRAs work but with overhead.

### Per-component format selection matrix

```
Available VRAM:     LOW (<8GB)    MEDIUM (8-16GB)   HIGH (16-24GB)   FULL (>24GB)
─────────────────────────────────────────────────────────────────────────────────
text_encoder:       int8          FP8               FP8              BF16
transformer:        FP8+offload   FP8               BF16+offload     BF16
VAE:                BF16          BF16              BF16             BF16
```

The system walks this matrix at runtime. User never sees it.

---

## 5. Model Registry Metadata

Each model declares its components and their properties:

```yaml
z-image:
  tiers:
    quality:
      model_variant: ZImageBase
      default_steps: 30
      default_cfg: 4.0
    speed:
      model_variant: ZImageTurbo
      default_steps: 8
      default_cfg: 3.5
  
  components:
    text_encoder:
      type: text_encoder
      size_bf16_gb: 9.5          # T5-XXL
      quantizable: true
      class: T5EncoderModel
    
    transformer:
      type: transformer
      size_bf16_gb: 12.0
      quantizable: true           # can quantize if needed
      precision_critical: true    # but prefer not to
      class: FluxTransformer2DModel
    
    vae:
      type: vae
      size_bf16_gb: 0.3
      quantizable: false
      group_offload_compatible: false  # 0.37.0 bug
      class: AutoencoderKL
```

```yaml
anima:
  tiers:
    quality:
      model_variant: anima-base
      default_steps: 30
      default_cfg: 4.0
    speed:
      model_variant: anima-base  # no Turbo variant
      default_steps: 15
      default_cfg: 4.0
  
  components:
    text_encoder:
      type: text_encoder
      class: Qwen3ForCausalLM      # Qwen3-0.6B-Base
      size_bf16_gb: 1.2
      quantizable: true
    
    llm_adapter:
      type: text_encoder            # treated as text pipeline (quantizable)
      class: Custom
      size_bf16_gb: 0.3
      quantizable: true             # small, doesn't matter
    
    transformer:
      type: transformer
      class: CosmosTransformer3DModel
      size_bf16_gb: 3.9
      quantizable: true
      precision_critical: true
    
    vae:
      type: vae
      class: AutoencoderKLQwenImage
      size_bf16_gb: 0.25
      quantizable: false
      group_offload_compatible: false  # 0.37.0 bug with post_quant_conv
```

```yaml
ace-step:
  tiers:
    quality:
      model_variant: ACE-Step-1.5
      default_steps: 50             # DiT flow matching
    speed:
      model_variant: ACE-Step-Turbo # if available
      default_steps: 20
  
  components:
    text_encoder:
      type: text_encoder
      class: Qwen3ForCausalLM       # Qwen3-based
      quantizable: true
    
    transformer:
      type: transformer
      class: AceStepTransformer1DModel  # diffusers 0.38.0
      quantizable: true
      precision_critical: true
    
    vae:
      type: vae
      class: AutoencoderOobleck     # audio VAE
      quantizable: false
```

---

## 6. Model Coverage

### Models that fit the framework cleanly

| Model | Text Encoder | Transformer | VAE | Notes |
|-------|-------------|-------------|-----|-------|
| Z-Image | T5-XXL | ZImageTransformer | AutoencoderKL | Turbo variant for speed tier |
| FLUX.1 | T5-XXL + CLIP | FluxTransformer2D | AutoencoderKL | schnell=4 steps (speed), dev=20 steps (quality) |
| FLUX.2 | T5-XXL + CLIP | Flux2Transformer | Flux2VAE | Small decoder available (0.38.0) for faster decode |
| Anima | Qwen3-0.6B + LLM adapter | CosmosTransformer3D | AutoencoderKLQwenImage | Custom pipeline, component metadata required |
| ACE-Step | Qwen3 | AceStepTransformer1D | AutoencoderOobleck | Audio generation, 0.38.0 native support |
| Wan 2.1/2.2 | UMT5 | WanTransformer3D | AutoencoderKLWan | Video generation |
| LTX-2 | Gemma3 | LTXVideoTransformer3D | LTXVAE | Two-stage pipeline (base + upscaler) |
| Qwen-Image | Qwen2.5-VL | QwenImageTransformer | AutoencoderKLQwenImage | VLM component quantizable |
| HunyuanVideo | LLaMA | HunyuanTransformer | AutoencoderKLHunyuanVideo | 0.38.0 modular pipeline |

### Models that need custom runners (Tier 4)

| Model | Why custom | Component mapping |
|-------|-----------|-------------------|
| TRELLIS (3D) | Not in diffusers | SLAT transformer + decoder |
| Kokoro (TTS) | Not in diffusers | Autoregressive, not diffusion |
| MOSS (TTS) | Not in diffusers | Autoregressive, not diffusion |
| Index TTS | Not in diffusers | Autoregressive, not diffusion |

These still benefit from the component-type rules — their "transformer" component
can use the same format/offload logic. The difference is the pipeline orchestration.

---

## 7. Post-Processing DAG Flows

Post-processing features are **separate DAG flows** that can be toggled,
chained, or run independently. They are NOT part of the tier definition.

### Prompt Enhancement
- LLM rewrites/expands the user's prompt
- Runs BEFORE generation
- Toggle: on/off (default: on for Quality tier, off for Speed tier)
- DAG node: `PromptEnhance → Generate → ...`

### Best-of-N (Batch)
- Generate N images, return the best (or all)
- Runs DURING generation (batch_size=N)
- Toggle: N=1/2/4
- DAG node: parallel generation → selection

### Latent Upscale
- Upscales the generated image using a separate model
- Runs AFTER generation
- **Separate DAG entirely** — can be called independently on any image
- Toggle: on/off, or invoked as a standalone tool
- DAG: `Generate → [done]` then optionally `Upscale(image) → [enhanced]`

### DAG composition example

```
User request: Z-Image, Quality tier, with prompt enhancement + latent upscale

DAG:
  [PromptEnhance] → [Z-Image Generate] → [Latent Upscale] → [Output]
         │                    │                   │
    LLM rewrite          BF16 resident       Separate model
    (fast, CPU)          (~15s on 4090)      (~5s, own VRAM budget)
```

Each node is independently schedulable, scalable, and can be cached.
Prompt enhancement might run on CPU. Generation runs on GPU. Upscale might
run on a different GPU or after generation completes (freeing VRAM).

---

## 8. How This Replaces mmGP

mmGP had a static profile system:
```
mmGP profiles:
  1. HighRAM_HighVRAM_Fastest    → all in VRAM
  2. HighRAM_LowVRAM_Fast        → parts in VRAM
  3. LowRAM_HighVRAM_Medium      → VRAM + quantization
  4. LowRAM_LowVRAM_Slow         → parts in VRAM + quantization
  5. VerylowRAM_LowVRAM_Slowest  → parts + quant + no RAM copy
```

mmGP picked ONE profile for the ENTIRE model. All components got the same
treatment. The text encoder and transformer were quantized identically.

### The new system is component-aware

```
Old (mmGP):                    New (adaptive):
  whole model                   per-component
  → one profile                 → text_encoder: FP8
  → one format                  → transformer: BF16
  → one offload strategy        → VAE: BF16 resident
                                → offload: only if needed
```

### What we keep from mmGP
- The IDEA of adaptive VRAM management (detect VRAM, adjust strategy)
- The user shouldn't know offloading is happening

### What we gain
- Component-level granularity (quantize encoder, keep transformer BF16)
- User-facing product tiers (Quality / Speed)
- Post-processing as composable DAG flows
- Standard diffusers APIs (maintained by HuggingFace, not one developer)
- Compatibility with torch.compile, cache acceleration (on appropriate paths)
- No license restrictions

### What we lose
- mmGP's hand-tuned per-model optimizations (some models were very fast)
- Single-developer knowledge of edge cases (tied weights, custom modules)
- Zero configuration (mmGP just worked; we need metadata per model)

---

## 9. Diffusers 0.38.0 Improvements

We're on 0.37.0. Upgrading to 0.38.0 brings:

### Group offloading + TorchAo
- Improved compatibility between `enable_group_offload` and `TorchAoConfig`
- Resolves tensor subclass swapping issues (the `swap_tensors` problem)
- This may improve the group_offload + quantization path significantly

### New model support (zero-cost additions)
- **ACE-Step 1.5**: `AceStepTransformer1DModel` — audio generation, fits our framework
- **LLaDA2**: discrete diffusion for text generation
- **Nucleus-MoE Image**: 2B active / 17B total MoE — efficient sparse architecture
- **ERNIE-Image**: 8B parameter image generation
- **LongCat-AudioDiT**: text-to-audio diffusion
- **Flux.2 Small Decoder**: faster VAE decode for FLUX.2

### Modular pipeline support
- LTX-2 and Hunyuan 1.5 get modular pipeline APIs
- Better component isolation → cleaner per-component optimization

### Flash Attention 4 backend
- New attention backend for supported GPUs
- Potentially faster attention computation

### Recommendation
Upgrade to 0.38.0 after Phase 1 benchmarks are complete. The group_offload +
TorchAo improvement alone may change our optimization strategy.

---

## Implementation Priority

```
Phase 1 (DONE):     Benchmark FLUX-schnell → verify native APIs work
Phase 2 (NEXT):     Build adaptive loader with component-type rules
Phase 3:            Add model metadata to registry (start with Z-Image, Anima)
Phase 4:            Benchmark each tier × VRAM scenario (fair comparison)
Phase 5:            Quality comparison (save images, visual evaluation)
Phase 6:            Upgrade to diffusers 0.38.0, re-benchmark
Phase 7:            Integrate with Ray Serve DAG (tiers, toggles, post-processing)
Phase 8:            Migrate models one by one from mmGP
Phase 9:            Delete Wan2GP
```

Each phase produces a committed artifact. No phase depends on unverified
assumptions from a previous phase.
