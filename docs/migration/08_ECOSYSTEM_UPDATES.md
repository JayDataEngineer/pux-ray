# Ecosystem Updates — Model Support, Speed Recipe, Crash Root Cause

> **Date:** 2026-06-14
> **Source:** AI research (model support matrix, speed optimization, crash analysis)
> **Status:** VERIFIED on pod (diffusers 0.37.0, PyTorch 2.10.0, RTX 4090)

---

## 1. Model Support in Diffusers (CHANGES MIGRATION PLAN)

### Natively Supported (no custom handlers needed)

| Model | Pipeline Class | Status | Migration Impact |
|-------|---------------|--------|-----------------|
| **Z-Image** | `ZImagePipeline`, `ZImageImg2ImgPipeline`, `ZImageInpaintPipeline` | ✅ Native | One-liner: `ZImagePipeline.from_pretrained(...)` |
| **Anima** | `ModularPipeline` | ✅ Native (NEW!) | **Replaces 656-line anima_main.py!** See below |
| **FLUX.2 klein** | `Flux2KleinPipeline` | ✅ Native | We have 4B on disk already |
| **FLUX.1** | `FluxPipeline` | ✅ Native | Already tested |
| **Wan 2.1/2.2** | `WanPipeline`, `WanImageToVideoPipeline` | ✅ Native | |
| **LTX-2** | `LTX2Pipeline` | ✅ Native (modular in 0.38.0) | |
| **ACE-Step** | `AceStepPipeline` | ✅ Native (0.38.0) | Audio generation |

### NOT in Diffusers (need separate containers)

| Model | Library | Architecture | Migration Path |
|-------|---------|-------------|----------------|
| **Qwen3-TTS** | `transformers` / `qwen-tts` package | Autoregressive (not diffusion) | Dedicated worker container |
| **Kokoro/MOSS/Index TTS** | Custom repos | Autoregressive | Dedicated worker containers |
| **TRELLIS** | Microsoft repo | Custom SLAT | Dedicated worker container |

### The ModularPipeline Discovery (GAME CHANGER for Anima)

```python
# OLD: 656 lines of custom code (anima_main.py)
from models.anima.anima_main import AnimaModel
handler = AnimaModel(...)
handler.generate(...)

# NEW: ModularPipeline (if verified available)
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers")
image = pipe(prompt="...", num_inference_steps=30).images[0]
```

**NEEDS VERIFICATION:** Is ModularPipeline available in diffusers 0.37.0?
If yes: Anima migration goes from weeks to hours.
If no (only in 0.38.0): upgrade diffusers first, then Anima is trivial.

---

## 2. The "Max Speed" Recipe (for models that FIT in VRAM)

This is the optimization stack for Path A (fully resident). It's DIFFERENT
from what we've been testing (group_offload). We need to test this.

```python
import torch
from torchao.quantization import autoquant

# Step 1: Inductor config flags (free speedup)
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False

# Step 2: Load pipeline
pipe = Flux2KleinPipeline.from_pretrained("...", torch_dtype=torch.bfloat16)

# Step 3: Fully resident (NO offloading)
pipe.to("cuda")

# Step 4: Channels-last memory format (5-10% free speedup)
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# Step 5: Fuse QKV attention projections
pipe.fuse_qkv_projections()

# Step 6: Regional compile (transformer + VAE decode)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# Step 7: torchao autoquant (compiler-native quantization)
pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)

# Step 8: Warmup (triggers compilation — first run is slow)
_ = pipe(prompt="warmup", num_inference_steps=20)

# Step 9: Production speed
image = pipe(prompt="real prompt", num_inference_steps=20).images[0]
```

### Techniques we HAVEN'T tested yet from this recipe:

| Technique | Expected Speedup | Status |
|-----------|-----------------|--------|
| `channels_last` memory format | 5-10% free | ❌ Not tested |
| Inductor config flags | Unknown | ❌ Not tested |
| `fuse_qkv_projections()` | Moderate (fewer matmuls) | ❌ Not tested |
| `torchao.autoquant` | Quantization + compile synergy | ❌ Not tested (torchao not installed) |
| Regional compile on VAE decode | VAE decode is slow | ❌ Not tested separately |

**These only work on the FULLY RESIDENT path** (no group_offload).
They're mutually exclusive with offloading — but potentially much faster.

### When to use this recipe vs group_offload

```
Model fits in VRAM (<18GB quantized)?
  YES → Use max speed recipe (resident + compile + channels_last + fused QKV)
  NO  → Use group_offload (streaming, no compile, accepts overhead)
```

For FLUX.2-klein-4B (3.8GB): ALWAYS use max speed recipe. Fits easily.
For FLUX.1-dev (23GB BF16 / ~12GB FP8): Use group_offload (doesn't fit resident).
For Anima (3.9GB): ALWAYS use max speed recipe. Fits easily.
For Z-Image (6.5GB): Use max speed recipe if VRAM available, else group_offload.

---

## 3. Crash Root Cause — CPU RAM, NOT GPU VRAM (CRITICAL)

### The problem

Every exit code 137 (SIGKILL) we experienced was **CPU RAM exhaustion**,
not GPU VRAM OOM. The root cause:

```
FLUX-dev pipeline loading sequence:
  1. Read 33GB safetensors from disk
  2. Instantiate tensors in host memory: ~33GB
  3. Copy to nn.Parameter tensors: +33GB (DOUBLE ALLOCATION)
  4. Total CPU RAM needed: ~65GB

Our pod has: 59GB total RAM
  - OS + Ray + forge service: ~15GB used
  - Available for loading: ~44GB
  - Needed for FLUX-dev: ~65GB
  - Result: OOM KILL → exit 137
```

This explains why:
- FLUX-schnell (same 33GB size) also crashed sometimes during loading
- group_offload paths were more stable (they don't load the whole transformer to RAM at once)
- The forge service loading models made it worse (contention for the same RAM)

### The fixes

**Fix 1: Disable memory-mapping (prevents Kubernetes page-cache kills)**
```python
# Add to TOP of benchmark script, before any imports
import os
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"
```

**Fix 2: Scale forge to 0 during benchmarks**
```bash
serve scale forge=0  # frees VRAM + CPU RAM
# run benchmarks
serve scale forge=1  # restore after
```

**Fix 3: Use low_cpu_mem_usage=True (avoids double allocation)**
```python
pipe = FluxPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # streams weights directly, no double copy
)
```

**Fix 4: Quantize T5-XXL text encoder (saves 5GB RAM during loading)**
```python
# T5-XXL is 9.5GB — the largest single component
# Loading it as FP8 or GGUF saves ~5GB during the critical loading phase
pipe.text_encoder_2.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```

### Updated benchmark environment setup

```python
import os
# Must be set before ANY torch/diffusers imports
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from diffusers import FluxPipeline

# Always use low_cpu_mem_usage for large models
pipe = FluxPipeline.from_pretrained(
    "/models/flux-dev",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

---

## 4. API Availability (VERIFIED on pod, diffusers 0.37.0)

| API | Available? | Notes |
|-----|-----------|-------|
| `ModularPipeline` | ✅ YES (experimental) | "subject to breaking changes" warning |
| `Flux2KleinPipeline` | ✅ YES | FLUX.2 klein native |
| `ZImagePipeline` | ✅ YES | Z-Image native |
| `GGUFQuantizationConfig` | ✅ YES | Can load GGUF models! |
| `channels_last` memory format | ✅ YES | PyTorch native |
| Inductor config flags | ✅ YES | `torch._inductor.config` |
| `fuse_qkv_projections()` | ❌ NO | Not on FluxPipeline in 0.37.0 |
| `AceStepPipeline` | ❌ NO | Needs diffusers 0.38.0 |
| `torchao.autoquant` | ❌ NO | Needs `pip install torchao` |

### Key takeaways from verification

1. **ModularPipeline is AVAILABLE** — Anima can be loaded as one call (experimental)
2. **GGUFQuantizationConfig is AVAILABLE** — we CAN test the GGUF resident path
3. **channels_last + Inductor flags are AVAILABLE** — we CAN test the max speed recipe
4. **fuse_qkv_projections is NOT available** — skip that optimization step
5. **torchao NOT installed** — need `pip install torchao` for autoquant
6. **AceStep needs 0.38.0** — upgrade required for ACE-Step pipeline

---

## 5. GGUF Resident Path (UNTESTED — now possible with GGUFQuantizationConfig)

The AI suggests an alternative to group_offload for FLUX-dev:
load the transformer as Q4 GGUF (~6GB), fit entirely resident, then compile.

```python
from diffusers import FluxPipeline, AutoModel, GGUFQuantizationConfig

transformer = AutoModel.from_single_file(
    "flux1-dev-Q4_K_M.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe = FluxPipeline.from_pretrained(
    "/models/flux-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")  # everything resident
```

**This would test:** GGUF Q4 resident + compile vs BF16 group_offload streaming
**Quality concern:** Q4 is lower quality than BF16 or FP8
**Speed expectation:** Could be very fast (fully resident + compiled + small weights)
**NEEDS:** GGUF model file (would need to download or convert)

---

## Updated Migration Impact

### Models that are now TRIVIAL to migrate

| Model | Old approach | New approach | Effort |
|-------|-------------|-------------|--------|
| Z-Image | Wan2GP handler | `ZImagePipeline.from_pretrained()` | Hours |
| Anima | 656-line custom factory | `ModularPipeline.from_pretrained()` (if available) | Hours |
| FLUX.2 klein | Wan2GP handler | `Flux2KleinPipeline.from_pretrained()` | Hours |
| FLUX.1 | Wan2GP handler | `FluxPipeline.from_pretrained()` | Done (tested) |
| Wan | Wan2GP handler | `WanPipeline.from_pretrained()` | Hours |
| LTX-2 | Wan2GP handler | `LTX2Pipeline.from_pretrained()` | Hours |
| ACE-Step | Separate repo | `AceStepPipeline.from_pretrained()` (0.38.0) | Hours |

### Models that still need custom runners

| Model | Why | Library |
|-------|-----|---------|
| Qwen3-TTS | Autoregressive, not diffusion | `transformers` / `qwen-tts` |
| Kokoro/MOSS/Index TTS | Autoregressive | Custom repos |
| TRELLIS | Custom SLAT architecture | Microsoft repo |

### What this means

The migration is MUCH simpler than we thought. Most models are one-liners
in native diffusers. The Wan2GP handler layer (261,721 lines) is almost
entirely replaceable with `from_pretrained()` calls.

The real work is:
1. Building the adaptive loader (VRAM detection + format selection)
2. Integrating with Ray Serve DAG
3. Testing quality across configurations
4. Handling the 4-5 models that need custom runners
