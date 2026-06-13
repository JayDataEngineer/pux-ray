# Native API Reference — mmGP Replacement Cheat Sheet

> Quick lookup for the diffusers/PEFT APIs that replace mmGP functionality.
> All APIs verified present in diffusers 0.37.0 on the production worker.

---

## VRAM Offloading

### Block-level async stream offloading (replaces mmGP core)

```python
# For models that don't fit entirely in VRAM
# Streams transformer blocks between CPU↔GPU with async CUDA prefetch

transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",       # "block_level" or "leaf_level"
    use_stream=True,                   # async CUDA stream prefetch
    num_blocks_per_group=None,         # tune: more blocks = fewer transfers, more VRAM
)
```

- `block_level`: groups consecutive transformer blocks together
- `leaf_level`: offloads at the finest granularity (individual leaf modules)
- `use_stream=True`: prefetches next group on background CUDA stream while computing current

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

transformer = CosmosTransformer3DModel.from_pretrained(
    "circlestone-labs/Anima",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
# Apply offload + casting
transformer.enable_group_offload(onload_device=torch.device("cuda"), use_stream=True)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn)
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

```bash
# Install
pip install "sglang[diffusion]"

# Serve a model
sglang serve --model-path Qwen/Qwen-Image --port 30010

# LTX with two-stage mode
sglang serve --model-path Lightricks/LTX-2.3 \
    --ltx2-two-stage-device-mode snapshot

# Generate (CLI)
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A sunset" --save-output
```

```python
# Call via OpenAI-compatible API
import openai
client = openai.Client(base_url="http://localhost:30010/v1", api_key="none")
response = client.images.generate(
    model="qwen-image",
    prompt="A sunset over mountains",
)
```

---

## Quick Decision Matrix

| Situation | API to use |
|-----------|-----------|
| Model fits in VRAM (≤~20GB) | `pipe.to("cuda")` — done |
| Model + pipeline don't fit | `pipe.enable_model_cpu_offload()` |
| Large model, need streaming | `transformer.enable_group_offload(use_stream=True)` |
| Need to cut VRAM 50% | `transformer.enable_layerwise_casting(fp8)` |
| Load LoRAs | `pipe.load_lora_weights()` + `pipe.set_adapters()` |
| VAE OOMs on decode | `pipe.vae.enable_tiling()` |
| Production serving, standard model | SGLang Diffusion `sglang serve` |
| Niche/custom model | Custom runner calling diffusers directly |

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
