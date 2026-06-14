# CLEAN REBUILD — Complete Architecture Plan

> **Date:** 2026-06-14
> **Status:** ACTIVE — this is the CURRENT plan, supersedes all previous docs
> **Principle:** One Docker. SGLang kernels. Transformers 4. No mmGP. Ever.

---

## THE DECISION

We are starting over. The previous approach (incrementally patching Wan2GP/mmGP,
running side-by-side with native diffusers) is dead. We are building a clean,
unified system from scratch.

**What died:**
- mmGP (6,361 lines) — replaced by `loader.py` (276 lines)
- Wan2GP handlers (261,721 lines) — replaced by `registry.py` (153 lines)
- wgp.py Gradio monolith (12,330 lines) — deleted, not needed
- Separate SGLang container — merged into single Docker
- All migration/incremental approaches

**What survived:**
- Ray Serve orchestration (gateway, forge, workflows)
- The forge auto-evict VRAM management system
- DAG workflow engine
- Model registry download infrastructure
- All benchmark data and findings (docs/migration/04, 07, 10)

---

## DOCKER ARCHITECTURE

### Single Image: `tech-noir/native`

**Base:** nvidia/cuda 12.8 + SGLang's compiled kernel stack

```
Layer 1: CUDA 12.8 runtime (nvidia/cuda base)
Layer 2: PyTorch 2.10+ (CUDA 12.8 wheels)
Layer 3: SGLang kernels (sgl-kernel, flashinfer, cache-dit)
Layer 4: SageAttention + FlashAttention
Layer 5: transformers 4.57.x (PINNED — NOT 5.x)
         diffusers 0.37+
         PEFT, accelerate, torchao
Layer 6: Our service code (services/native/)
Layer 7: Model configs (config/model_registry.yaml)
```

**Key constraint:** `transformers>=4.54.0,<5.0.0` — transformers v5 has
breaking changes (tokenizer API, decode returns list, special tokens renamed).
We stay on v4 for stability.

**No mmGP anywhere.** The Dockerfile does NOT install mmgp. The service code
does NOT import mmgp. All VRAM management uses native diffusers APIs:
- `enable_group_offload(use_stream=True)` — block-level streaming (replaces mmGP core)
- `enable_layerwise_casting(fp8)` — on-the-fly quantization (replaces mmGP quant)
- `compile_repeated_blocks()` — kernel fusion (mmGP couldn't do this)
- `apply_first_block_cache()` — step skipping (mmGP couldn't do this)
- PEFT LoRAs — compile-compatible (mmGP's weren't)

**File:** `infra/docker/Dockerfile.native`

---

## SERVICE ARCHITECTURE

### File Structure

```
services/native/
├── __init__.py          # Package docstring
├── registry.py          # 10 models: pipeline class + repo + defaults
├── loader.py            # Adaptive VRAM management (replaces mmGP)
├── lora.py              # PEFT LoRA manager
├── service.py           # NativeService (load/unload/infer)
└── forge_adapter.py     # Forge integration adapter
```

Total: 835 lines (replacing 280,000+ lines of Wan2GP/mmGP)

### registry.py — Model Definitions

10 models registered, each with pipeline class, repo, defaults:

**Image Models (7):**
| Name | Pipeline | Repo | Steps | License |
|------|----------|------|-------|---------|
| z-image | ZImagePipeline | Tongyi-MAI/Z-Image | 30 | Apache-2.0 |
| z-image-turbo | ZImagePipeline | Tongyi-MAI/Z-Image-Turbo | 8 | Apache-2.0 |
| flux-schnell | FluxPipeline | black-forest-labs/FLUX.1-schnell | 4 | Apache-2.0 |
| flux-dev | FluxPipeline | black-forest-labs/FLUX.1-dev | 20 | Non-commercial |
| flux2-klein-4b | Flux2KleinPipeline | black-forest-labs/FLUX.2-klein-4B | 8 | Apache-2.0 |
| anima | ModularPipeline | circlestone-labs/Anima-Base-v1.0-Diffusers | 30 | CircleStone |
| qwen-image | QwenImagePipeline | Qwen/Qwen-Image | 30 | Apache-2.0 |

**Video Models (3):**
| Name | Pipeline | Repo | Steps | License |
|------|----------|------|-------|---------|
| ltx-video | LTXPipeline | Lightricks/LTX-Video | 50 | Apache-2.0 |
| wan-t2v | WanPipeline | Wan-AI/Wan2.1-T2V-14B-Diffusers | 30 | Apache-2.0 |
| wan-i2v | WanImageToVideoPipeline | Wan-AI/Wan2.1-I2V-14B-480P-Diffusers | 30 | Apache-2.0 |

Each model is a one-liner to load: `from_pretrained(repo)` → done.
No handler code. No translation layer. No reverse-engineering.

### loader.py — Adaptive VRAM Management (mmGP Replacement)

The core of the system. Replaces mmGP's 6,361 lines with 276 lines.

**How it works:**
1. After `from_pretrained()`, measure actual component sizes
2. Check available VRAM
3. Select optimal strategy from the degradation chain:

```
BF16 resident (pipe.to("cuda") + compile + cache)
  ↓ doesn't fit
BF16 group_offload (stream blocks via CUDA streams)
  ↓ doesn't fit
FP8 resident (halve weights via layerwise_casting)
  ↓ doesn't fit
FP8 group_offload (stream FP8 blocks)
  ↓ doesn't fit
model_cpu_offload (component swap fallback)
```

**Component-type rules applied automatically:**
- VAE: ALWAYS BF16 (too small to quantize, precision-critical for decode)
- Text encoder: quantizable (invisible quality impact)
- Transformer: gets best format VRAM allows (precision-critical)

**Key APIs used (all native diffusers/transformers):**
- `pipe.transformer.enable_group_offload(use_stream=True, record_stream=True)`
- `pipe.text_encoder.enable_layerwise_casting(storage_dtype=fp8, compute_dtype=bf16)`
- `pipe.transformer.compile_repeated_blocks(fullgraph=True)`
- `apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.05))`

### Verified Compatibility Matrix (from benchmark testing)

| Combination | Compatible? | Why |
|-------------|------------|-----|
| group_offload + torch.compile | ❌ | swap_tensors vs TensorWeakRef guards |
| group_offload + cache_accel | ❌ | block-skipping breaks prefetch chain |
| torch.compile + cache_accel | ❌ | @torch.compiler.disable graph break |
| group_offload + PEFT LoRA | ✅ | Load LoRAs BEFORE enabling offload |
| layerwise_casting + group_offload | ✅ | Cast first, then offload |
| compile + layerwise_casting | ✅ | Both on resident path |
| model_cpu_offload + compile | ✅ | compile_repeated_blocks works |
| model_cpu_offload + cache_accel | ✅ | Cache works on CPU offload path |

### lora.py — PEFT LoRA Manager

Replaces mmGP's 560-line `load_loras_into_model` with 62 lines.

**API:**
```python
lora_mgr.load("/path/to/lora.safetensors", name="style")    # Load adapter
lora_mgr.set_active(["style"], scales=[0.85])                # Activate with scale
lora_mgr.set_scale("style", 0.5)                             # Dynamic (<5ms)
lora_mgr.unload("style")                                     # Remove adapter
lora_mgr.list()                                              # List loaded
```

**Rules:**
1. Load LoRAs BEFORE group_offload (hooks need adapter params)
2. PEFT is compile-compatible (mmGP's monkey-patching wasn't)
3. Multiple LoRAs with independent scales
4. No circular module references (mmGP's hooks caused recursion bugs)

### service.py — NativeService

The ForgeService implementation. Three methods:
- `load(model_name)` → from_pretrained + VRAM plan + apply
- `unload()` → delete pipeline + release VRAM
- `infer(payload)` → extract params + generate + format output

Handles:
- Text-to-image, text-to-video, image-to-video, image editing
- LoRA loading from payload
- Seed control for reproducibility
- Negative prompts
- Image inputs (base64) for I2V/editing
- Video output (frames list)
- Image output (base64 PNG)
- Metrics (latency, VRAM, strategy)

---

## THREE-PHASE IMPLEMENTATION PLAN

### Phase 1: Image Models (DONE)

**Scope:** Z-Image, Z-Image-Turbo, FLUX.1-schnell, FLUX.1-dev, FLUX.2-klein-4b,
Anima, Qwen-Image

**Status:** ✅ Code written, 4 models tested on pod:
- Z-Image Turbo: 9.6s resident, 10.4s group_offload (7.8GB VRAM)
- LTX-Video: 0.9s for 25 frames (14GB VRAM)
- FLUX.1-schnell: 6.4s group_offload (9.5GB VRAM)

**What's implemented:**
- registry.py with 7 image models
- loader.py with adaptive VRAM (5 strategies)
- lora.py with PEFT LoRA manager
- service.py with full generation pipeline
- forge_adapter.py for Ray Serve integration

### Phase 2: Custom VRAM Package (DONE — code written, needs packaging)

**Scope:** Package loader.py as a standalone pip-installable module.

**Current state:** The loader is `services/native/loader.py`. It works as part
of the service but isn't a standalone package.

**To package:**
```bash
mkdir vrampkg/
cp services/native/loader.py vrampkg/__init__.py
# Add setup.py / pyproject.toml
# pip install -e . from the package directory
```

The loader uses ONLY diffusers/transformers/PyTorch APIs — zero dependencies
on our service code. It can be extracted and used by any diffusers pipeline.

### Phase 3: LTX Video Hyper-Optimization (TODO)

**Scope:** Build a HyperOptimizedLTXSequencer using the deep research blueprint.

**Source:** The detailed LTX optimization document (Piecewise CUDA Graphs,
static padding, guiding latents, IC-LoRA masking).

**Key techniques to implement:**

#### 3a. Static Latent Shape Padding
- Fix temporal dimension to discrete profiles: {9, 17, 33, 65, 97} frames
- Pad shorter sequences with zeros to nearest profile
- Apply 3D attention mask: valid frames = 1.0, padded = -inf
- Prevents recompilation when user changes frame count

#### 3b. Piecewise CUDA Graphs (PCG)
- Divide DiT into individual transformer block subgraphs
- Each block wrapped in CUDAPiecewiseBackend
- Dynamic operations (masking, slicing) stay in eager mode between blocks
- Capture graphs in reverse size order (largest → smallest) for memory reuse

#### 3c. Discrete Shape Profiling
- Pre-capture CUDA graphs for profiles: {9, 17, 33, 65, 97}
- Binary search maps logical frame count → nearest profile
- Global memory pool shared across all profiles
- Total VRAM stays within 24GB (RTX 4090/5090)

#### 3d. Guiding vs Replacing Latents
- **Replacing:** Direct overwrite of latent frame with VAE-encoded image
  - Use for: hard scene cuts, strict keyframe alignment
  - Breaks ODE solver trajectories (may cause flicker)
- **Guiding:** Additive spatial-temporal signal onto noise latents
  - Use for: smooth transitions, in-betweening
  - Gaussian decay around keyframe: exp(-0.5 * distance²)
  - Preserves solver continuity

#### 3e. IC-LoRA Attention Masking
- Parse `reference_downscale_factor` from loaded LoRAs
- Load spatial mask (B, 1, F, H, W), convert to grayscale, normalize [0,1]
- Downsample to latent space with causal temporal alignment
- Multiply by conditioning_attention_strength (γ)
- Inject into self-attention query-key projections

#### 3f. Dual-Stage Pipeline
- Stage 1: Coarse generation at half resolution (W/2, H/2)
- Stage 2: 2× spatial upscale with distilled LoRA, 4-8 steps
- Use second-order res_2s solver for fewer total steps
- FP8 quantization via `--quantization fp8-scaled-mm`

**Lightricks native pipelines available:**
- `TI2VidTwoStagesPipeline` — standard high-quality
- `TI2VidTwoStagesHQPipeline` — second-order solver
- `TI2VidOneStagePipeline` — fast prototyping
- `DistilledPipeline` — 8-step fixed path, no CFG
- `ICLoraPipeline` — video-to-video with IC-LoRA control

**File to create:** `services/native/ltx_sequencer.py`

---

## FORGE INTEGRATION

The forge's SERVICE_MAP routes model requests:

```python
SERVICE_MAP = {
    "native":    ("services.native.forge_adapter", "NativeForgeService"),
    "z-image":   ("services.native.forge_adapter", "NativeForgeService"),
    "anima":     ("services.native.forge_adapter", "NativeForgeService"),
    "wan2gp":    ("services.wan2gp.forge_adapter", "Wan2GPForgeService"),  # legacy fallback
    "comfyui":   ("services.image.comfyui",        "ComfyUIService"),
    "llm":       ("services.llm.deployment",        "LLMService"),
    ...
}
```

The native service handles VRAM via the forge's auto-evict system:
1. User requests model X via forge
2. Forge checks VRAM, evicts other services if needed
3. NativeService.load(X) is called
4. loader.py inspects VRAM and selects strategy
5. Generation runs
6. User requests different model → forge evicts → loads new one

**Auto-evict is critical:** The forge manages VRAM between ALL services
(native, comfyui, llm). When native needs VRAM, forge evicts comfyui/llm.
When comfyui needs VRAM, forge evicts native. This is already implemented
in `services/forge.py` — we just need to use it properly.

---

## ENVIRONMENT VARIABLES

Required in the Docker image:
```bash
SAFETENSORS_DISABLE_MMAP=1                    # Prevent Kubernetes OOM kills
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce fragmentation
HF_HOME=/models/hf_cache                       # Persistent HF cache
HF_HUB_CACHE=/models/hf_cache/hub
PYTHONPATH=/app
```

For SGLang kernels on RTX 4090 (SM89):
```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:\
/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH}
```

---

## MODEL STORAGE

All models on persistent volume at `/models/`:

```
/models/
├── native/                    # Native diffusers format models
│   ├── z-image-turbo/         # ✅ Cached (14GB)
│   ├── z-image/               # TODO: download
│   ├── anima/                 # TODO: download
│   ├── flux-schnell/          # ✅ Cached (32GB)
│   ├── flux-dev/              # TODO: download
│   └── ...
├── ltx-video/                 # ✅ Cached (5GB)
├── flux-schnell/              # Legacy path (also cached)
├── hf_cache/                  # HuggingFace download cache
└── bench_fair/                # Benchmark results + output images
```

Download via model_sync.py or directly:
```python
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="/models/native/z-image-turbo")
```

---

## VERIFIED BENCHMARK RESULTS

From E2E testing on RTX 4090 (docs/migration/10_NATIVE_E2E_RESULTS.md):

| Model | Strategy | Time | VRAM | Notes |
|-------|----------|------|------|-------|
| Z-Image Turbo | BF16 resident | 9.6s | 22.0GB | 8 steps, 1024×1024 |
| Z-Image Turbo | group_offload | 10.4s | 7.8GB | Same model, less VRAM |
| LTX-Video | BF16 resident | 0.9s | 14.0GB | 10 steps, 25 frames, 512×320 |
| FLUX.1-schnell | group_offload | 6.4s | 9.5GB | 23GB transformer in 9.5GB! |
| FLUX.1-schnell | fp8 group_offload | 3.4s | 12.5GB | Earlier benchmark |
| FLUX.1-dev (20 steps) | fp8 group_offload | 16.1s | 12.5GB | Earlier benchmark |
| FLUX.1-dev (20 steps) | bf16 group_offload | 20.6s | 12.5GB | Earlier benchmark |

Key finding: group_offload uses 50-65% less VRAM than resident at <2x speed cost.
The adaptive loader selects this automatically when VRAM is limited.

---

## WHAT'S LEFT

1. **Build the Docker image** — `podman build -f infra/docker/Dockerfile.native`
2. **Deploy to cluster** — Update KubeRay config to use new image
3. **Download remaining models** — Z-Image Base, Anima, Wan, FLUX.2-klein
4. **Test LoRA loading** — With actual LoRA files on a model
5. **Build LTX sequencer** — Phase 3, the hyper-optimized video pipeline
6. **Update MCP for DAG** — Already written (mcp/dag/server.py), needs testing
7. **Remove Wan2GP** — Once all models migrated, delete from Docker and code
