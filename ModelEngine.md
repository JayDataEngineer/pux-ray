# Model Engine

Universal PyTorch model execution with mmgp VRAM management.
Every GPU model decomposes into `{name: nn.Module}`. mmgp manages placement.
One handler per model family. Multiple models per GPU.

## Architecture

```
                    FORGE GATEWAY
           /forge {"service": "ace_step", ...}
                    │
          ┌─────────▼──────────┐
          │   ModelExecutor     │
          │   (owns the GPU)    │
          │                     │
          │  ┌───────────────┐  │
          │  │  MMGP POOL    │  │
          │  │               │  │
          │  │ ace_step:     │  │
          │  │   transformer │  │
          │  │   codec       │  │
          │  │   text_enc_2  │  │
          │  │               │  │
          │  │ wan2gp:       │  │
          │  │   transformer │  │
          │  │   vae         │  │
          │  │   text_enc    │  │
          │  │               │  │
          │  │ trellis:      │  │
          │  │   transformer │  │
          │  │   decoder     │  │
          │  └───────────────┘  │
          │                     │
          │  mmgp manages ALL   │
          │  modules — only     │
          │  active inference   │
          │  components in VRAM │
          └─────────────────────┘
                    │
                    ▼
                KubeRay
         head + worker pods
         autoscaling: idle → 0
```

## Multi-Model on One GPU

Unlike Forge's current one-at-a-time eviction, the Model Executor loads
multiple model families into a **shared mmgp pool**. mmgp offloads inactive
modules to RAM and loads active ones to VRAM on demand.

Example on RTX 4090 (24GB):

```
Loaded models:
  ace_step_v1_5_turbo   (~7GB quantized)
  wan/t2v-14B           (~14GB quantized)

Active VRAM budget: ~10GB
  - ace_step transformer (active): 2.5GB in VRAM
  - wan text_encoder (active):     1.2GB in VRAM
  - Everything else:               offloaded to RAM

Request comes in for ace_step:
  → mmgp loads ace_step modules to VRAM
  → mmgp offloads wan modules to RAM
  → No explicit unload needed

Request comes in for wan:
  → mmgp loads wan modules to VRAM
  → mmgp offloads ace_step modules to RAM
  → Seamlessly swapped by mmgp
```

The key: **no explicit unload**. mmgp handles module-level swapping.
Models stay registered in the pool — only their weights move between
VRAM and RAM based on what's actively running.

## The Handler Contract

Every handler package: `handlers/<family>/`

```
handlers/<family>/
  __init__.py      # BaseHandler implementation + variant metadata
  modules.py       # Load nn.Modules, build pipe dict, extract weights
  orchestrator.py  # Raw forward() calls — the inference logic
```

### BaseHandler

```python
class BaseHandler(ABC):
    def supported_types(self) -> list[str]
    def load_model(model_type, model_path, **kwargs) -> LoadResult
    def get_variant(model_type) -> ModelVariant

@dataclass
class LoadResult:
    pipeline: Any                       # orchestrator — callable with payload
    pipe: dict[str, nn.Module]          # mmgp manages these
    co_tenants: dict[str, list[str]]    # concurrent VRAM sharing rules
```

### modules.py

Produces a dataclass holding all raw nn.Modules + the mmgp pipe dict.
Every module loaded independently. No pipeline wrappers.

Example: `AceStepModules` holds transformer, text_encoder_2, codec, lm_model.
Pipe dict: `{"transformer": ..., "text_encoder_2": ..., "codec": ...}`.
LM managed separately (tied weights incompatible with mmgp hooks).

### orchestrator.py

Takes the modules dataclass, runs inference via direct `.forward()` calls.
No abstractions. Every tensor op is explicit.

Example: `AceStepOrchestrator` runs 8 phases — CoT, codes, text encoding,
latents, reference audio, conditioning, denoising, VAE decode.

### __init__.py

Thin wrapper implementing `BaseHandler`. Routes model_type → modules.load().
Defines `ModelVariant` entries with defaults per variant.

## ModelExecutor

One executor per GPU. Manages the shared mmgp pool.

```python
executor = ModelExecutor(models_root, mmgp_profile=1)
executor.register_handler("ace_step", AceStepHandler())
executor.register_handler("wan2gp", Wan2GPHandler())

# Load multiple models — all go into shared mmgp pool
executor.load("ace_step_v1_5_turbo")
executor.load("wan/t2v-14B")

# Inference — mmgp handles module swapping
result = executor.infer("ace_step_v1_5_turbo", {"prompt": "jazz piano"})
result = executor.infer("wan/t2v-14B", {"prompt": "sunset timelapse"})

# Both models stay loaded. mmgp swaps modules in/out of VRAM.
```

### Pool Management

When `load()` is called:
1. Handler decomposes model into pipe dict
2. Pipe dict merged into the shared mmgp pool
3. mmgp re-profiles with combined co_tenants map
4. VRAM budget divided across all loaded models

When VRAM is tight:
1. Executor checks combined vram_estimate_gb of all loaded models
2. If exceeding GPU capacity, evicts least-recently-used model
3. Removes its modules from the pool, re-profiles mmgp

## Wan2GP Integration

Wan2GP already uses mmgp. Its `family_handler` pattern maps directly:

```
Wan2GP                          Model Engine
──────────                      ────────────
family_handler                  BaseHandler subclass
family_handler.load_model()     handler.load_model() → LoadResult
family_handler.query_model_def()  absorbed into load_model()
V2V_MODELS registry             ModelVariant entries
model.generate()                orchestrator(payload)
```

### Supported Wan2GP Families

| Family | Handler Module | Models |
|--------|---------------|--------|
| WAN video | `models.wan.wan_handler` | t2v-14B, i2v-14B |
| Hunyuan | `models.hyvideo.hunyuan_handler` | t2v |
| Flux | `models.flux.flux_handler` | t2i |
| ACE-Step | `models.TTS.ace_step_handler` | v1_5 (replaced by native handler) |
| IndexTTS | `models.TTS.index_tts2_handler` | v2 (blocked — transformers compat) |

The Wan2GP handler wraps these dynamically — imports the vendor handler,
calls `load_model()`, adapts the output to `LoadResult`.

## Migration Priority

| Priority | Service | Complexity | Why |
|----------|---------|-----------|-----|
| 1 | Wan2GP video | Low (wrap) | Already mmgp. 90+ models. Biggest ROI. |
| 2 | TRELLIS | Medium | Accessible pipeline. Clear decomposition. Proves 3D pattern. |
| 3 | MOSS-SoundEffect | Medium-High | 8B model needs mmgp quantization most. |
| 4 | HY-Motion | High | Multi-model (Qwen3-8B + CLIP + diffusion). Biggest VRAM hog. |
| 5 | AniGen | High | fp32 + patching. Same pattern once patched. |
| 6 | See-Through | Highest | Opaque vendor. May need fork first. |

## Better Than Wan2GP

| Aspect | Wan2GP | Tech Noir Model Engine |
|--------|--------|----------------------|
| Multi-model | Per-handler mmgp, no eviction strategy | Shared pool with LRU eviction + VRAM budgets |
| Scale | Single GPU, single machine | KubeRay: 1 GPU → N GPUs, 1 node → N nodes |
| API | Python scripts | REST via Traefik, service-agnostic |
| Observability | Print statements | Grafana + VictoriaMetrics + Loki |
| Models | What Wan2GP ships | Any model with a handler |
| Scheduling | Load whatever | VRAM-aware: concurrent when possible, evict when needed |

## KubeRay Scaling

```
Single GPU today (RTX 4090 24GB):
  - Multiple models in shared mmgp pool
  - Module-level VRAM swapping
  - ~4s generation for 10s audio, ~30s for 5s video

Multi-GPU (add KubeRay worker pods):
  - Worker 1 (GPU 0): music + audio models
  - Worker 2 (GPU 1): video + image models
  - Worker 3 (GPU 2): 3D + motion models
  - Each worker = independent ModelExecutor
  - Ray Serve routes requests to available workers
  - Autoscaling: idle workers → 0, queue → scale up
```

## File Map

```
services/model_engine/
  __init__.py           # Package docstring
  base_handler.py       # BaseHandler, LoadResult, ModelVariant
  executor.py           # ModelExecutor — GPU owner, mmgp pool manager
  handlers/
    __init__.py
    ace_step/           # ACE-Step v1.5 text-to-music (proven)
      __init__.py       # AceStepHandler
      modules.py        # AceStepModules dataclass + loader
      orchestrator.py   # AceStepOrchestrator — 8-phase generation
      lm_engine.py      # CoT metadata + audio code generation
      audio_codes.py    # Audio code vocabulary + prompt building
      models/           # Model config + architecture classes
    wan2gp/             # Wan2GP wrapper (video, image, TTS)
      __init__.py       # Wan2GPHandler
```
