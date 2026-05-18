# Workflow Orchestrator — System Design

## What This Is

A system for writing multi-model orchestration pipelines on top of Wan2GP's
model execution stack. Workflows call Wan2GPService.load()/.infer() in
sequence, piping intermediates between model calls. No Wan2GP internals
are modified — no family_handlers added, no wgp.py edits.

## Core Insight

"Models" and "Pipelines" (workflows) are distinct:

- **Models do inference**: QWEN-Edit, LTX Video, Z-Image, HY-Motion, TRELLIS,
  etc. Each has a Wan2GP handler that loads weights and provides generate().
  Wan2GP handles VRAM management (mmgp, quantization, CUDA graphs, LoRAs).

- **Workflows orchestrate models**: VNCCS, WDC, Tech Noir Studio build stages.
  Each is a Python function that sequences multiple Wan2GPService calls,
  routing outputs of one to inputs of the next.

This separation means:
- Zero Wan2GP code changes to add a new workflow
- Every model call gets Wan2GP's full optimization stack for free
- Workflows are plain Python — no DAG framework, no YAML DSL

## Architecture

```
CLIENT
  |
  v
Gateway (/v1/workflows/*)   ← Starlette routes in gateway/ingress.py
  |
  v
services/workflows/*.py      ← Workflow functions (Python)
  |
  v
Wan2GPService                ← services/wan2gp/deployment.py
  ├── load(model_name)       ← handler.load_model() + mmgp profiling
  └── infer(payload)         ← pipeline.generate(**kwargs)
        |
        v
Wan2GP model handler         ← opt/wan2gp/models/*/*_handler.py
  ├── load_model()           ← weight loading, pipe dict
  └── _Pipeline.generate()   ← the actual inference
```

## Directory Layout

```
services/workflows/
  __init__.py
  base.py                    — Wan2GPService singleton, encode/decode helpers
  utils/
    __init__.py
    body_mesh.py             — BodyMeshRenderer (Anny 3D → 2D, CPU)
  vnccs.py                   — 6 VNCCS workflows (char_sheet, emotions, sprite, pose_edit, clone, detailer)
  wdc.py                     — 4 WDC workflows (ltx_fflf_2stage, ltx_fflf_3stage, ltx_audio, timeline)
  tech_noir.py               — 12 Tech Noir build stages (generate, sheet, emotions, sprites, outfit, etc.)

gateway/routes/workflows.py  — Route registration: GET/POST /v1/workflows/{name}

spec/
  vnccs/                    — Pipeline spec + optimization gaps
  wdc/                      — Pipeline spec + optimization gaps
  tech_noir/                — Pipeline spec + optimization gaps
```

## Wan2GPService API

```python
from services.workflows.base import get_service

svc = get_service()

# Load a model (mmgp-managed, ~10-30s first load, no-op if already loaded)
svc.load("qwen-image-edit")

# Run inference
result = svc.infer({
    "input_prompt": "Draw character from image2",
    "image_b64": character_image_b64,
    "seed": 42,
    "sampling_steps": 4,
    "guide_scale": 1.0,
    "loras_selected": ["VNCCS/EmotionCoreV1.safetensors"],
})

# Response format
# {
#     "status": "ok",                     # or "error"
#     "data": "<base64-encoded bytes>",   # image/video/audio
#     "media_type": "image/png",
#     "model": "qwen-image-edit",
# }
```

## How a Workflow Works

```python
def char_sheet(prompt: str, seed: int = 42) -> dict:
    """Text → character base sheet. Two-stage: SD base + QWEN refine."""
    svc = get_service()

    svc.load("z_image")          # Load SD base model (or no-op if hot)
    base = svc.infer({...})      # Generate initial character image

    svc.load("qwen-image-edit")  # Load QWEN model
    refined = svc.infer({        # Refine character details
        "image_b64": base["data"],
        ...
    })
    return refined
```

Key property: models stay hot in VRAM between calls. Only `svc.load()` with
a different model name triggers unload + reload.

## The Three Workflow Families

### 1. VNCCS (6 workflows)
Character creation/sprite generation. Each ComfyUI workflow JSON is
replaced by a Python function calling Wan2GPService.

| Workflow | Backbone Model | ComfyUI Equivalent |
|----------|---------------|-------------------|
| char_sheet | Z-Image + QWEN-Edit | Step 1 |
| emotions | QWEN-Edit + EmotionCore LoRA | Step 3 |
| sprite | BodyMesh + QWEN-Edit | Step 4 |
| pose_edit | BodyMesh + QWEN-Edit | Pose Studio |
| clone | QWEN-Edit | Step 1.1 |
| detailer | QWEN-Edit | QWEN Detailer |

### 2. WDC (4 workflows)
LTX Video conditioning strategies. All are single-call with specific params.

| Workflow | Model | WDC Equivalent |
|----------|-------|---------------|
| ltx_fflf_2stage | LTX Video | I2V FFLF 2-stage |
| ltx_fflf_3stage | LTX Video | I2V FFLF 3-stage |
| ltx_audio | LTX Video | I2V + Audio |
| timeline | LTX Video | LTX Director |

### 3. Tech Noir Studio (12 workflows)
Build system stages. Each maps to a `stages_character/stages/` module.

| Workflow | Model | Build Stage |
|----------|-------|-------------|
| generate | Z-Image | generate.py |
| sheet | QWEN-Edit | sheet.py (clone) |
| face_detailer | QWEN-Edit | sheet.py (FaceDetailer inline) |
| emotions | QWEN-Edit | emotions.py |
| sprites_static | QWEN-Edit | sprites_static.py |
| sprites_animated | BodyMesh + QWEN-Edit | sprites_animated.py |
| motion_npz | HY-Motion | sprites_animated.py (hymotion) |
| outfit | QWEN-Edit | outfit.py |
| state | QWEN-Edit | state.py |
| trellis | TRELLIS | trellis.py |
| video | LTX Video | video.py |
| lora_dataset | Post-process | lora.py |

## Key Design Decisions

1. **No Wan2GP internals changed.** No family_handlers added, no wgp.py edited.
   Workflows live strictly in services/workflows/ and call Wan2GPService.

2. **No abstraction layer.** Workflows are plain Python functions. No DAG
   framework, no YAML DSL, no WorkflowSpec dataclass. This avoids the
   "abstract over Wan2GP" trap.

3. **BodyMeshRenderer extracted as utility.** The ComfyUI custom node
   for 3D mesh → 2D image rendering is ported to
   services/workflows/utils/body_mesh.py. Standalone, CPU-only, no Wan2GP dep.

4. **Composited images for multi-image conditioning.** VNCCS uses 3-image
   reference latent injection (mesh + character + skeleton). Our approximation:
   composite the images side-by-side and feed as single image_b64.

5. **Routes are pseudo-OpenAI spec.** GET /v1/workflows lists, GET/POST
   /v1/workflows/{name} for metadata/execution. Compatible with future
   MCP-like discovery.

## Adding a New Workflow

```python
# 1. Write the function in services/workflows/my_family.py
def my_workflow(param1: str, param2: int = 42) -> dict:
    """Docstring becomes API description."""
    svc = get_service()
    svc.load("some-model")
    return svc.infer({"input_prompt": param1, "seed": param2})

# 2. Register it in gateway/routes/workflows.py
_register(my_workflow, "my-family/my-workflow", "What it does")
```

No other files need changing. The function signature defines the API.

## Relationship to Other Systems

| System | Role | Integration |
|--------|------|-------------|
| Forge (services/forge.py) | VRAM-aware GPU manager | Not used — Wan2GPService self-manages mmgp |
| Wan2GPService (services/wan2gp/) | Model loading + inference | Direct import |
| ComfyUI (services/image/comfyui/) | Old VNCCS/WDC runtime | Being replaced; routes stay until consumers migrate |
| llama.cpp (services/llm/) | LLM subprocess | Not involved |
| Tech Noir Studio (reference/) | Asset build system | Target consumer — will call /v1/workflows/* instead of ComfyUI /prompt |
