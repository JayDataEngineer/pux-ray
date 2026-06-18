# Inference Pool System — 4-Tier Priority Fallback

> **Status**: Active. Replaces the monolithic `gpu-all` "all-in-one" docker with
> four independent serving tiers. Configured declaratively from
> [`config/inference_pools.yaml`](../config/inference_pools.yaml).

## TL;DR

```
                 ┌───────────────────────────────────────────────┐
                 │            web-ui / MCP client                 │
                 │   (chat, studio, playground, external app)     │
                 └───────────────────┬───────────────────────────┘
                                     │  MCP tools
                                     ▼
                 ┌───────────────────────────────────────────────┐
                 │  mcp/dag/server.py    mcp/inference/server.py  │
                 │  start_workflow       resolve_model           │
                 │  get_run_status       list_inference_pools    │
                 └───────────────────┬───────────────────────────┘
                                     │  HTTP
                                     ▼
                 ┌───────────────────────────────────────────────┐
                 │           gateway/ingress.py :30080            │
                 │   /v1/wf/* (DAG)    /v1/inference/* (new)      │
                 └───────────────────┬───────────────────────────┘
                                     │  dispatch
                                     ▼
                 ┌───────────────────────────────────────────────┐
                 │  services/workflows/engine.py                  │
                 │    step(service, model) →                      │
                 │      services.inference.dispatch.resolve_step  │
                 └───────────────────┬───────────────────────────┘
                                     │  priority fallback
                                     ▼
   ┌─────────────────┬───────────────────┬─────────────────┬─────────────────┐
   │  Tier A         │  Tier B           │  Tier C         │  Tier D         │
   │  Specialized    │  Omni vLLM        │  SGLang         │  Diffusers      │
   │                 │                   │                 │                 │
   │  moss :8050     │  omni-vllm :8093  │  sglang :8081   │  diffusers :8095│
   │  diarization    │  • qwen-image-    │  • ideogram4    │  • kimodo       │
   │  llama          │    edit (FP8+)    │  • z-image      │  • ace-step     │
   │  llama-bee      │  • wan-vace (FP8) │  • ltx-video    │  • kokoro       │
   │  comfyui        │  • z-image        │                 │  • vibevoice    │
   │                 │  • wan-t2v/i2v    │                 │                 │
   │                 │  • cosmos         │                 │                 │
   │                 │  • anima-base ⏳  │                 │                 │
   └─────────────────┴───────────────────┴─────────────────┴─────────────────┘
```

## The Four Tiers

Each tier is **one or more independent dockers**, mostly upstream images we
don't own. "With a few small exceptions" = our custom FP8 weight-only patches
that are bind-mounted over the in-image pipeline.

### Tier A — Specialized dockers (one service per docker)
Highest priority. Each model gets a dedicated docker.

| Pool | Framework | Port | Models | Notes |
|------|-----------|------|--------|-------|
| `moss` | MOSS | 8050 | `moss_tts`, `moss_soundeffect`, `moss_tts_realtime` | Standalone MOSS container |
| `diarization` | pyannote 3.1 | 8051 | `diarization` | Speaker diarization |
| `llama` | llama.cpp (upstream) | 8052 | `llama` | `llama-server-upstream` binary |
| `llama-bee` | BeeLlama.cpp | 8053 | `llama-bee` | DFlash + TurboQuant fork |
| `comfyui` | ComfyUI | 8054 | `comfyui` | Node-based image/video |

### Tier B — Omni vLLM (single multi-model docker)
All Diffusion Transformer (DiT) models under one `vllm/vllm-omni:latest`
container, with per-model launch scripts that bind-mount the right FP8 patch:

| Model | Optimization | Patch |
|-------|--------------|-------|
| `qwen-image-edit` | FP8 weight-only + Cache-DiT + TaylorSeer + VAE tiling/slicing | `pipeline_qwen_image_edit_plus_patch.py` |
| `wan-vace` | FP8 weight-only + TeaCache 0.01 (≈70% speedup) | `pipeline_wan2_2_vace_patch.py` |
| `z-image` | FP8 + Cache-DiT + TaylorSeer | (to be created) |
| `wan-t2v` / `wan-i2v` | FP8 | (to be created) |
| `cosmos` | BF16 + CPU offload (FP8 blocked on fused LLM params) | — |
| `anima-base` | (future — replaces cosmos slot) | — |

### Tier C — SGLang (single docker)
| Model | Optimization | Note |
|-------|--------------|------|
| `ideogram4` | NF4 (16 GB on 24 GB card) | Typography-aware T2I |
| `z-image` | FP8 + Cache-DiT (FN=2, BN=1) | Fallback for omni-vllm — 1.61s/img |
| `ltx-video` | ModelOpt FP8 (two-stage) | Standard FP8 crashes on CPU init |

### Tier D — Diffusers generalized (catch-all)
| Models |
|--------|
| `kimodo`, `ace-step`, `kokoro`, `index-tts`, `vibevoice-asr`, `faster-whisper`, `see-through` |

## How the Pieces Connect

### 1. Declarative config — `config/inference_pools.yaml`

Single source of truth. Adding a model = adding a route entry. No Python changes.

```yaml
pools:
  omni-vllm:
    tier: B
    priority: 100
    image: vllm/vllm-omni:latest
    port: 8093
    models: [qwen-image-edit, wan-vace, z-image, ...]
    model_launchers:
      qwen-image-edit:
        script: scripts/run_omni_qwen_img_edit_fp8.sh
        patch: scripts/pipeline_qwen_image_edit_plus_patch.py
        optimization: { quant: fp8-weight-only, cache_dit: true, taylorseer: true }

routes:
  qwen-image-edit: { primary: omni-vllm }     # FP8 patch is omni-only
  z-image:                                    # multi-tier fallback
    primary: omni-vllm
    fallback: [sglang]
```

### 2. Pool manager — `services/inference/manager.py`

Pure resolver. Loads the YAML, validates it, and turns a model name into an
ordered list of `ResolvedTarget(pool, launcher, is_primary)` candidates.
No Docker, no Ray, no async — fully testable.

```python
from services.inference import PoolManager
mgr = PoolManager.from_yaml()
for target in mgr.resolve("z-image"):
    print(target.pool.name, target.pool.tier, target.is_primary)
# omni-vllm B True
# sglang    C False
```

### 3. Launcher — `services/inference/launcher.py`

Wraps Docker lifecycle (`start`, `stop`, `status`) by invoking the per-model
launch scripts referenced from the YAML. Falls back to synthesized `docker run`
for pools without a script.

### 4. Dispatch bridge — `services/inference/dispatch.py`

This is the **workflow ↔ pool** bridge. The DAG engine calls:

```python
from services.inference.dispatch import resolve_step
plan = resolve_step(service="forge", model="qwen-image-edit", action="edit")
# → DispatchPlan[ DispatchHop(omni-vllm, .../v1/images/edits) ]
```

Each `DispatchHop` carries the full URL, HTTP method, and a TNAP envelope
builder. The workflow engine iterates the plan, skipping unhealthy pools.

The `action` argument is the **preferred** action key from the launcher's
`api:` map (e.g. `"edit"` vs `"generate"`). If a pool doesn't declare the
preferred action, the first declared action is used instead — so a pool
that only exposes `/v1/images/generations` can still serve an `"edit"` step
via that endpoint. `_pick_action(target, preferred, fallback)` implements
this preference cascade.

### 5. Workflow engine integration — `services/workflows/engine.py`

The engine has three ways to invoke pools, in order of abstraction:

1. **`forge` step type** — legacy path. Routes through the Ray Serve Forge
   deployment (VRAM-aware scheduler). Kept for backward compatibility.
2. **Specialized step types** (`img_edit`, `vace_generate`, `ltx_generate`) —
   each has its own executor that crafts service-specific payloads (e.g.
   multipart form-data for Omni) but **resolves URLs via the dispatch bridge**:
   ```python
   from services.inference.dispatch import resolve_step
   api_url = _resolve_api_url("qwen-image-edit", action="edit")
   # → http://127.0.0.1:8093/v1/images/edits  (from inference_pools.yaml)
   ```
   Falls back to a hard-coded legacy map if the pool config is missing.
3. **`pool` step type** *(new)* — generic pool client. Uses `resolve_step()`
   to get an ordered plan, then walks hops in priority order with auto-fallback
   on connection errors or unhealthy pools. The right choice for any new model
   that speaks OpenAI-compatible JSON over HTTP.

### 6. Workflow specs — `config/workflows/*.yaml`

Workflows reference models **by name**. They don't know which pool serves them.
The dispatch bridge resolves that at runtime, so the YAMLs stay portable.

```yaml
# config/workflows/tech_noir_state.yaml
steps:
  edit_state:
    type: img_edit               # specialized executor → dispatch-resolved URL
    model: qwen-image-edit       # ← canonical model name
    params: { ... }
  generate:
    type: pool                   # generic pool dispatch (new)
    service: native
    model: z-image
    params: { ... }
```

### 7. MCP servers — `mcp/dag/` + `mcp/inference/`

Two MCP servers expose the system to web-ui / external clients:

- **`mcp/dag/server.py`** — workflow operations: `list_workflows`,
  `start_workflow`, `get_run_status`, `cancel_run`, `list_native_models`.
- **`mcp/inference/server.py`** — pool operations:
  `list_inference_pools`, `resolve_model`, `get_pool_status`,
  `get_model_optimization`, `start_inference_pool`, `stop_inference_pool`.

A chat UI can ask `resolve_model("z-image")` to learn it's served by
omni-vllm (with FP8+Cache-DiT) and falls back to sglang (1.61s benchmark),
then `start_workflow("native_generate", {"model": "z-image", ...})` to run.

### 8. HTTP routes — `gateway/routes/inference.py`

Same surface area as the MCP server, exposed as REST under `/v1/inference/*`.
Both interfaces consume the same Python modules and YAML config — the web-ui
can use whichever is more convenient per call.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/inference/pools` | All pools in priority order |
| `GET` | `/v1/inference/pools/{name}` | Single pool docker + health status |
| `POST` | `/v1/inference/pools/{name}/start` | Start pool (body: `{"model": "..."}`) |
| `POST` | `/v1/inference/pools/{name}/stop` | Stop and remove container |
| `GET` | `/v1/inference/models` | Every routable model |
| `GET` | `/v1/inference/models/{model}/resolve` | Ordered resolution chain |
| `GET` | `/v1/inference/models/{model}/optimization` | Optimization + benchmark |
| `GET` | `/v1/inference/resolve/{model}` | Alias for resolve |

## CLI — `scripts/inference/pool_ctl.py`

```bash
pool_ctl list                     # all pools in priority order
pool_ctl tier B                   # just Tier B
pool_ctl models                   # all routable models
pool_ctl resolve qwen-image-edit  # full resolution chain + optimization
pool_ctl summary                  # one-line per model
pool_ctl status                   # docker + health for all pools
pool_ctl status omni-vllm         # one pool
pool_ctl start omni-vllm qwen-image-edit
pool_ctl stop omni-vllm
pool_ctl validate                 # check config for warnings
```

## Adding a New Model

1. **Pick a pool** (or create one in `pools:`).
2. **Add the model** to the pool's `models:` list.
3. **Optionally declare a launcher** under `model_launchers:` with script,
   patch, and optimization knobs.
4. **Add a route** under `routes:` with `primary:` and (optional) `fallback:`.
5. Run `pool_ctl validate` — should report no warnings.

That's it. The dispatch bridge, MCP tools, and CLI pick it up automatically.

## Adding a New Pool

1. **Add an entry** under `pools:` with `tier`, `priority`, `image`, `port`,
   `framework`, `models`, `vram_mb`, `health_path`.
2. **Optional**: declare `model_launchers:` for per-model scripts.
3. **Reference** the pool from at least one `routes:` entry.
4. Run `pool_ctl validate`.

## Validation

`PoolManager.validate()` checks for:
- Routes pointing at undefined pools
- Models listed in pools but missing from routes (unreachable)
- Pools serving models not declared in `models:` or `model_launchers:`
- Tier values outside A/B/C/D
- Port collisions across pools

Run `pool_ctl validate` after every config change.

## Tests

```bash
pytest tests/inference/ -v
```

36 tests cover config integrity, tier invariants, per-model optimization
preservation, multi-pool fallback chains, and the dispatch bridge.
