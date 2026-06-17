# Tech Noir Inference System — Documentation

## Document Index

| Document | Description |
|----------|-------------|
| **ARCHITECTURE.md** | Complete system architecture: 4-tier pool system, port allocation, model routing, dispatch flow, VRAM budget, Docker image registry |
| **INFERENCE-ENGINES.md** | Per-engine profiling results: benchmarks, configuration, API references, known issues for every inference engine |
| **FP8-WORKFLOW.md** | FP8 quantization deep-dive: Triton fp8e4nv issue, FP8 weight-only pipeline patch approach, ModelOpt format, Cache-DiT, format comparison |
| **MODEL-LIFECYCLE.md** | Model storage layout, quantization format reference, conversion pipeline, deployment checklist, VRAM strategy |
| **TROUBLESHOOTING.md** | All known errors and their fixes: Triton crash, CUDA fork, OOM, missing packages, model format issues |
| **MOSS-GGUF-MIGRATION.md** | MOSS TTS GGUF migration path: why GGUF, download steps, build instructions, model inventory |

## Registry System (v2.0)

The model registry (`config/model_registry.yaml`) is the **single source of truth** for all models on disk. It documents every model's:

  - **Physical path** on disk (`path:`)
  - **Download source** (`source:`) and method (`download:`)
  - **Size and VRAM estimates** (`size_gb:`, `vram_estimate_gb:`)
  - **Device** (`device: cpu | gpu`)
  - **Status** (`status: active | legacy | pending`)
  - **Pool cross-reference** (`serves:` — which pool-facing name this satisfies)

### served-models section

The `served-models:` section at the bottom maps **pool-facing names** (used in `inference_pools.yaml` routes) to their physical model entries via `pool_ref:`. This is the lookup table for "what pool name → what physical model".

```yaml
# Example: served-models entry
z-image:
  description: Pool-facing name for video/z-image-turbo-fp8
  pool_ref: video/z-image-turbo-fp8
  status: active
  path: native/z-image-turbo-fp8
  size_gb: 16.0
```

### Cross-reference flow

```
Workflow step (service="native", model="z-image")
  → DISPATCH: resolve_step() uses inference_pools.yaml routes
  → POOL: omni-vllm has registry_ref: served-models/z-image
  → REGISTRY: served-models/z-image has pool_ref: video/z-image-turbo-fp8
  → PHYSICAL: video/z-image-turbo-fp8 has path: native/z-image-turbo-fp8
```

### Querying the registry

```bash
# List all served models
python3 -c "import yaml; r=yaml.safe_load(open('config/model_registry.yaml')); [print(k) for k in r.get('served-models',{})]"

# Show physical entry for a pool model
python3 -c "
import yaml
r=yaml.safe_load(open('config/model_registry.yaml'))
ref=r['served-models']['z-image']['pool_ref']
cat,entry=ref.split('/',1)
print(r[cat][entry])
"

# Run storage audit
python3 -m registry.audit --summary

# Reconcile registry vs disk (dry-run)
python3 -m registry.reconcile --dry-run
```

### Audit & GC tools

| Tool | Purpose |
|------|---------|
| `registry/audit.py` | Scan disk against registry — finds stale entries, HF caches, orphans, duplicates |
| `registry/gc.py` | Safe garbage collection — purge HF caches, hardlink dupes, delete orphans |
| `registry/reconcile.py` | Remove registry entries whose paths no longer exist on disk |

## Quick Start — Testing a Model

```bash
# ── Auto-evict GPU before switching models ──
./scripts/auto_evict_gpu.sh --status          # check current VRAM usage
./scripts/auto_evict_gpu.sh                    # stop all inference containers
./scripts/auto_evict_gpu.sh --keep diarization # stop all except diarization

# ── Qwen-Image-Edit (working) ──
./scripts/run_omni_qwen_img_edit_fp8.sh
curl http://localhost:8093/health
curl -X POST http://localhost:8093/v1/images/edits \
  -F "image=@input.png" -F "prompt=add text"

# ── Z-Image Turbo (pipeline patch ready) ──
./scripts/auto_evict_gpu.sh                    # free VRAM from previous model
./scripts/run_omni_z_image_fp8.sh              # start on port 8094
curl http://localhost:8094/health
curl -X POST http://localhost:8094/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/z-image-fp8","prompt":"a cat","n":1,"size":"1024x1024"}'

# ── MOSS SoundEffect (working) ──
./scripts/auto_evict_gpu.sh
docker run -d --gpus all -v /mnt/data/models/audio:/models/audio \
  -p 8050:8081 --name inference-moss \
  forge-reg.local:30500/tech-noir/moss:latest
docker exec inference-moss pip install diffusers
docker exec inference-moss apt-get install -y build-essential
curl -X POST http://localhost:8050/load -d '{"model":"moss-soundeffect-v2"}'
curl -X POST http://localhost:8050/generate \
  -d '{"prompt":"rain","model":"moss-soundeffect-v2","seconds":3}'

# ── CrispASR (working) ──
./scripts/auto_evict_gpu.sh
docker run -d --gpus all -p 8051:8080 \
  -e CRISPASR_AUTO_DOWNLOAD=1 \
  ghcr.io/crispstrobe/crispasr:main-cuda-12 \
  crispasr --server -m auto --backend whisper --auto-download
curl -X POST http://localhost:8051/v1/audio/transcriptions \
  -F "file=@speech.wav" -F "model=whisper"

# ── Check build status ──
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

## Critical Architecture Decisions

1. **`vllm/vllm-omni:latest` vs `fork-v1`**: Always prefer `latest` unless fork-v1
   has a specific feature you need. `latest` has spawn fix; `fork-v1` has fork bug.

2. **Pipeline patches are mandatory**: For FP8 models on RTX 4090, the W8A8 Block FP8
   Triton kernel crashes. The pipeline patch approach (FP8 weight-only dequant to BF16)
   is the only way to run 20B models on 24 GB.

3. **DRY FP8 patch module**: All model pipeline patches now import from
   `scripts/fp8_weight_only_patch.py` (shared `apply_fp8_weight_only_patch()` function).
   This eliminates duplicated patch code across qwen, vace, and z-image pipelines.
   See `scripts/fp8_weight_only_patch.py` for the shared implementation.

4. **VRAM is the constraint**: With a single RTX 4090, only ONE large model can run
   at a time. Use `scripts/auto_evict_gpu.sh` between container swaps.

5. **Container images vs upstream**: Where possible use upstream images directly
   (vllm/vllm-omni, ghcr.io/crispstrobe/crispasr). Custom images are tagged
   under `forge-reg.local:30500/tech-noir/`.

6. **Model Registry v2.0 is the single source of truth**: All model metadata
   (paths, sources, sizes, VRAM estimates) lives in `config/model_registry.yaml`.
   The `inference_pools.yaml` cross-references it via `registry_ref:` fields.
   Registry audit/reconcile/GC tools in `registry/` keep disk in sync.
   The `served-models:` section maps pool-facing names to physical entries.

## GPU Memory State (Typical)

```
Total:    24 GB (RTX 4090)
OS/etc:    2 GB
Usable:   22 GB

Qwen-Image-Edit:  21 GB  ← near full allocation
MOSS:             13 GB  ← can coexist if < 22 GB
CrispASR:          0.2 GB ← always safe
ACE-Step:          8 GB
```
