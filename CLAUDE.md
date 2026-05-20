# Tech Noir Ray

Ray-based AI infrastructure orchestrating services (LLM, TTS, ASR, image gen, 3D, music, vision) on a home server with an RTX 4090 (24GB VRAM) + 64GB RAM. **Architecture: k3s + KubeRay + MCP** — Ray handles GPU compute, MCP provides tooling, Traefik unifies routing.

## Tiered Service Architecture

Services are categorized into three tiers based on reliability and maintenance effort:

### Tier 1 — First-Class Citizens (auto-deployed via Ray Serve)
Reliable, tested, deployed by default. Registered in `infra/k8s/serve_config.py`.

**Direct services** (independent Ray deployments):

| Service | Type | Description |
|---------|------|-------------|
| kokoro | CPU TTS | Kokoro 82M multi-voice |
| espeak | CPU TTS | eSpeak-NG phoneme synthesis |
| faster_whisper | CPU ASR | Distil-Whisper large-v3 |
| faster_qwen3_tts | GPU TTS | CUDA graph accelerated Qwen3-TTS 1.7B (5x faster than baseline) |
| index_tts | GPU TTS | IndexTTS v2 neural voice cloning |
| vibevoice_cpp_gpu | GPU TTS+ASR | vibevoice.cpp GGML quantized TTS + ASR, CUDA backend |
| vibevoice_cpp_cpu | CPU TTS+ASR | vibevoice.cpp GGML quantized TTS + ASR, CPU backend |

**Forge services** (VRAM-aware GPU management via `/forge`):

| Service | Type | Description |
|---------|------|-------------|
| trellis | GPU 3D | TRELLIS.2 image-to-3D mesh |
| ace_step | GPU Music | ACE-Step 1.5 text-to-music |
| comfyui | GPU Image | ComfyUI 0.20.1, subprocess proxy |
| hy_motion | GPU Motion | HY-Motion 1.0 text-to-3D motion |
| moss_soundeffect | GPU Audio | MOSS-SoundEffect 8B text-to-sound |
| anigen | GPU 3D | AniGen image-to-rigged-3D |
| see_through | GPU Image | See-Through anime layer decomposition |
| llm | GPU LLM | llama.cpp server (GGUF models via subprocess) |
| avatar | GPU Avatar | Kimodo + FluxRT text-to-avatar pipeline |

The **Forge** (`services/forge.py`) is a VRAM-aware GPU manager that claims `num_gpus: 1.0`, tracks VRAM in MB per service, and allows concurrent GPU services when VRAM permits. Evicts only when needed. Services implement `ForgeService` (3 methods: `load()`, `unload()`, `infer(dict) -> dict`). Accessed via route `/forge` with `{"service": "trellis|ace_step|comfyui|hy_motion|moss_soundeffect|anigen|see_through|llm|avatar|wan2gp|vibevoice_microsoft|vibevoice_community_tts|phi4mm", ...}`.

### Tier 2 — Available via Forge (not auto-deployed)
Registered in `services/forge.py` SERVICE_MAP. Available on demand through `/forge`. Models present on PVC.

| Service | Type | Description | Note |
|---------|------|-------------|------|
| vibevoice_microsoft | GPU ASR | VibeVoice Microsoft — microsoft/VibeVoice-ASR 7B with diarization (16GB) | Replaced by vibevoice.cpp for Tier 1 |
| vibevoice_community_tts | GPU TTS | VibeVoice Community — vibevoice/VibeVoice-7B multi-speaker TTS (18.7GB) | Long-form synthesis |
| phi4mm | GPU Multi | Phi-4-multimodal 5.6B (text+vision+speech) | Needs model download (24GB) |

### Tier 3 — Blocked (needs Docker image changes)
Not auto-deployed. Commented out in `serve_config.py`.

| Service | Issue |
|---------|-------|
| gpt_sovits | Needs GPT_SoVITS package in Docker image |

## Shared Infrastructure (infra namespace)

Shared database and storage services in the `infra` namespace. Other services and machines across Tailscale connect to these via URL.

| Service | Type | Description | Access |
|---------|------|-------------|--------|
| Postgres 16 | Database | Apache AGE (graph) + pgvector (vector) extensions. Multi-database: `langfuse`, `web_research`, `infisical` | Internal: `postgres.infra.svc.cluster.local:5432`, Tailscale: `100.86.69.57:30432` |
| Garage | S3 Storage | Lightweight Rust S3-compatible object store (~27MB). Single-node mode | Internal: `garage-s3.infra.svc.cluster.local:3900`, Tailscale: `100.86.69.57:30390` |
| Langfuse | Observability | LLM tracing and evaluation | `http://100.86.69.57:30080/langfuse` |
| Infisical | Secrets | Self-hosted secret management (MIT). CLI-first, K8s operator for sync | `http://100.86.69.57:30080/infisical` |

**Monitoring stack** (also in `infra` namespace):

| Component | Purpose | RAM |
|-----------|---------|-----|
| VictoriaMetrics | Metrics storage (Prometheus-compatible) | ~300MB |
| Grafana Loki | Log aggregation | ~300MB |
| Vector | Log + metrics collector (DaemonSet, one per node) | ~64MB/node |
| Node Exporter | Hardware metrics per machine (DaemonSet) | ~32MB/node |
| Kube State Metrics | K8s object state (pods, deployments) | ~128MB |
| Grafana | Dashboard UI | ~150MB |

**Grafana access:** `http://100.86.69.57:30080/grafana` (Traefik) or `http://100.86.69.57:30031` (NodePort direct)
**Dashboard import IDs:** `7249` (k3s cluster), `1860` (node), `6336` (pods), `16110` (Ray)

**Deploy monitoring:**
```bash
task infra:monitor         # Deploy the full monitoring stack
task infra:monitor:status  # Check monitoring pods + scrape targets
```

**Credentials**: Single source of truth in `config/secrets.env` (gitignored). Two paths to k8s:

**Secrets workflow:**
```bash
cp config/secrets.env.example config/secrets.env   # First time
$EDITOR config/secrets.env                         # Set values
task secrets:sops                                  # Encrypt + commit (Flux path — recommended)
task secrets:sync                                  # Push directly to k8s (imperative, quick updates)
```

- `task secrets:sops` — AGE-encrypts secrets into `shared-infra.enc.yaml`, commits for Flux to decrypt and apply
- `task secrets:sync` — pushes directly via kubectl (used by `build_mcp.sh` for MCP deployments)

All deployments reference a single secret name `shared-infra` in their namespace.

**Image**: `forge-reg/tech-noir/postgres-age-vector:latest` — Postgres 16 + AGE + pgvector. Built from `infra/docker/Dockerfile.postgres-age`.

**Manage:**
```bash
task infra:build    # Build Postgres image + import to k3s
task infra:deploy   # Deploy all shared infra
task infra:status   # Show pods, services, PVCs
task infra:pg       # Open psql shell
task infra:s3       # List Garage buckets
task infra:down     # Tear down (keeps PVCs)
```

**Adding a new database consumer:**
1. Add database name to the init SQL in `infra/k8s/shared/postgres.yaml` ConfigMap
2. Add the DATABASE_URL or password to `config/secrets.env`
3. Reference via `secretKeyRef: shared-infra` in the consumer's deployment
4. Run `task secrets:sync`

## MCP Servers (Standalone k3s Deployments)

MCP servers run as standard K8s Deployments in the `mcp` namespace — completely separate from Ray. They are lightweight (50-150MB RAM) and don't need Ray's actor overhead. Currently deployed on the GPU node; node selectors ready for migration to a dedicated k3s worker.

| Service | Type | Description |
|---------|------|-------------|
| media-analysis-mcp | CPU Vision | YOLOv8, Florence-2, SAM2, InsightFace, transcription, OCR |
| web-research-mcp | CPU Search | Web scraping, search, crawling, structured extraction |
| equibles-mcp | CPU Finance | SEC filings, stock prices, insider trades, congressional trades, economic indicators, VIX (scale-to-zero via KEDA) |

**Web Research dependencies** (all in `mcp` namespace):
- Redis 7 (Celery broker/cache)
- SearXNG (metasearch engine)
- Celery worker + beat (background scraping tasks)
- PostgreSQL provided by shared infra (`DATABASE_URL` env var)

**Equibles dependencies** (all in `mcp` namespace):
- ParadeDB (dedicated Postgres with `pg_search` + `pgvector` — BM25 indexes require ParadeDB-specific extension)
- Equibles Worker (scrapers: SEC EDGAR, Yahoo Finance, FINRA, FRED, CFTC, Congress)
- KEDA v2 (scale-to-zero operator, ~100MB RAM)
- Source: `vendor/equibles/` (upstream clone, never edited — pull updates with `git pull`)

**Build & deploy:**
```bash
bash infra/k8s/build_mcp.sh                # Clone repos, build images, import to k3s, deploy
kubectl get pods -n mcp                    # Check MCP pods
kubectl apply -f infra/k8s/mcp/            # Re-apply manifests only
```

**Worker migration** (future): Uncomment `nodeSelector` in manifests, label the worker node with `node-role.kubernetes.io/mcp=true`, re-apply.

### Adding a new MCP server

1. Create Dockerfile in `mcp/<name>/Dockerfile`
2. Create K8s manifest in `infra/k8s/mcp/<name>.yaml` (Deployment + Service)
3. Add Traefik route in `infra/k8s/traefik-ingress.yaml` with PathPrefix + stripPrefix middleware
4. Add image config to `infra/k8s/build_mcp.sh`
5. If it needs secrets, add keys to `config/secrets.env` and reference `secretKeyRef: shared-infra` in the manifest

## Architecture: Ray + MCP, k3s + KubeRay

**Container runtime:** k3s (lightweight k8s) with its own containerd. Images pushed to Forge Registry (`forge-reg`) via Traefik NodePort 30500. Docker used for builds only.

**GPU scheduling:** NVIDIA Device Plugin. NOT the heavy GPU Operator.

**Ray orchestration:** KubeRay Operator manages Ray head + worker pods from declarative RayService YAML.

**Image standard:** GPU image `tech-noir/gpu-all:latest` is a multi-stage build with Wan2GP base (CUDA 12.8 + PyTorch 2.10+cu128 + Python 3.10). Stage 2 (`wan2gp-base`) replicates Wan2GP's Dockerfile using `uv` for speed. Stage 3 adds Ray + CUDA extensions + our services. CPU images use `python:3.12-slim-bookworm`. Host Python (3.13) is only for `kubectl`.

**GPU Image Architecture:** `Dockerfile.gpu-all` has three stages: (1a/1b) llama.cpp CUDA 12.8 builders, (2) Wan2GP base with curated deps (mmgp, spacy, misaki, insightface, etc.), (3) final image with Ray, CUDA extensions, git repos, vendor code, project code. All pip installs use `uv` (not pip). torchaudio comes as a binary wheel (no source build needed with stock PyTorch).

**Storage:** `local-path` provisioner (ships with k3s). Single PVC for models, shared across all pods.

**No more:** HTTPToolMixin, subprocess container management, `runtime_env["container"]`, `compose.workers.yaml`, GPUScheduler, duplicate torch/flash-attn builds.

### Conventions
- Python 3.10 for GPU worker (Wan2GP base on Ubuntu 22.04), Python 3.12 for CPU images
- `tech-noir/gpu-all:latest` uses Wan2GP's curated deps — do NOT duplicate packages Wan2GP already provides
- Downstream images MUST NOT reinstall torch/torchaudio/flash-attn (use `grep -v` to filter requirements)
- Ray Service YAML is the source of truth (not Python scripts)
- Autoscaling: idle GPU pods killed after 5min to free VRAM
- Custom Ray resources pin deployments to specific worker groups
- ForgeSubprocessMixin services use port 9000 (Ray Serve proxy uses 8000)
- Async `_ensure_loaded()` runs model loading in a thread to avoid blocking the event loop
- Images do NOT set ENTRYPOINT — Ray Serve starts the process

### Build & Deploy
```bash
# First-time setup: configure K3s containerd to resolve "forge-reg"
sudo mkdir -p /etc/rancher/k3s
sudo cp config/registries.yaml.example /etc/rancher/k3s/registries.yaml
sudo systemctl restart k3s

# Deploy Forge Registry (first time only)
kubectl apply -f infra/k8s/shared/forge-registry.yaml
kubectl apply -f infra/k8s/traefik-config.yaml   # adds registry entrypoint
kubectl apply -f infra/k8s/traefik-ingress.yaml   # adds registry TCP route

# Build all images and push to Forge Registry
bash infra/k8s/build_and_import.sh

# Or individually:
docker build -f infra/docker/Dockerfile.gpu-all -t 100.86.69.57:30500/tech-noir/gpu-all:latest .
docker push 100.86.69.57:30500/tech-noir/gpu-all:latest

# Apply RayService (in-place update, no pod restart)
kubectl apply -f infra/k8s/ray-service.yaml

# Apply networking (Traefik routes + dedicated serve proxy)
kubectl apply -f infra/k8s/ray-serve-proxy.yaml
kubectl apply -f infra/k8s/traefik-ingress.yaml

# Force fresh code pickup (source mount changes need pod restart)
kubectl delete pod -n ai-services -l app.kubernetes.io/name=tech-noir-ray-head
```

## Remote Access (Tailscale)

The server is accessible over Tailscale at `100.86.69.57` (local IP fallback: `192.168.1.184`).

```bash
# SSH into the server
ssh user@100.86.69.57

# If Tailscale is down (e.g. after reboot before unlock), use local IP
ssh user@192.168.1.184

# LUKS unlock after power outage (Dropbear initramfs SSH)
ssh root@192.168.1.184   # passphrase prompt → cryptroot-unlock
```

### Key URLs (Tailscale)

| Service | URL |
|---|---|
| API (all endpoints) | `http://100.86.69.57:30080` |
| Dashboard | `http://100.86.69.57:30080/dashboard` |
| Studio Switcher | `http://100.86.69.57:30080/studio` |
| Wan2GP Studio (MCP UI) | `http://100.86.69.57:30080/studio/` |
| Ray Dashboard | `http://100.86.69.57:30080/ray-dashboard/` |
| Ray Client | `ray://100.86.69.57:10001` |
| ComfyUI | `http://100.86.69.57:30080/comfyui/` |
| Grafana | `http://100.86.69.57:30080/grafana` |

### Working Remotely

From a dev PC, you can:
- **Edit code** over SSH (VS Code Remote, or just SSH + vim)
- **Call APIs** directly: `curl http://100.86.69.57:30080/llm/v1/chat/completions`
- **Use Ray Client**: `ray.init(address="ray://100.86.69.57:10001")`
- **Monitor** via Ray Dashboard at `http://100.86.69.57:30080/ray-dashboard/`
- **Transfer models**: `rsync -avP ./model.gguf user@100.86.69.57:/mnt/data/models/LLM/`

The project lives at `/home/user/Documents/programs/ray/` on the server. Models are at `/mnt/data/models/` (dedicated NVMe).

## Quick Commands

```bash
# Flux GitOps (primary — push to master, Flux handles the rest)
task boot              # Verify k3s + Flux health, start Docker services
task heal              # Force-reconcile all Flux Kustomizations
task status            # Show Flux kustomizations + Docker services + Ray status

# KubeRay cluster
bash infra/k8s/build_and_import.sh       # Build images + import to k3s
kubectl apply -f infra/k8s/ray-service.yaml  # Deploy/update RayService
kubectl get pods -n ai-services          # Check pod status
kubectl get rayservice tech-noir-ray -n ai-services  # Check serve status

# MCP servers
bash infra/k8s/build_mcp.sh             # Build MCP images + deploy
kubectl get pods -n mcp                 # Check MCP pods
kubectl logs -n mcp -l app=media-analysis-mcp  # Media analysis logs
kubectl logs -n mcp -l app=web-research-mcp    # Web research logs

# Model management
task models:list     # Show all models and download status
task models:pull     # Download missing models

# Cloud burst (SkyPilot)
task cloud:setup     # Install SkyPilot + verify cloud access
task cloud:up        # Launch SkyServe endpoint
task cloud:status    # Show cloud status + GPU prices
task cloud:push      # Push images to GHCR
task cloud:down      # Terminate cloud endpoint
task cloud:enable <url>   # Enable cloud burst (set CLOUD_SERVE_URL)
task cloud:disable        # Disable cloud burst (clear CLOUD_SERVE_URL)
task cloud:config         # Show runtime config (timeouts)
task cloud:metrics        # Show Prometheus metrics
task cloud:tune '{"local_timeout":5}'  # Adjust timeout at runtime (no restart)

# Testing
task test            # Run pytest
python scripts/test_services_v2.py  # Integration tests against live cluster
```

## Architecture

```
gateway/        → API ingress (Starlette, routed via Traefik :30080), ComfyUI manager
services/       → AI service implementations (Ray Serve deployments)
  base.py       → BaseGPUDeployment, SubprocessMixin
  tts/          → Kokoro, eSpeak, IndexTTS, FasterQwen3TTS, VibeVoiceCpp
  asr/          → Faster-Whisper
  image/        → ComfyUI (subprocess proxy)
  creative/     → TRELLIS, ACE-Step, HY-Motion, AniGen, SeeThrough
  forge.py      → VRAM-aware GPU manager (replaces MasterRouter + GPUGovernor)
  forge_base.py → ForgeService base class (load/unload/infer)
  model_engine/ → Universal GPU model execution with shared mmgp pool

### GPU Scheduling

Only one heavy GPU model runs at a time (24GB VRAM), but lightweight services can coexist. The **Forge** (`services/forge.py`) claims `num_gpus: 1.0` and tracks VRAM per service in MB. It allows concurrent GPU services when their combined VRAM fits, and evicts the largest loaded service only when a new service needs more VRAM than free. Services implement `ForgeService` with `load()`, `unload()`, `infer(dict) -> dict`. Self-managed services (vram_mb=0, like Wan2GP) always fit alongside other services.

### Model Engine

Universal GPU model execution framework. Every model family gets a handler that decomposes it into `{name: nn.Module}`. mmgp manages VRAM/CPU/RAM placement. Multiple models share a single GPU through a shared mmgp pool — module-level swapping, not model-level eviction.

**Architecture:** `ModelEngine.md` — full design doc with migration priorities and comparison vs Wan2GP.

**Handler contract:** `services/model_engine/base_handler.py` — `BaseHandler` → `load_model()` → `LoadResult(pipeline, pipe, co_tenants)`.

**Executor:** `services/model_engine/executor.py` — `ModelExecutor` owns the GPU, manages shared mmgp pool, LRU eviction when models exceed VRAM budget.

**Pattern:** Every handler follows the 3-file structure:
```
handlers/<family>/
  __init__.py      # BaseHandler implementation + variant metadata
  modules.py       # Load nn.Modules, build pipe dict, extract weights
  orchestrator.py  # Raw forward() calls — the inference logic
```

**Current handlers:**
- `ace_step/` — ACE-Step v1.5 text-to-music (proven, 8-phase generation)
- `wan2gp/` — Wraps Wan2GP vendor handlers (wan, hunyuan, flux — 4 model families)

**Migration targets** (by priority): Wan2GP (done), TRELLIS, MOSS-SoundEffect, HY-Motion, AniGen, See-Through.

## Service Development

### Adding a new Ray Serve service

1. Create a deployment class in `services/` inheriting `ForgeService` from `services/forge_base.py` for GPU services, or a standalone deployment for CPU services
2. Register it in `infra/k8s/serve_config.py` with `YourDeployment.bind()`
3. Add entry to `infra/k8s/ray-service.yaml` serveConfigV2 with route_prefix and autoscaling
4. If it needs models, add entries to `config/model_registry.yaml`
5. If it's a heavy GPU service (>4GB VRAM), add to `SERVICE_MAP` in `services/forge.py` and register in `services/registry.py`

### Configuration

Machine-specific config lives in `config/local.yaml` (git-ignored). Template: `config/local.yaml.example`.

```python
from registry.config import Config
port = Config().get("services.comfyui.port", 18465)
root = Config().models_root
api_key = Config().get("secrets.api_key", "")
```

Env vars override config: `TECH_NOIR_MODELS_ROOT`, `HF_TOKEN`, `TECH_NOIR_API_KEY`, etc.

**ComfyUI workflows** (`config/workflows/comfyui/`): Source-tracked workflow JSON files with a `manifest.yaml` catalog.

```
config/workflows/comfyui/
├── manifest.yaml      # All workflows cataloged with descriptions, upstream names, categories
├── wdc/               # WhatDreamsCost — LTX Director video workflows (5)
├── vnccs/             # VNCCS character pipeline — QWEN + SDXL variants (9)
│   └── sdxl/          # SDXL-only variants (no QWEN guidance)
└── vnccs_utils/       # VNCCS utilities — pose studio, detailer, camera (4)
```

Custom nodes declared in `config/comfyui_extensions.yaml`. To update workflows from upstream: clone repo, copy JSON files, update manifest.

## API Routes

All proxied through Traefik at port 30080:

### Tier 1 (auto-deployed)
| Route | Service |
|---|---|
| `/tts/kokoro/*` | Kokoro TTS (CPU) |
| `/tts/espeak/*` | eSpeak TTS (CPU) |
| `/tts/faster-qwen3-tts/*` | Faster Qwen3-TTS (GPU, CUDA graphs) |
| `/tts/index-tts/*` | IndexTTS (GPU) |
| `/tts/vibevoice-cpp-gpu/*` | vibevoice.cpp TTS+ASR (GGML quantized, CUDA backend) |
| `/tts/vibevoice-cpp-cpu/*` | vibevoice.cpp TTS+ASR (GGML quantized, CPU backend) |
| `/asr/whisper/*` | Faster-Whisper (CPU) |
### Forge (VRAM-aware GPU, route `/forge`)
Heavy GPU services share a single RTX 4090 with VRAM-aware scheduling. Send `{"service": "<name>", ...}` to `/forge`.

| Service key | Description |
|---|---|---|
| `trellis` | TRELLIS.2 image-to-3D mesh |
| `ace_step` | ACE-Step 1.5 text-to-music |
| `comfyui` | ComfyUI 0.20.1 image generation (workflow adapter + raw API proxy) |
| `hy_motion` | HY-Motion 1.0 text-to-3D motion |
| `moss_soundeffect` | MOSS-SoundEffect 8B text-to-sound |
| `anigen` | AniGen image-to-rigged-3D |
| `see_through` | See-Through — anime layer decomposition |
| `llm` | llama.cpp GGUF inference |
| `avatar` | Avatar Pipeline — Kimodo motion gen + FluxRT render |

### MCP Services (standalone K8s, `mcp` namespace)
| Route | Service |
|---|---|
| `/mcp/media/*` | Media Analysis MCP (CPU, YOLOv8/Florence-2/SAM2) |
| `/mcp/web/*` | Web Research MCP (CPU, search/scrape/extract) |
| `/mcp/equibles/*` | Equibles Financial MCP (CPU, SEC/stock/insider/economic data, scale-to-zero) |

### Cloud Burst (SkyPilot/SkyServe)
| Route | Service |
|---|---|
| `/overflow/*` | Overflow proxy (local → cloud fallback) |

### Tier 2 (via Forge)
| Service key | Description |
|---|---|
| `vibevoice_microsoft` | VibeVoice Microsoft — microsoft/VibeVoice-ASR 7B ASR with diarization |
| `vibevoice_community_tts` | VibeVoice Community — vibevoice/VibeVoice-7B multi-speaker TTS |
| `phi4mm` | Phi-4-multimodal (text+vision+speech) |

### Tier 3 (blocked — needs Docker image changes)
| Route | Service |
|---|---|
| `/tts/gpt-sovits/*` | GPT-SoVITS (needs GPT_SoVITS package) |

Auth: `X-API-Key` header or `?api_key=` query param. Unset = no auth (dev mode).

## Port Allocation

| Port | Service |
|---|---|
| 30080 | Traefik (all services — Ray Serve, MCP, monitoring, dashboard) |
| 30090 | Flux Operator UI (NodePort) |
| 30500 | Forge Registry (NodePort) |
| 30432 | Postgres (NodePort) |
| 30390 | Garage S3 (NodePort) |
| 10001 | Ray Client |

## Boot Procedure

1. Power on → LUKS encrypted drive → Dropbear SSH at `192.168.1.184`
2. `ssh root@192.168.1.184` → `cryptroot-unlock` → type passphrase → OS boots
3. Tailscale auto-starts → server reachable at `100.86.69.57`
4. k3s auto-starts via systemd → Flux controllers start
5. systemd `tech-noir.service` waits for k3s → bootstraps Flux → verifies health
6. Flux reconciles all Kustomization layers in dependency order → self-healing

**Flux Kustomization layers** (dependency order):
`namespaces → infra-storage → infra-secrets → helm → infra-services + git → ai-services + mcp → networking`

Each layer has health checks and `retryInterval: 1m` for self-healing. Push to master → Flux auto-syncs within 2 minutes.

**Recovery:** `task heal` force-reconciles all layers. Flux re-applies manifests on health check failure automatically.

## Cloud Burst (SkyPilot)

When local GPU is overloaded (queue full, 503s), the overflow gateway automatically forwards requests to a SkyServe cloud endpoint. Cloud instances run the same Docker image (`ghcr.io/jaydataengineer/tech-noir/gpu-all:latest`) and auto-scale from 0→8→0 based on request queue depth.

**Architecture:**
```
Request → Traefik → Overflow Gateway (k8s Deployment)
                         │
                    Local Ray Serve OK?
                    YES → process locally
                    NO  → SkyServe (cheapest cloud GPU)
```

**Key files:**
- `infra/skypilot/serve.yaml` — SkyServe autoscaling config
- `infra/skypilot/setup.sh` — SkyPilot install + credential check
- `gateway/overflow_proxy.py` — FastAPI proxy (local → cloud fallback)
- `infra/k8s/shared/overflow-gateway.yaml` — K8s Deployment
- `infra/docker/push-images.sh` — Push images to GHCR

**Workflow:**
1. `task cloud:setup` — install SkyPilot, add cloud API keys to `config/secrets.env`
2. `task cloud:push` — push Docker images to GHCR
3. `task cloud:up` — launch SkyServe endpoint
4. `task cloud:enable <url>` — set CLOUD_SERVE_URL on the k8s deployment
5. Requests to `/overflow/*` auto-fallback to cloud when local is overloaded
6. `task cloud:disable` — clear CLOUD_SERVE_URL (stops cloud bursting)
7. `task cloud:tune '{"local_timeout":5}'` — adjust at runtime (no restart)
8. `task cloud:down` — terminate cloud endpoint (scales to zero)

**Metrics (Prometheus — auto-scraped via ServiceMonitor):**
| Metric | Type | Labels | Description |
|---|---|---|---|
| `overflow_requests_total` | Counter | source, status | Requests by local/cloud + 2xx/4xx/5xx |
| `overflow_request_duration_seconds` | Histogram | source | Latency buckets by source |
| `overflow_cloud_enabled` | Gauge | — | 1 if cloud URL configured |

**Grafana:** Import `overflow_*` metrics into the existing FastAPI dashboard
at `http://100.86.69.57:30080/grafana`. Add a panel with:
```
rate(overflow_requests_total{source="cloud"}[5m])
```
to see cloud fallback rate over time.

## Conventions

- Python 3.10 for GPU worker (Wan2GP base), host uses Python 3.13 + **uv** for kubectl
- No co-authored-by in git commits
- Tests preferred — integration style, "prove" over "assert"
- `config/local.yaml` is git-ignored; never commit secrets
- Docker images prefixed `tech-noir/` (e.g. `tech-noir/gpu-all:latest`)
- `vendor/` = upstream git clones (NEVER EDIT) — all adaptation in `services/`
- Source mount (`hostPath`) makes code changes instant on pods, but requires pod restart for Python to pick up changes
- All setup is idempotent — safe to re-run

## Flux Tooling

**Flux Operator** — Official Flux web UI at `http://100.86.69.57:30090`. Kustomization dependency graphs, reconciliation history, click-to-reconcile. Managed via HelmRelease in `infra/flux/helm/releases/infra/flux-operator.yaml`. Source: OCI chart at `ghcr.io/controlplaneio-fluxcd/charts/flux-operator`. NodePort 30090.

**Renovate** — Automated dependency updates (Docker images, Python deps). Runs weekly via Gitea Actions (`.gitea/workflows/renovate.yaml`). Config in `renovate.json`. Custom `forge-reg` images are excluded. Setup: create Gitea token with `repo` scope, add as `RENOVATE_TOKEN` secret. Optional `GH_PAT` for GitHub changelogs.

**Pre-commit** — Validates K8s manifests on commit: kubeconform (schema), yamllint, detect-secrets. Config in `.pre-commit-config.yaml`. Run `pre-commit run --all-files` to validate manually.

```bash
pre-commit install           # Install hooks
pre-commit run --all-files   # Validate all K8s manifests
```
