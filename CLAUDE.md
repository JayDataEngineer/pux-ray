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
| vibevoice_cpp | GPU TTS+ASR | vibevoice.cpp (C++/GGML) quantized TTS + ASR via subprocess |

**Master Router services** (exclusive GPU, explicit model swapping via `/forge`):

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

The **Master Router** (`services/creative/master_router.py`) is infrastructure, not a service — it claims `num_gpus: 1.0` and explicitly `_load()`/`_unload()` heavy GPU models to prevent VRAM collisions on a single RTX 4090. Accessed via route `/forge` with `{"service": "trellis|ace_step|comfyui|hy_motion|moss_soundeffect|anigen|see_through|llm", ...}`.

### Tier 2 — Available via Forge (not auto-deployed)
Registered in master_router.py HEAVY_SERVICES. Available on demand through `/forge`. Models present on PVC.

| Service | Type | Description | Note |
|---------|------|-------------|------|
| vibevoice_asr | GPU ASR | VibeVoice 7B ASR with diarization (16GB) | Replaced by vibevoice.cpp for Tier 1 |
| vibevoice | GPU TTS | VibeVoice 7B multi-speaker TTS (18.7GB) | Long-form synthesis |
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

**Credentials**: Single source of truth in `config/secrets.env` (gitignored). Synced to k8s via `task secrets:sync`.

**Secrets workflow:**
```bash
cp config/secrets.env.example config/secrets.env   # First time
$EDITOR config/secrets.env                         # Set values
task secrets:sync                                  # Push to all k8s namespaces
```

All deployments reference a single secret name `shared-infra` in their namespace. The sync script (`infra/secrets_sync.py`) reads `config/secrets.env` and creates identical `shared-infra` secrets in infra, mcp, and ai-services namespaces. Deploy tasks (`infra:deploy`, `build_mcp.sh`) run sync automatically.

**Image**: `localhost/tech-noir/postgres-age-vector:latest` — Postgres 16 + AGE + pgvector. Built from `infra/docker/Dockerfile.postgres-age`.

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

**Web Research dependencies** (all in `mcp` namespace):
- Redis 7 (Celery broker/cache)
- SearXNG (metasearch engine)
- Celery worker + beat (background scraping tasks)
- PostgreSQL provided by shared infra (`DATABASE_URL` env var)

**Build & deploy:**
```bash
bash infra/k8s/build_mcp.sh                # Clone repos, build images, import to k3s, deploy
kubectl get pods -n mcp                    # Check MCP pods
kubectl apply -f infra/k8s/mcp/            # Re-apply manifests only
```

**Worker migration** (future): Uncomment `nodeSelector` in manifests, label the worker node with `node-role.kubernetes.io/mcp=true`, re-apply.

### Adding a new MCP server

1. Create Dockerfile in `mcp-servers/<name>/Dockerfile`
2. Create K8s manifest in `infra/k8s/mcp/<name>.yaml` (Deployment + Service)
3. Add Traefik route in `infra/k8s/traefik-ingress.yaml` with PathPrefix + stripPrefix middleware
4. Add image config to `infra/k8s/build_mcp.sh`
5. If it needs secrets, add keys to `config/secrets.env` and reference `secretKeyRef: shared-infra` in the manifest

## Architecture: Ray + MCP, k3s + KubeRay

**Container runtime:** k3s (lightweight k8s) with its own containerd. Images imported via `sudo k3s ctr images import -`. Docker used for builds only.

**GPU scheduling:** NVIDIA Device Plugin. NOT the heavy GPU Operator.

**Ray orchestration:** KubeRay Operator manages Ray head + worker pods from declarative RayService YAML.

**Image standard:** All GPU images inherit from `tech-noir/ray-base:latest` (CUDA 12.4.1 + Python 3.12 + PyTorch 2.6.0 + ray 2.55.1 + vllm-flash-attn + nvdiffrast + torchaudio). CPU images use `python:3.12-slim-bookworm`. Host Python (3.13) is only for `kubectl`.

**Golden Base Image:** `tech-noir/ray-base:latest` contains the expensive-to-compile dependencies (nvdiffrast ~10min) so downstream service images are thin layers that build in seconds. All GPU Dockerfiles use `FROM tech-noir/ray-base:latest`. Flash attention via `vllm-flash-attn` (open source, ABI-compatible with PyPI torch). The official `flash-attn` wheels have CXX11 ABI mismatch with PyPI torch — do NOT use them.

**Storage:** `local-path` provisioner (ships with k3s). Single PVC for models, shared across all pods.

**No more:** HTTPToolMixin, subprocess container management, `runtime_env["container"]`, `compose.workers.yaml`, GPUScheduler, duplicate torch/flash-attn builds.

### Conventions
- Python 3.12 for all Ray worker images (CUDA 12.4 standard)
- `tech-noir/ray-base:latest` is the golden base — all GPU images inherit from it
- Downstream images MUST NOT reinstall torch/torchaudio/flash-attn (use `grep -v` to filter requirements)
- Ray Service YAML is the source of truth (not Python scripts)
- Autoscaling: idle GPU pods killed after 5min to free VRAM
- Custom Ray resources pin deployments to specific worker groups
- SubprocessProxyMixin services use port 9000 (Ray Serve proxy uses 8000)
- Async `_ensure_loaded()` runs model loading in a thread to avoid blocking the event loop
- Images do NOT set ENTRYPOINT — Ray Serve starts the process

### Build & Deploy
```bash
# Build all images and import into k3s (ray-base, gpu-all, model-sync)
bash infra/k8s/build_and_import.sh

# Or individually:
docker build -f infra/docker/Dockerfile.gpu-all -t localhost/tech-noir/gpu-all:latest .
docker save localhost/tech-noir/gpu-all:latest | sudo k3s ctr images import -

# Apply RayService (in-place update, no pod restart)
kubectl apply -f infra/k8s/ray-service.yaml

# Apply networking (Traefik routes + dedicated serve proxy)
kubectl apply -f infra/k8s/ray-serve-proxy.yaml
kubectl apply -f infra/k8s/traefik-ingress.yaml

# Force fresh code pickup (source mount changes need pod restart)
kubectl delete pod -n ai-services -l ray.io/cluster=tech-noir-ray-s8mcd
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
| API Ingress | `http://100.86.69.57:18080` |
| Dashboard | `http://100.86.69.57:18080/dashboard` |
| Studio | `http://100.86.69.57:18080/studio` |
| Ray Dashboard | `http://100.86.69.57:18265` |
| Ray Client | `ray://100.86.69.57:10001` |
| ComfyUI | `http://100.86.69.57:18465` |
| Grafana | `http://100.86.69.57:30080/grafana` |

### Working Remotely

From a dev PC, you can:
- **Edit code** over SSH (VS Code Remote, or just SSH + vim)
- **Call APIs** directly: `curl http://100.86.69.57:18080/llm/v1/chat/completions`
- **Use Ray Client**: `ray.init(address="ray://100.86.69.57:10001")`
- **Monitor** via Ray Dashboard at port 18265
- **Transfer models**: `rsync -avP ./model.gguf user@100.86.69.57:/home/user/Documents/models/LLM/`

The project lives at `/home/user/Documents/programs/ray/` on the server. Models are at `/home/user/Documents/models/`.

## Quick Commands

```bash
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
gateway/        → API ingress (Starlette port 18080), ComfyUI manager
services/       → AI service implementations (Ray Serve deployments)
  base.py       → BaseGPUDeployment, SubprocessProxyMixin
  tts/          → Kokoro, eSpeak, IndexTTS, FasterQwen3TTS, VibeVoiceCpp
  asr/          → Faster-Whisper
  image/        → ComfyUI (subprocess proxy)
  creative/     → TRELLIS, ACE-Step, HY-Motion, MasterRouter

### GPU Scheduling

Only one heavy GPU model runs at a time (24GB VRAM). The **Master Router** (`services/creative/master_router.py`) claims `num_gpus: 1.0` and explicitly swaps heavy models (trellis, ace_step, comfyui, hy_motion, moss_soundeffect, anigen, see_through) with `_load()`/`_unload()` + `torch.cuda.empty_cache()`. Lightweight GPU services (faster_qwen3_tts, index_tts, vibevoice_cpp) coexist with small VRAM footprints.

## Service Development

### Adding a new Ray Serve service

1. Create a deployment class in `services/` inheriting `BaseGPUDeployment`
2. Register it in `infra/k8s/serve_config.py` with `YourDeployment.bind()`
3. Add entry to `infra/k8s/ray-service.yaml` serveConfigV2 with route_prefix and autoscaling
4. If it needs models, add entries to `config/model_registry.yaml`
5. If it's a heavy GPU service (>4GB VRAM), route through the master router instead of a direct deployment

### Configuration

Machine-specific config lives in `config/local.yaml` (git-ignored). Template: `config/local.yaml.example`.

```python
from registry.config import Config
port = Config().get("services.comfyui.port", 18465)
root = Config().models_root
api_key = Config().get("secrets.api_key", "")
```

Env vars override config: `TECH_NOIR_MODELS_ROOT`, `HF_TOKEN`, `TECH_NOIR_API_KEY`, etc.

## API Routes

All proxied through the ingress at port 18080:

### Tier 1 (auto-deployed)
| Route | Service |
|---|---|
| `/tts/kokoro/*` | Kokoro TTS (CPU) |
| `/tts/espeak/*` | eSpeak TTS (CPU) |
| `/tts/faster-qwen3-tts/*` | Faster Qwen3-TTS (GPU, CUDA graphs) |
| `/tts/index-tts/*` | IndexTTS (GPU) |
| `/tts/vibevoice-cpp/*` | vibevoice.cpp TTS+ASR (GPU/CPU, quantized GGUF) |
| `/asr/whisper/*` | Faster-Whisper (CPU) |
### Master Router (exclusive GPU, route `/forge`)
Heavy GPU services share a single RTX 4090 via explicit model swapping. Send `{"service": "<name>", ...}` to `/forge`.

| Service key | Description |
|---|---|---|
| `trellis` | TRELLIS.2 image-to-3D mesh |
| `ace_step` | ACE-Step 1.5 text-to-music |
| `comfyui` | ComfyUI 0.20.1 image generation |
| `hy_motion` | HY-Motion 1.0 text-to-3D motion |
| `moss_soundeffect` | MOSS-SoundEffect 8B text-to-sound |
| `anigen` | AniGen image-to-rigged-3D |
| `see_through` | See-Through anime layer decomposition |
| `llm` | llama.cpp GGUF inference |

### MCP Services (standalone K8s, `mcp` namespace)
| Route | Service |
|---|---|
| `/mcp/media/*` | Media Analysis MCP (CPU, YOLOv8/Florence-2/SAM2) |
| `/mcp/web/*` | Web Research MCP (CPU, search/scrape/extract) |

### Cloud Burst (SkyPilot/SkyServe)
| Route | Service |
|---|---|
| `/overflow/*` | Overflow proxy (local → cloud fallback) |

### Tier 2 (via forge master router)
| Service key | Description |
|---|---|
| `vibevoice_asr` | VibeVoice 7B ASR with diarization |
| `vibevoice` | VibeVoice 7B multi-speaker TTS |
| `phi4mm` | Phi-4-multimodal (text+vision+speech) |

### Tier 3 (blocked — needs Docker image changes)
| Route | Service |
|---|---|
| `/tts/gpt-sovits/*` | GPT-SoVITS (needs GPT_SoVITS package) |

Auth: `X-API-Key` header or `?api_key=` query param. Unset = no auth (dev mode).

## Port Allocation

| Port | Service |
|---|---|
| 80 | Traefik (unified routing: Ray + MCP) |
| 10001 | Ray Client |
| 18080 | API Ingress (Starlette) |
| 18265 | Ray Dashboard |
| 18800 | Ray Serve HTTP |

## Boot Procedure

1. Power on → LUKS encrypted drive → Dropbear SSH at `192.168.1.184`
2. `ssh root@192.168.1.184` → `cryptroot-unlock` → type passphrase → OS boots
3. Tailscale auto-starts → server reachable at `100.86.69.57`
4. systemd `tech-noir.service` runs `tech-noir boot` → all services start

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

- Python 3.12 for Ray worker images (CUDA 12.4 standard), host uses Python 3.13 + **uv** for kubectl
- No co-authored-by in git commits
- Tests preferred — integration style, "prove" over "assert"
- `config/local.yaml` is git-ignored; never commit secrets
- Docker images prefixed `tech-noir/` (e.g. `tech-noir/gpu-all:latest`)
- `vendor/` = upstream git clones (NEVER EDIT) — all adaptation in `services/`
- Source mount (`hostPath`) makes code changes instant on pods, but requires pod restart for Python to pick up changes
- All setup is idempotent — safe to re-run
