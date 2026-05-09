# Tech Noir Server — AI Agent Context

## What is this?

Tech Noir is a home server running multiple application stacks on Ubuntu with an RTX 4090. This repo (`ray/`) contains the **boot and service management system** for the entire server, plus the AI compute services.

**Server**: Tailscale `100.86.69.57` | Local IP `192.168.1.184` | User `user`
**GPU**: NVIDIA RTX 4090 (24GB VRAM)
**OS**: Ubuntu with LUKS full disk encryption (Dropbear SSH unlock)
**Storage**: Samsung 980 PRO 2TB (OS + projects) + Samsung 970 EVO Plus 2TB (backup, mounted at /mnt/data)

## Quick Commands

```bash
task status          # Show ALL services on the server
task boot            # Start everything (Ray + Docker + processes)
task boot:ray        # Start Ray cluster + serve + ingress only
task boot:docker     # Start all Docker Compose services only
task up <name>       # Start specific service
task down            # Stop everything
task stop <name>     # Stop specific service
```

CLI equivalent: `tech-noir boot` / `tech-noir status` / `tech-noir stop`

## Server Stacks

The server runs three types of services, all managed from `boot/services.py`:

### 1. Ray Serve (AI Compute)
GPU-accelerated AI services managed by Ray Serve with a Starlette ingress on port 18080.
- LLM (llama.cpp), TTS (Kokoro, IndexTTS, Qwen-TTS), ASR (Faster-Whisper)
- Image gen (ComfyUI), 3D (TRELLIS, AniGen), Music (ACE-Step), Creative (See-Through)
- GPU scheduler coordinates model swaps (only one GPU model at a time, 24GB VRAM)
- Web UIs: Dashboard (`/dashboard`), Studio (`/studio`)

### 2. Docker Compose (Infrastructure + Apps)
Multiple Docker Compose projects in `/home/user/Documents/programs/`:
- **redshiftdb** — Full app stack: PostgreSQL, MongoDB, MinIO, Vault, Zitadel (auth), Prometheus, Grafana, Loki, Tempo, CRM app, UI, monitoring agents
- **local-web-mcp** — Web content MCP server with Celery workers, Postgres, Redis, SearxNG, VPN, Caddy, TimescaleDB
- **media-analysis-mcp** — Media analysis service
- **act-scheduler-bot** — Telegram bot (aiogram + FastAPI + PostgreSQL + Redis)
- **jellyfin** — Jellyfin media server + Nextcloud AIO

### 3. Persistent Processes
- **ingress** — Starlette API gateway on port 18080 (proxies to Ray Serve and MCP servers)

## Directory Layout

```
/home/user/Documents/programs/
├── ray/                    # This repo — boot system + AI services
│   ├── boot/               #   Service lifecycle (manages ENTIRE server)
│   │   ├── cli.py          #     CLI: tech-noir boot/status/stop
│   │   ├── services.py     #     Service registry (all services on server)
│   │   ├── health.py       #     Health checks (TCP, HTTP, Docker, Ray)
│   │   └── config.py       #     Resolves paths from Config
│   ├── gateway/            #   API ingress + web UIs
│   ├── services/           #   AI service implementations
│   ├── registry/           #   Config + model registry
│   ├── config/local.yaml   #   Machine-specific paths (git-ignored)
│   ├── Taskfile.yml        #   Task runner
│   ├── AGENTS.md           #   This file
│   └── pyproject.toml      #   uv-managed, entry points: tech-noir, ray-noir
├── local-web-mcp/          # Web MCP (Docker Compose, 10 containers)
├── media-analysis-mcp/     # Media MCP (Docker Compose)
├── redshiftdb/             # CRM + infra (Docker Compose, 20 containers)
│   ├── ops/docker/         #   compose.dev.yaml
│   └── ...
├── act-scheduler-bot/      # Telegram bot (Docker Compose)
├── jellyfin_act/           # Jellyfin + Nextcloud (Docker Compose)
├── media_server/           # Media server (Next.js + Zitadel)
├── zitadel/                # Zitadel identity platform
├── terraform-provider-zitadel/
├── jellyfin-plugin-sso/
├── postal/                 # Email infrastructure
└── models/                 # Model files (LLM weights, etc.)
```

## Service Registry

All services are registered in `boot/services.py` as `Service` dataclasses. Each has:
- `name` — unique identifier (used in `task up <name>`)
- `type` — `DOCKER`, `RAY`, or `PROCESS`
- `working_dir` — where to start/stop
- `port` — for health checks
- `depends_on` — started before this service

### Current Services

| Name | Type | Port | Description |
|---|---|---|---|
| local-web-mcp | Docker | 18327 | Web content MCP (Celery, Postgres, Redis, SearxNG, Caddy) |
| media-analysis-mcp | Docker | 18101 | Media analysis service |
| redshiftdb | Docker | — | Full infra (Postgres, MongoDB, MinIO, Vault, Zitadel, monitoring, CRM, UI) |
| act-scheduler-bot | Docker | 8621 | Telegram bot (aiogram + FastAPI) |
| jellyfin | Docker | — | Jellyfin + Nextcloud AIO |
| ray-cluster | Ray | 18265 | Ray head node (1 GPU, 16 CPUs) |
| ray-serve | Ray | 18800 | Ray Serve deployments (14 AI services) |
| ingress | Process | 18080 | API gateway (proxies all routes) |

### How to Add a New Service

1. Open `boot/services.py`
2. Add a `register(Service(...))` call:
   ```python
   register(Service(
       name="my-new-service",
       type=ServiceType.DOCKER,       # or ServiceType.RAY / ServiceType.PROCESS
       working_dir=f"{PROGRAMS}/my-service",
       port=9000,                      # optional, for health checks
       compose_file="compose.yml",     # only for Docker, if not docker-compose.yml
   ))
   ```
3. Run `task status` to verify it appears

## GPU Governor — VRAM Coordination

All heavy GPU services (trellis, ace_step, moss_soundeffect, anigen, see_through, hy_motion, comfyui, llm) are independent Ray Serve deployments with `num_gpus: 0` coordinated by the **GPUGovernor** actor in `gateway/gpu_governor.py`.

- Governor holds a lease for whichever heavy service is currently loaded
- Before loading a new service, Governor proactively evicts the current holder (calls `unload_model()`)
- Lightweight services (kokoro_tts, index_tts, vibevoice_cpp, etc.) coexist without leases
- `num_gpus: 0` on all deployments — Governor manages VRAM, not Ray's GPU ledger

### Serialization Note

Ray's `@serve.deployment` has a serialization check that fails for some classes when imported from module context (`GenericModule` / `_Ops` errors). The workaround is to define wrapper subclasses in the module where the `@serve.deployment` decorator is applied. `serve_config.py` uses this pattern for `moss_soundeffect` and `anigen`.

### Docker Images (KubeRay)

The KubeRay RayService requires `localhost/tech-noir/gpu-all:latest` in the in-cluster registry. Build:
```bash
docker build --network=host -f infra/docker/Dockerfile.base -t tech-noir/ray-base:latest .
docker build --network=host -f infra/docker/Dockerfile.gpu-all -t localhost/tech-noir/gpu-all:latest .
docker build -f infra/docker/Dockerfile.model-sync -t tech-noir/model-sync:latest .
docker tag tech-noir/model-sync:latest localhost/tech-noir/model-sync:latest
# Push to in-cluster registry (kubectl port-forward or direct)
```

Ray-base build requires `--network=host` because `git clone` in Docker can't reach GitHub otherwise.

### Host Testing (no Docker)

```bash
uv run ray start --head --num-gpus=1
uv run python deploy_all.py
# All 15 services available at http://localhost:8000/{route_prefix}
```

## Configuration

Config lives in `config/local.yaml` (git-ignored, machine-specific). Access via:
```python
from registry.config import Config
port = Config().get("services.comfyui.port", 18465)
root = Config().models_root
```

## Key Access Points (Tailscale network)

- **API Ingress**: http://100.86.69.57:18080 (LLM, TTS, ASR, 3D, music, creative, MCP, jobs, dashboard, studio)
- **Dashboard**: http://100.86.69.57:18080/dashboard
- **Studio**: http://100.86.69.57:18080/studio
- **Ray Dashboard**: http://100.86.69.57:18265
- **Ray Client**: `ray.init(address="ray://100.86.69.57:10001")`
- **Grafana**: http://100.86.69.57:3001
- **Prometheus**: http://100.86.69.57:9090
- **MinIO**: http://100.86.69.57:9002
- **Zitadel**: http://100.86.69.57:8082

## Boot Procedure (power outage / reboot)

1. Power on → LUKS encrypted drive → Dropbear initramfs SSH at `192.168.1.184`
2. SSH as root → `cryptroot-unlock` → type passphrase → OS boots
3. Tailscale auto-starts → server at `100.86.69.57`
4. systemd `tech-noir.service` calls `tech-noir boot` → all services start automatically

## System Prerequisites (zero-to-running)

On a fresh Ubuntu 26.04 install, run these in order:

```bash
# 1. System provisioning (apt packages, CUDA header patch, sysctl, swap)
sudo python -m infra.setup system --fix

# 2. Clone all tool repos
python -m infra.setup.clone

# 3. Download all models (HF + ModelScope + Civitai)
task models:pull

# 4. Set up bare-metal tool venvs + llama.cpp build
python -m infra.setup all

# 5. Build Docker worker images (TRELLIS, AniGen, VibeVoice)
python -m infra.setup docker

# 6. Start everything
task boot
```

All steps are idempotent — safe to re-run. No manual steps required.

## Adding / Moving Projects

Projects live in `/home/user/Documents/programs/`. When adding a new project:
1. Clone or restore to that directory
2. If it has a Docker Compose file, register it in `boot/services.py`
3. If it needs to start on boot, add it to the service registry with appropriate type
4. Run `task status` to confirm it's recognized
5. Run `task up <name>` to start it
