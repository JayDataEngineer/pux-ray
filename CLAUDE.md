# Tech Noir Ray

Ray-based AI infrastructure orchestrating 14+ services (LLM, TTS, ASR, image gen, 3D, music) on a home server with an RTX 4090. Uses Ray Serve for GPU scheduling, Docker for isolation, and Starlette for unified API ingress.

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
| Grafana | `http://100.86.69.57:3001` |

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
task status          # Show all services
task boot            # Start everything (Ray + Docker + processes)
task boot:ray        # Ray cluster + serve + ingress only
task boot:docker     # Docker Compose services only
task up <name>       # Start specific service
task down            # Stop everything
task stop <name>     # Stop specific service

task models:list     # Show all models and download status
task models:pull     # Download missing models
task test            # Run pytest
task test:integration # Full E2E tests (needs running cluster)
```

CLI: `tech-noir boot` / `tech-noir status` / `tech-noir stop`

## Architecture

```
boot/           → Service lifecycle (CLI, registry, health checks, config)
gateway/        → API ingress (Starlette port 18080), GPU scheduler, ComfyUI manager
services/       → AI service implementations (Ray Serve deployments)
  base.py       → BaseGPUDeployment, SubprocessMixin, CLIToolMixin, HTTPToolMixin
  llm/          → llama.cpp (Docker container)
  tts/          → Kokoro, eSpeak, IndexTTS, Qwen-TTS, VibeVoice, GPT-SoVITS
  asr/          → Faster-Whisper, VibeVoice ASR, Qwen ASR
  image/        → ComfyUI (Docker container)
  creative/     → TRELLIS.2, AniGen, HY-Motion, ACE-Step, See-Through
  mcp/          → Model Context Protocol services
registry/       → Model registry CLI + config (pull from HF, ModelScope, Civitai)
config/         → local.yaml (machine-specific, git-ignored), model_registry.yaml
infra/          → Docker images, setup scripts, compose files
scripts/        → deploy_services.py (Ray Serve deployment)
sdk/            → Client SDK utilities
```

### Service Types

- **RAY** — Deployed via Ray Serve with GPU scheduling. Managed in `scripts/deploy_services.py`.
- **DOCKER** — Docker Compose stacks (infra, apps). Registered in `boot/services.py`.
- **PROCESS** — Bare processes (ingress gateway). Managed by boot system.

### GPU Scheduling

Only one GPU-heavy model runs at a time (24GB VRAM). The `GPUScheduler` (named Ray actor) serializes access — services must acquire the GPU before loading, and unload before another can run. Docker workers use Compose profiles to enforce GPU exclusivity.

## Service Development

### Adding a new Ray Serve service

1. Create a deployment class in `services/`:
   - Inherit `BaseGPUDeployment` + appropriate mixin
   - Use `HTTPToolMixin` for Docker-containerized services
   - Use `CLIToolMixin` for bare-metal subprocess tools (temporary, migrating to Docker)
2. Register it in `scripts/deploy_services.py` with `serve.run()`
3. If Docker-based, add a `Dockerfile.*` in `infra/docker/` and register in `compose.workers.yaml`
4. If it needs models, add entries to `config/model_registry.yaml`
5. Add health check port in `boot/services.py` if standalone

### Adding a new Docker Compose service

1. Add `register(Service(...))` in `boot/services.py`
2. Run `task status` to verify it appears
3. Run `task up <name>` to start it

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

| Route | Service |
|---|---|
| `/llm/*` | LLM (llama.cpp) |
| `/tts/kokoro/*` | Kokoro TTS (CPU) |
| `/tts/espeak/*` | eSpeak TTS (CPU) |
| `/tts/index-tts/*` | IndexTTS (GPU) |
| `/tts/qwen-tts/*` | Qwen3-TTS (GPU) |
| `/tts/vibevoice/*` | VibeVoice TTS (GPU, Docker) |
| `/tts/gpt-sovits/*` | GPT-SoVITS (GPU, Docker) |
| `/asr/whisper/*` | Faster-Whisper (CPU) |
| `/asr/vibevoice/*` | VibeVoice ASR (GPU) |
| `/asr/qwen/*` | Qwen ASR (GPU) |
| `/comfyui/*` | ComfyUI (GPU, Docker) |
| `/3d/trellis/*` | TRELLIS.2 (GPU, Docker) |
| `/3d/anigen/*` | AniGen (GPU, Docker) |
| `/3d/hy-motion/*` | HY-Motion (GPU, Docker) |
| `/creative/see-through/*` | See-Through (GPU, Docker) |
| `/music/ace-step/*` | ACE-Step music (GPU, Docker) |

Auth: `X-API-Key` header or `?api_key=` query param. Unset = no auth (dev mode).

## Port Allocation

| Port | Service |
|---|---|
| 10001 | Ray Client |
| 18080 | API Ingress (Starlette) |
| 18265 | Ray Dashboard |
| 18327 | Web MCP (Docker) |
| 18399 | llama.cpp Server |
| 18401 | TRELLIS Docker worker |
| 18402 | AniGen Docker worker |
| 18403 | VibeVoice Docker worker |
| 18404 | HY-Motion Docker worker |
| 18465 | ComfyUI |
| 18800 | Ray Serve HTTP |

## Boot Procedure

1. Power on → LUKS encrypted drive → Dropbear SSH at `192.168.1.184`
2. `ssh root@192.168.1.184` → `cryptroot-unlock` → type passphrase → OS boots
3. Tailscale auto-starts → server reachable at `100.86.69.57`
4. systemd `tech-noir.service` runs `tech-noir boot` → all services start

## Conventions

- Python 3.13, managed by **uv** (`uv sync`, `uv run`)
- No co-authored-by in git commits
- Tests preferred — integration style, "prove" over "assert"
- `config/local.yaml` is git-ignored; never commit secrets
- Docker images prefixed `tech-noir/` (e.g. `tech-noir/trellis-spz:latest`)
- Port range 18xxx to avoid conflicts
- All setup is idempotent — safe to re-run
