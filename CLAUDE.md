# Tech Noir Ray

Ray-based AI infrastructure orchestrating services (LLM, TTS, ASR, image gen, 3D, music, vision) on a home server with an RTX 4090 (24GB VRAM) + 64GB RAM. **Architecture: k3s + KubeRay** — Ray-first, tiered citizenship.

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
| llm | GPU LLM | llama.cpp server (GGUF models via subprocess) |

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

The **Master Router** (`services/creative/master_router.py`) is infrastructure, not a service — it claims `num_gpus: 1.0` and explicitly `_load()`/`_unload()` heavy GPU models to prevent VRAM collisions on a single RTX 4090. Accessed via route `/forge` with `{"service": "trellis|ace_step|comfyui|hy_motion|moss_soundeffect|anigen|see_through", ...}`.

### Tier 2 — Second-Class Citizens (standalone, scale-to-zero)
Working but not Ray-native. Standalone K8s Deployments, not in RayService.

| Service | Type | Notes |
|---------|------|-------|
| florence2 | GPU Vision | Needs transformers compat patches |

### Tier 3 — Third-Class Citizens (broken/experimental)
Not auto-deployed. Commented out in `serve_config.py`. Uncomment for debugging.

| Service | Issue |
|---------|-------|
| gpt_sovits | Complex sys.path hacks, NLTK issues |
| qwen_asr | Old Qwen model, broken auto_map (replaced by vibevoice.cpp ASR) |
| vibevoice_asr | Microsoft VibeVoice ASR (replaced by vibevoice.cpp) |
| vibevoice (community) | 7B TTS, huge, times out |
| phi4mm | Model not on PVC |

## Architecture: Ray-First, k3s + KubeRay

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
# KubeRay cluster
bash infra/k8s/build_and_import.sh       # Build images + import to k3s
kubectl apply -f infra/k8s/ray-service.yaml  # Deploy/update RayService
kubectl get pods -n ai-services          # Check pod status
kubectl get rayservice tech-noir-ray -n ai-services  # Check serve status

# Model management
task models:list     # Show all models and download status
task models:pull     # Download missing models

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
  vision/       → Florence-2 (Tier 2)
registry/       → Model registry CLI + config (pull from HF, ModelScope, Civitai)
config/         → local.yaml (machine-specific, git-ignored), model_registry.yaml
infra/
  docker/       → Dockerfile.base, Dockerfile.gpu-all, Dockerfile.model-sync
  k8s/          → ray-service.yaml, serve_config.py, build_and_import.sh
sdk/            → Client SDK utilities
boot/           → Service lifecycle (CLI, registry, health checks)
scripts/        → boot_services.sh, test_services_v2.py
vendor/         → Upstream git clones (NEVER EDIT — adapt in services/)
```

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
| `/llm/*` | llama.cpp LLM (GPU, GGUF models) |

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

### Tier 2/3 (commented out in serve_config.py)
| Route | Service |
|---|---|
| `/tts/qwen-tts/*` | Qwen3-TTS legacy (Tier 3, replaced by faster-qwen3-tts) |
| `/tts/vibevoice/*` | VibeVoice Community 7B TTS (Tier 3) |
| `/tts/gpt-sovits/*` | GPT-SoVITS (Tier 3) |
| `/asr/vibevoice/*` | VibeVoice ASR (Tier 3, replaced by vibevoice.cpp) |
| `/asr/qwen/*` | Qwen ASR (Tier 3) |
| `/vision/florence2/*` | Florence-2 vision (Tier 2) |
| `/3d/hy-motion/*` | HY-Motion (Tier 3, now via master router) |

Auth: `X-API-Key` header or `?api_key=` query param. Unset = no auth (dev mode).

## Port Allocation

| Port | Service |
|---|---|
| 10001 | Ray Client |
| 18080 | API Ingress (Starlette) |
| 18265 | Ray Dashboard |
| 18327 | Web MCP (Docker) |
| 18800 | Ray Serve HTTP |

## Boot Procedure

1. Power on → LUKS encrypted drive → Dropbear SSH at `192.168.1.184`
2. `ssh root@192.168.1.184` → `cryptroot-unlock` → type passphrase → OS boots
3. Tailscale auto-starts → server reachable at `100.86.69.57`
4. systemd `tech-noir.service` runs `tech-noir boot` → all services start

## Conventions

- Python 3.12 for Ray worker images (CUDA 12.4 standard), host uses Python 3.13 + **uv** for kubectl
- No co-authored-by in git commits
- Tests preferred — integration style, "prove" over "assert"
- `config/local.yaml` is git-ignored; never commit secrets
- Docker images prefixed `tech-noir/` (e.g. `tech-noir/gpu-all:latest`)
- `vendor/` = upstream git clones (NEVER EDIT) — all adaptation in `services/`
- Source mount (`hostPath`) makes code changes instant on pods, but requires pod restart for Python to pick up changes
- All setup is idempotent — safe to re-run
