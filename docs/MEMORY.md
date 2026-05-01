# Tech Noir Ray Migration

## Project Location
- **Local dev PC**: `/home/ubuntu/Documents/programs/ray/` (no GPU currently)
- **Remote GPU server** (Tailscale: `100.86.69.57`, local IP: `192.168.1.184`): `/home/user/Documents/programs/ray/`
- Central model registry: `/home/user/Documents/models/` (on remote)
- GPU: NVIDIA RTX 4090 (24GB VRAM) — on the remote server
- Ray version: 2.55.1
- Python: 3.13 (via uv)

## Remote Server Boot Procedure (LUKS + Dropbear)
- Server has **Full Disk Encryption (LUKS)** — must be unlocked before OS boots
- **Dropbear initramfs SSH**: on reboot, tiny SSH server runs in initramfs
  - SSH to **local IP only** (Tailscale isn't running yet): `ssh root@192.168.1.184`
  - Use `expect` locally to automate: `cryptroot-unlock` → passphrase → boot
- After unlock: OS boots, Tailscale auto-starts, server appears at `100.86.69.57`
- **systemd `tech-noir.service`** auto-starts all services: `ExecStart=... -m boot.cli boot`
- **Samsung 980 PRO firmware**: Updated from 3B2QGXA7 → **5B2QGXA7** (2026-04-30)
  - Root cause of fumagician failing: missing `unzip` package
  - Device order may swap between nvme0/nvme1 after reboot
- Storage: 980 PRO 2TB (OS) + 970 EVO Plus 2TB (backup, mounted at /mnt/data)

## Architecture (validated)
- Ray Serve manages GPU AI services as deployments (compute fabric, NOT HTTP router)
- **Ingress (port 18080)** is the single HTTP router — Starlette, proxies to services
- **Ray Serve (port 18800)** handles deployment lifecycle, not exposed to users
- `num_gpus: 0.01` (fractional) for GPU deployments - gives CUDA access without blocking
- CPU services (espeak, kokoro, whisper) run alongside GPU models with `num_gpus: 0`
- GPUScheduler named actor coordinates model swaps via `handle.options(method_name=...).remote()`
- Subprocess wrapping for llama.cpp and ComfyUI (they're servers, not libs)
- `runtime_env` per service for dependency isolation
- **Creative tools use CLIToolMixin** - called via subprocess with tool's own venv Python
  - Reason: compiled CUDA extensions (flash-attn, o_voxel, pytorch3d) can't be pip-installed dynamically
  - Each tool has incompatible torch/CUDA versions (cu124, cu128, cu130)
  - Model loads fresh per subprocess call (~30-60s overhead, acceptable for batch generation)
  - Paths read from config: `services.creative.{tool}.venv_python/script/working_dir`
- **MCP servers are persistent processes** — NOT Ray deployments
  - Always-on CPU services, no GPU, no lifecycle management needed
  - `scripts/start_mcp.sh` manages start/stop/status/restart
  - Ingress proxies directly to ports (18327 web, 18101 media)
  - Reason: Ray added complexity (serialization, health checks) without value for persistent HTTP servers

## Remote Access (from any Tailscale machine)
- **API Ingress**: `http://100.86.69.57:18080` (all routes: LLM, TTS, ASR, 3D, music, creative, MCP, jobs, dashboard, studio)
- **Ray Dashboard**: `http://100.86.69.57:18265`
- **Ray Client** (Python API): `ray.init(address="ray://100.86.69.57:10001")`
- **SSH**: `ssh user@100.86.69.57` (key-based auth)
- Ray namespace: `tech_noir` (required for scheduler + ingress to find each other)

## Key API Patterns (Ray Serve 2.55)
- `serve.run(app.bind(), name="x", route_prefix="/x")` deploys an application
- `serve.get_deployment_handle("name", "app")` gets a handle
- `handle.options(method_name="load_model").remote(args)` calls specific methods
- `DeploymentResponse` is awaitable but NOT a coroutine - use `asyncio.new_event_loop().run_until_complete()` in sync code
- `ray.get()` does NOT accept `DeploymentResponse` - only `ObjectRef`
- Unix socket path limit (107 chars) - use `/tmp/ray` not deep project paths

## Boot System (added 2026-04-30)
- **`boot/` package** — Python CLI for entire server lifecycle (not just Ray)
- `boot/services.py` — Service registry with typed dataclasses (DOCKER, RAY, PROCESS)
- `boot/health.py` — Health checks: TCP, HTTP, Docker Compose, Ray
- `boot/cli.py` — `tech-noir boot/status/stop` CLI with rich output
- `Taskfile.yml` — `task boot/up/down/status` shortcuts
- `AGENTS.md` — AI context for fresh sessions
- `scripts/tech-noir.service` — systemd unit, calls `tech-noir boot` on startup
- Replaced: `scripts/start_mcp.sh`, `scripts/boot_services.sh`
- Manages: Ray cluster, Ray Serve, ingress, Docker Compose (redshiftdb, MCP, bot, jellyfin)
- **Docker migration**: All compose projects moved from `/home/user/projects/` to `/home/user/Documents/programs/`

## VibeVoice Architecture (important context)
- **Microsoft released 2 VibeVoice repos**:
  1. `microsoft/VibeVoice` — the CODE repo (has inference scripts, custom model classes). Removed the 7B model weights.
  2. Separate Microsoft repo for ASR/lite version WITH model weights
- **vibevoice/VibeVoice-7B** on HuggingFace — community re-upload of the removed 7B model weights (18.7GB)
- So: Microsoft code + community model weights = working TTS
- VibeVoice-7B is **TTS only** (not ASR). ASR is a separate model/deployment.
- Uses CUSTOM model classes: `VibeVoiceForConditionalGenerationInference`, `VibeVoiceProcessor` (NOT AutoModelForCausalLM)
- Requires `transformers==4.51.3` (pinned, conflicts with everything else)
- Requires compiled `flash-attn` for CUDA
- CLIToolMixin subprocess is the correct pattern (dependency isolation)
- Inference script: `demo/inference_from_file.py` with `--model_path`, `--txt_path`, `--speaker_names`, `--output_dir`
- Text format: `Speaker 1: ...\nSpeaker 2: ...` (up to 4 speakers)
- Default voices: Andrew, Ava (built into repo's voices/ directory)
- Config: `services.tts.vibe_voice.{venv_python,script,working_dir}`

## Remote Server Services (restored from 970 backup)
| Project | Type | Containers | Description |
|---|---|---|---|
| local-web-mcp | Docker | 10 | Web MCP (Celery, Postgres, Redis, SearxNG, Caddy, Timescale) |
| media-analysis-mcp | Docker | 1 | Media analysis |
| redshiftdb | Docker | 20 | Infra (Postgres, MongoDB, MinIO, Vault, Zitadel, monitoring, CRM) |
| act-scheduler-bot | Docker | 6 | Telegram bot (aiogram + FastAPI) |
| jellyfin_act | Docker | — | Jellyfin + Nextcloud (needs volume setup) |
| ray-cluster | Ray | 1 | Ray head node (1 GPU, 16 CPUs) |
| ray-serve | Ray | 14 | AI service deployments |
| ingress | Process | 1 | Starlette API gateway (port 18080) |

## Shell → Python Conversion (2026-05-01)
- `infra/setup/` package replaces `infra/setup_venvs.sh`
  - `infra/setup/clone.py` — clone/update all tool repos (replaces `clone_repos.sh`)
  - `infra/setup/venvs.py` — create venvs + build llama.cpp (replaces `setup_venvs.sh`)
  - `infra/setup/__main__.py` — entry point for `python -m infra.setup`
- `boot/services.py` `_start_ray()` now calls `ray start --head` directly (no more `start_cluster.sh`)
- Taskfile updated: `setup:repos` → `python -m infra.setup.clone`, `setup:tools` → `python -m infra.setup all`
- `start_cluster.sh` still exists but no longer called by boot system

## Model Registry — Full IaC (2026-05-01)
- 34 models with automated download sources (32 HF, 2 Civitai)
- 5 skip entries (system packages/placeholders): espeak-ng, maya1, qwen-asr, vibevoice-asr-engine, cache, voices/emma
- ZERO manual download entries
- `task models:pull` downloads everything in one command
- Key sources: unsloth (LLM GGUF), Kijai/LTX2.3_comfy (ComfyUI VAE/TAE), vibevoice/VibeVoice-7B (community TTS weights), microsoft/VibeVoice-ASR (public ASR weights)
- Civitai API download support in registry CLI (`download: civitai`, `source: civitai://model_id`)

## Key Files
- `services/base.py` - BaseGPUDeployment, SubprocessMixin, CLIToolMixin, wait_for_port()
- `services/creative/trellis.py` - TRELLIS.2 image-to-3D (subprocess CLI)
- `services/creative/anigen.py` - AniGen rigged 3D (subprocess CLI)
- `services/creative/ace_step.py` - ACE-STEP music generation (subprocess CLI)
- `services/creative/see_through.py` - See-Through layer decomposition (subprocess CLI)
- `services/llm/deployment.py` - llama.cpp subprocess wrapper (auto-loads on first request)
- `gateway/gpu_scheduler.py` - GPU swap coordinator
- `gateway/ingress.py` - Starlette API router (port 18080, proxies to all services)
- `gateway/studio.py` - Studio switcher backend (STUDIO_APPS registry, switch/release endpoints)
- `gateway/studio.html` - Studio switcher UI (sidebar + iframe, dark zinc/indigo theme)
- `gateway/dashboard.py` - GPU metrics collector + port-based external service checks
- `gateway/dashboard.html` - Single-page GPU dashboard (SVG sparklines)
- `registry/config.py` - Config singleton with dotted key access
- `config/local.yaml` - Machine-specific paths (git-ignored, differs per machine)
- `scripts/start_cluster.sh` - `ray start --head`
- `scripts/deploy_services.py` - Deploys all Ray Serve services
- `scripts/start_mcp.sh` - DELETED (replaced by boot/ package)
- `scripts/boot_services.sh` - DELETED (replaced by boot/ package)
- `boot/services.py` - Service registry (all services on server)
- `boot/health.py` - Health checks (TCP, HTTP, Docker, Ray)
- `boot/cli.py` - CLI: tech-noir boot/status/stop
- `AGENTS.md` - AI agent context for fresh sessions
- `Taskfile.yml` - Task runner (boot, up, down, status, setup)

## GPU Dashboard (added 2026-04-26)
- Real-time GPU monitoring at `http://localhost:18800/dashboard`
- Inspired by DreamServer (Light-Heart-Labs/DreamServer on GitHub)
- Backend: `gateway/dashboard.py` — GPUMetricsCollector (daemon thread, 5s nvidia-smi polling)
  - Rolling 60-sample deque (5-min window) for sparkline history
  - Queries: utilization, VRAM, temp, power, fan speed, GPU processes
  - Also shows GPUScheduler state (current service/model) and Ray Serve deployment status
- Frontend: `gateway/dashboard.html` — single HTML+CSS+JS file, no build step
  - Dark zinc/indigo theme matching DreamServer aesthetic
  - SVG sparkline charts (util, VRAM, temp, power) — no external chart library
  - Services table with status dots (green/yellow/red/gray), GPU/CPU badges
  - 5-second polling interval via fetch API
- API endpoints (public, no auth):
  - `GET /dashboard` — HTML page
  - `GET /dashboard/api/gpu` — current GPU snapshot + scheduler state
  - `GET /dashboard/api/gpu/history` — rolling 5-min samples
  - `GET /dashboard/api/services` — all 15 Ray Serve deployments with status
- Unit tests: 5 new tests in `tests/test_ingress.py::TestDashboardRoutes`

## E2E Test Status (5/5 passing)
- TestLLM::test_chat_simple - auto-loads model, verifies "4" in response
- TestLLM::test_chat_multiturn - multi-turn API structure validation
- TestCPU_TTS::test_espeak - 70KB WAV output
- TestVRAMSwap::test_load_llm_via_handle - VRAM drop/recover verified via nvidia-smi
- TestVRAMSwap::test_cpu_tts_during_gpu_load - CPU works while GPU loaded

## Venv Setup Status (completed 2026-04-26)
- **TRELLIS.2**: venv at `/home/ubuntu/Documents/programs/TRELLIS.2/.venv/` (Python 3.12, torch 2.6.0+cu124)
  - All 9 model safetensors downloaded (16GB total) at `/home/ubuntu/Documents/models/3d/trellis/TRELLIS.2-4B/ckpts/`
  - Extensions: o_voxel, flash_attn, nvdiffrast, CuMesh, FlexGEMM, nvdiffrec
  - CLI wrapper: `services/creative/wrappers/trellis_cli.py`
- **AniGen**: venv at `/home/ubuntu/Documents/programs/AniGen/.venv/` (Python 3.12, torch 2.9.1+cu130)
  - pytorch3d v0.7.9 built from source with pulsar removed (CUDA 13.1 compatibility)
  - CCCL includes from `/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl`
  - All models at `/home/ubuntu/Documents/models/3d/anigen/` (9GB)
  - `import anigen` works: `[SPARSE] Backend: spconv, Attention: flash_attn`

## Hybrid Cloud Dispatch (planned, 2026-04-26)
- **Local GPU = primary** for interactive/real-time work (always preferred)
- **Cloud serverless = burst/overflow** for batch jobs and models that don't fit in 24GB
- Key principle: **situational dispatch**, not fixed model-to-cloud mapping
  - Same model (e.g., ACE-Step) routes local for single interactive, cloud for batch of 20
  - LTX-Video unquantized → cloud only (32GB won't fit); quantized ComfyUI → local
- **Meta-jobs**: AI agent decomposes creative brief into N generation tasks, queues them to cloud
  - Keeps local GPU free for interactive use while batch runs unattended
  - Needs cost guardrails (budget cap per job)
- Job state managed via Ray primitives (Ray Jobs API, named actors, futures) — no external DB
- **Cloud providers evaluated**:
  - RunPod Serverless — most popular, community templates exist, Python SDK
  - Modal — cleaner DX for burst GPU jobs, `@app.gpu()` decorator, plain Python
  - ComfyUI Cloud (RunComfy, Comfy.icu) — upload workflow JSON, get API endpoint
  - LTX official API — direct, billed per second of output
- Cold start penalty: 20-60s on serverless, acceptable for batch/non-interactive

## Research Workflow
- **Gemini + Google Search grounding** for web research and real-time information
- **Claude** for architecture, codebase work, and implementation
- Gemini SDK: `google-genai` (not `google-generativeai`), supports `GoogleSearch` and `UrlContext` tools

## Port Allocation (Tech Noir Ray — 18xxx range)
| Port | Service | Notes |
|---|---|---|
| 18800 | Ray Serve HTTP | All deployment routes |
| 18080 | API Ingress | Starlette gateway, proxies to all services |
| 18265 | Ray Dashboard | Cluster UI |
| 18399 | llama.cpp Server | SubprocessMixin managed |
| 18465 | ComfyUI | SubprocessMixin managed |
| 18327 | Local Web MCP | Docker, persistent |
| 18101 | Media Analysis MCP | Docker, persistent |
| 10001 | Ray Client | gRPC, remote ray.init() |
- All ports use 18xxx range to avoid conflicts with Docker services on the same server.
- Matches `shared-docker-infra` convention (18880, 18443, 25432, etc.).

## Remaining Work
- [ ] Verify all 34 models downloaded on remote (pull running, ~88GB so far)
- [ ] Run `task test:integration` on remote — all creative tool E2E tests
- [ ] Wire GPU scheduler into actual deployment flow
- [ ] Refactor GPU TTS/ASR to CLIToolMixin subprocess pattern
- [ ] Compute SHA256 hashes for models in registry
- [ ] Hybrid cloud dispatch layer (job router, cloud adapters, cost guardrails)
- [ ] Jellyfin/Nextcloud: missing `nextcloud_aio_mastercontainer` Docker volume
- [ ] Loki in redshiftdb keeps restarting (pre-existing)
- [x] Samsung 980 PRO firmware updated to 5B2QGXA7
- [x] Docker services migrated to /home/user/Documents/programs/
- [x] Boot system: boot/ package + Taskfile + AGENTS.md + systemd service
- [x] ACT Scheduler Bot (Telegram) restored and running
- [x] Stop old Docker llama-server (freed 18.3GB VRAM)
- [x] Deploy all services (14/14 deploy successfully)
- [x] TRELLIS.2 venv + models (9/9 safetensors, 16GB)
- [x] AniGen venv + models (pytorch3d built, 9GB models)
- [x] Shell scripts converted to Python (infra/setup/ package)
- [x] Model registry: 100% automated downloads, zero manual entries

## Model Consolidation (completed 2026-04-25)
- All models consolidated to `/home/ubuntu/Documents/models/` (one model, one location)
- Recovered ~32G disk space (83G -> 115G free)
- Symlinks at old locations for backward compatibility
- Key moves: ACE-Step (54G), IndexTTS dedup (8.4G), Z-Image-Turbo (7.2G),
  Qwen TTS from HF cache (9.2G), TRELLIS (3.7G), Kokoro dedup, w2v-bert dedup
- ACE-Step: ACESTEP_CHECKPOINTS_DIR env var prevents auto-download
- Training outputs: kept LoRA adapters only, deleted full checkpoints (8.9G saved)

## Replaced
- Go model-orchestrator -> Ray GPUScheduler
- Docker Compose for AI services -> Ray Serve deployments
- LiteLLM API gateway -> Ray Serve HTTP proxy
- NVML-based VRAM tracking -> Ray resource management + nvidia-smi verification

## Preserved (still in Docker)
- Traefik reverse proxy
- PostgreSQL, MongoDB, Neo4j, MinIO databases
- Langfuse observability

## Infrastructure as Code (infra/docker-compose.yml)
- MCP repos cloned into `infra/repos/` on each machine
- `infra/.env` holds GITHUB_TOKEN (git-ignored)
- Old clone locations symlinked to `infra/repos/`

## Studio Switcher (added 2026-04-30)
- Unified UI for one-click GPU tool swapping at `/studio`
- Backend: `gateway/studio.py` — STUDIO_APPS registry with 17 services
- Frontend: `gateway/studio.html` — sidebar + iframe, auto-polls status
- Switch endpoint handles: stop ComfyUI subprocess → release scheduler GPU → load target
- MCP services show as "persistent" type (always-on, not switchable)
