# Tech Noir Ray Migration

## Project Location
- Ray project: `/home/ubuntu/Documents/programs/ray/`
- Central model registry: `/home/ubuntu/Documents/models/`
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Ray version: 2.55.1
- Python: 3.13 (via uv)

## Architecture (validated)
- Ray Serve manages all AI services as deployments
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

## Key API Patterns (Ray Serve 2.55)
- `serve.run(app.bind(), name="x", route_prefix="/x")` deploys an application
- `serve.get_deployment_handle("name", "app")` gets a handle
- `handle.options(method_name="load_model").remote(args)` calls specific methods
- `DeploymentResponse` is awaitable but NOT a coroutine - use `asyncio.new_event_loop().run_until_complete()` in sync code
- `ray.get()` does NOT accept `DeploymentResponse` - only `ObjectRef`
- Unix socket path limit (107 chars) - use `/tmp/ray` not deep project paths

## Key Files
- `services/base.py` - BaseGPUDeployment, SubprocessMixin, CLIToolMixin
- `services/creative/trellis.py` - TRELLIS.2 image-to-3D (subprocess CLI)
- `services/creative/anigen.py` - AniGen rigged 3D (subprocess CLI)
- `services/creative/ace_step.py` - ACE-STEP music generation (subprocess CLI)
- `services/creative/see_through.py` - See-Through layer decomposition (subprocess CLI)
- `services/llm/deployment.py` - llama.cpp subprocess wrapper (auto-loads on first request)
- `gateway/gpu_scheduler.py` - GPU swap coordinator
- `gateway/ingress.py` - Starlette API router (LLM, 3D, music, creative, admin, dashboard routes)
- `gateway/dashboard.py` - GPU metrics collector (nvidia-smi background thread) + API endpoints
- `gateway/dashboard.html` - Single-page GPU dashboard (dark zinc/indigo theme, SVG sparklines)
- `registry/models.py` - ModelRegistry singleton
- `registry/config.py` - Config singleton with dotted key access
- `config/local.yaml` - Machine-specific paths (git-ignored)
- `config/local.yaml.example` - Template with env var overrides
- `config/model_registry.yaml` - All model paths with HF sources
- `scripts/start_cluster.sh` - `ray start --head` (uses .venv/bin/ray)
- `scripts/deploy_services.py` - Deploys all services

## GPU Dashboard (added 2026-04-26)
- Real-time GPU monitoring at `http://localhost:8000/dashboard`
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

## Remaining Work
- [x] Stop old Docker llama-server (done - freed 18.3GB VRAM)
- [x] Deploy all services (14/14 deploy successfully)
- [x] TRELLIS.2 venv + models (9/9 safetensors, 16GB)
- [x] AniGen venv + models (pytorch3d built, 9GB models)
- [ ] Refactor GPU TTS/ASR to CLIToolMixin subprocess pattern
  - Each tool needs its own venv with incompatible torch/CUDA versions
  - IndexTTS, Qwen TTS, VibeVoice, GPT-SoVITS, VibeVoice ASR, Qwen ASR
- [ ] Wire GPU scheduler into actual deployment flow
- [ ] Compute SHA256 hashes for models in registry
- [ ] Hybrid cloud dispatch layer (job router, cloud adapters, cost guardrails)

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
- Media Analysis MCP: git-sync sidecar pulls from GitHub, builds container
  - Running at http://localhost:8101/mcp
  - Source: JayDataEngineer/media-analysis-mcp (private)
- Local Web MCP: same pattern
  - Source: JayDataEngineer/local-web-mcp (public)
- `infra/.env` holds GITHUB_TOKEN (git-ignored)
- `infra/repos/` holds cloned code (git-ignored, managed by git-sync)
- Old clone locations symlinked to `infra/repos/`
