# Tech Noir Ray

## Project Location
- **Local dev PC**: `/home/ubuntu/Documents/programs/ray/` (no GPU)
- **Remote GPU server** (Tailscale: `100.86.69.57`, local IP: `192.168.1.184`): `/home/user/Documents/programs/ray/`
- Models: `/home/user/Documents/models/` (on remote) | GPU: RTX 4090 (24GB) | Ray 2.55.1 | Python 3.13 (uv)

## Remote Server Safety Rules
- **During development**: max 2 concurrent SSH ops, sequential pull->deploy->test, MAX_JOBS=2
- **If SSH times out**: try local IP `192.168.1.184`
- **Goal**: `task boot` goes from bare metal to full cluster. Zero manual setup.
- Boot: LUKS + Dropbear SSH at `ssh root@192.168.1.184` → `cryptroot-unlock`
- systemd: `tech-noir.service` auto-starts on boot
- 64GB swap on `/mnt/data/swapfile`, `vm.overcommit_memory=1`, `RAY_memory_usage_threshold=0.98`

## Architecture → See [architecture.md](architecture.md)
## Service Details → See [services.md](services.md)
## Venv Status → See [venv_status.md](venv_status.md)

## Docker Workers (2026-05-02)
- TRELLIS (CUDA 12.4, port 18401), AniGen (CUDA 12.1, port 18402), VibeVoice (CUDA 12.4, port 18403)
- Docker Compose profiles enforce GPU exclusivity (one at a time)
- GPUScheduler starts/stops containers via `docker compose --profile <name> up/stop`
- DINOv3 fix: inline `sed` in Dockerfile.trellis (not a separate patch file)
- Service deployments use HTTPToolMixin (HTTP POST to localhost:<port>)
- Bare-metal tools (ACE-Step, See-Through, GPT-SoVITS) still use CLIToolMixin

## Docker Build Lessons (2026-05-02)
- `--no-build-isolation` requires `wheel`, `setuptools`, `packaging` pre-installed
- No GPU during `docker build` → must set `TORCH_CUDA_ARCH_LIST="8.9"` (RTX 4090 = sm_89)
- `MAX_JOBS=2` prevents OOM during parallel nvcc compilation
- TRELLIS.2 extension URLs: CuMesh, FlexGEMM → JeffreyXiang (not nv-tlabs/NVlabs)
- o-voxel is bundled inside TRELLIS.2 repo (not standalone), nvdiffrec uses JeffreyXiang fork (renderutils branch)
- nvdiffrast pinned to tag v0.4.0, flash-attn pinned to 2.7.3
- Runtime image needs `gcc libc6-dev` for Triton JIT (FlexGEMM imports triton at load)
- `pipeline.json` has host-specific model paths → `_patch_pipeline_json()` remaps for container
- `MODEL_PATH` must point to dir containing `pipeline.json` (e.g. `/models/TRELLIS.2-4B/ckpts`)
- trimesh `Scene.export(BytesIO)` requires explicit `file_type="glb"` (can't detect from extension)

## Key Files
- `services/base.py` - BaseGPUDeployment, SubprocessMixin, CLIToolMixin, HTTPToolMixin
- `services/creative/trellis.py` - TRELLIS.2 (HTTPToolMixin, Docker port 18401)
- `services/creative/anigen.py` - AniGen (HTTPToolMixin, Docker port 18402)
- `services/tts/vibe_voice.py` - VibeVoice (HTTPToolMixin, Docker port 18403)
- `services/creative/ace_step.py` - ACE-STEP music (CLIToolMixin, `thinking=false`)
- `services/creative/see_through.py` - See-Through (CLIToolMixin subprocess)
- `gateway/gpu_scheduler.py` - GPU swap coordinator with Docker container lifecycle
- `gateway/ingress.py` - Starlette API router (port 18080)
- `registry/cli.py` - Model pull CLI (HF, ModelScope, Civitai sources)
- `registry/config.py` - Config singleton
- `infra/setup/venvs.py` - Bare-metal venv setup only
- `infra/setup/system.py` - System provisioning (apt, sysctl, swap — no CUDA patching)
- `infra/setup/clone.py` - Clone bare-metal tool repos only
- `boot/services.py` - Service registry + lifecycle
- `config/model_registry.yaml` - 36 models, all automated downloads
- `AGENTS.md` - AI agent context

## CLIToolMixin Critical Fixes (2026-05-02)
- `Path.resolve()` follows uv symlinks → use `absolute()` instead
- uv venvs need `VIRTUAL_ENV` + `PYTHONPATH` set explicitly in subprocess env
- `_ensure_loaded()` — lazy init since Ray Serve doesn't call `load_model()` before `__call__`
- Subprocess `oom_score_adj` reset to 0 (Ray sets 1000)

## TRELLIS DINOv3 Dependency
- DINOv3 gated on HF → download from ModelScope (`download: modelscope`)
- `pipeline.json` auto-patched with local paths after model pull
- `image_feature_extractor.py` fix: inline `sed` in Dockerfile.trellis
- Patching automated in registry/cli.py (post-download pipeline.json patch)

## ACE-Step
- Must set `thinking = false` in TOML config to disable LM reasoning
- LM reasoning triggers interactive `input()` call — fails in subprocess

## E2E Test Status (2026-05-02)
- **7 PASSED**: TestLLM (2), TestCPU_TTS, TestCPU_ASR, TestVRAMSwap (2), TestTRELLIS
- **3 FAILED**: TestAniGen (no Docker image), TestACEStep/SeeThrough (VRAM exhaustion after TRELLIS)
- **2 SKIPPED**: TestComfyUI (SDXL workflow timeout — needs dedicated test run)
- TRELLIS Docker worker fully working end-to-end (~5min including model loading)

## Port Allocation (18xxx range)
| Port | Service |
|---|---|
| 18080 | API Ingress (Starlette) |
| 18265 | Ray Dashboard |
| 18399 | llama.cpp Server |
| 18465 | ComfyUI |
| 18327 | Web MCP (Docker) |
| 18401-18403 | Docker workers (trellis/anigen/vibevoice) |
| 18800 | Ray Serve HTTP |

## Remaining Work
- [x] Build TRELLIS Docker worker image on remote
- [x] Test TRELLIS end-to-end (PASSED)
- [ ] Build AniGen Docker worker image on remote
- [ ] Build VibeVoice Docker worker image on remote
- [ ] Wire GPU scheduler into deployment flow (unload before loading different service)
- [ ] Test ComfyUI end-to-end (SDXL workflow needs longer timeout)
- [ ] Verify VibeVoice TTS / GPT-SoVITS
- [ ] Compute SHA256 hashes for models
- [ ] 64GB RAM hardware upgrade
