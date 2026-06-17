# Tech Noir Inference System — Documentation

## Document Index

| Document | Description |
|----------|-------------|
| **ARCHITECTURE.md** | Complete system architecture: 4-tier pool system, port allocation, model routing, dispatch flow, VRAM budget, Docker image registry |
| **INFERENCE-ENGINES.md** | Per-engine profiling results: benchmarks, configuration, API references, known issues for every inference engine |
| **FP8-WORKFLOW.md** | FP8 quantization deep-dive: Triton fp8e4nv issue, FP8 weight-only pipeline patch approach, ModelOpt format, Cache-DiT, format comparison |
| **MODEL-LIFECYCLE.md** | Model storage layout, quantization format reference, conversion pipeline, deployment checklist, VRAM strategy |
| **TROUBLESHOOTING.md** | All known errors and their fixes: Triton crash, CUDA fork, OOM, missing packages, model format issues |

## Quick Start — Testing a Model

```bash
# ── Qwen-Image-Edit (working) ──
./scripts/run_omni_qwen_img_edit_fp8.sh
curl http://localhost:8093/health
curl -X POST http://localhost:8093/v1/images/edits \
  -F "image=@input.png" -F "prompt=add text"

# ── MOSS SoundEffect (working) ──
docker run -d --gpus all -v /mnt/data/models/audio:/models/audio \
  -p 8050:8081 --name inference-moss \
  forge-reg.local:30500/tech-noir/moss:latest
# Fix missing deps:
docker exec inference-moss pip install diffusers
docker exec inference-moss apt-get install -y build-essential
curl -X POST http://localhost:8050/load -d '{"model":"moss-soundeffect-v2"}'
curl -X POST http://localhost:8050/generate \
  -d '{"prompt":"rain","model":"moss-soundeffect-v2","seconds":3}'

# ── CrispASR (working) ──
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

3. **VRAM is the constraint**: With a single RTX 4090, only ONE large model can run
   at a time. Use auto-gpu-evict system between container swaps.

4. **Container images vs upstream**: Where possible use upstream images directly
   (vllm/vllm-omni, ghcr.io/crispstrobe/crispasr). Custom images are tagged
   under `forge-reg.local:30500/tech-noir/`.

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
