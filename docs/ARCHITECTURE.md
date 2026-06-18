# Tech Noir Inference Architecture

## Overview

Four-tier priority fallback pool system for GPU-accelerated inference. Each pool is an independent Docker container; models are routed to pools by the dispatch bridge at request time.

```
                  ┌──────────────────────────┐
                  │   Workflow Engine (DAG)  │
                  │  service="forge/model=X" │
                  └──────────┬───────────────┘
                             │
                  ┌──────────▼───────────────┐
                  │   Dispatch Bridge        │
                  │   resolve_step()         │
                  │   PoolManager.resolve()  │
                  └──────────┬───────────────┘
                             │ ordered hop chain
         ┌───────────────────┼────────────────────┐
         ▼                   ▼                    ▼
   ┌──────────┐       ┌──────────┐        ┌──────────┐
   │ Tier A   │       │ Tier B   │        │ Tier C/D │
   │ Special  │──────▶│ Omni-vLLM│───────▶│ SGLang/  │
   │ Dockers  │       │ (DiT)    │        │ Diffusers│
   └──────────┘       └──────────┘        └──────────┘
   priority 10-50     priority 100       priority 200-300
```

## Tier Structure

### Tier A — Specialized Dockers (priority 10-50)
Each service gets its own container with an upstream or near-upstream image.

| Pool | Port | Framework | Models | VRAM | Image Source |
|------|------|-----------|--------|------|-------------|
| moss | 8050 | moss (Python) | TTS, SoundEffect | 6 GB | `forge-reg.local:30500/tech-noir/moss:latest` |
| diarization | 8051 | crispasr (C++) | ASR, Diarization | 1.5 GB | `ghcr.io/crispstrobe/crispasr:main-cuda-12` |
| diarization-turbo | 8055 | crispasr (C++) | ASR turbo | 1.5 GB | Same image, different model GGUF |
| llama | 8052 | llama.cpp (C++) | LLM | 4 GB | `forge-reg.local:30500/tech-noir/gpu-all:latest` |
| llama-bee | 8053 | beellama (C++) | LLM (BeeLlama fork) | 4 GB | Same image, different binary |
| ace-step | 8056 | acetep-cpp (C++) | Music generation | 8 GB | `forge-reg.local:30500/tech-noir/ace-step:latest` |
| comfyui | 8054 | comfyui (Python) | Node-based pipeline | 8 GB | `forge-reg.local:30500/tech-noir/gpu-all:latest` |

### Tier B — Omni vLLM (priority 100)
Single container serving all Diffusion Transformer (DiT) models via `vllm/vllm-omni:latest`.
Port 8093. Uses FP8-weight-only via pipeline patches + Cache-DiT acceleration.

**Key constraint:** Only ONE DiT model can run at a time on a single RTX 4090 (24GB).
Models must be hot-swapped or evicted between requests.

| Model | Params | Quant | VRAM | Patch Required | Cache |
|-------|--------|-------|------|---------------|-------|
| qwen-image-edit | 20B | FP8 weight-only | 21.2 GB | `pipeline_qwen_image_edit_plus_patch.py` | Cache-DiT + TaylorSeer |
| wan-vace | 14B | FP8 weight-only | ~15 GB | `pipeline_wan2_2_vace_patch.py` | TeaCache |
| z-image-turbo | ~3B | W8A8 Block FP8 | ~6 GB | ❌ BLOCKED (Triton 3.6.0) | Cache-DiT |
| z-image-base | ~3B | W8A8 Block FP8 | ~6 GB | ❌ BLOCKED | Cache-DiT |
| wan-t2v | 14B | FP8 | ~15 GB | TBD | TBD |
| wan-i2v | 14B | FP8 | ~15 GB | TBD | TBD |
| cosmos | 8B | BF16 CPU offload | ~4 GB GPU | N/A | N/A |

### Tier C — SGLang (priority 200)
Single container at port 8081. Fallback for z-image when omni-vllm is occupied.
Also serves ideogram4 and ltx-video natively.

| Model | Quant | Notes |
|-------|-------|-------|
| ideogram4 | NF4 (bitsandbytes) | 16 GB on 24 GB card |
| z-image-turbo | FP8 | Fallback only (user prefers omni-vllm) |
| z-image-base | FP8 | Fallback only |
| ltx-video | ModelOpt FP8 | Two-stage: 1st stage + 2nd stage |

### Tier D — Diffusers (priority 300)
Catch-all at port 8095 for CPU-friendly and small models.

| Model | Type | Notes |
|-------|------|-------|
| kimodo | 3D Motion | Viser interactive |
| kokoro | TTS | CPU-capable |
| see-through | Anime | Layer decomposition |

## Model Resolution Flow

1. Workflow step specifies `model:` name
2. `PoolManager.resolve(model)` → ordered list of pool targets
3. Primary pool is tried first; fallback chain is walked if unhealthy
4. `ResolvedTarget` includes pool host:port, action URL, payload envelope
5. Each hop is tried in order until one returns 200

### Route Table Example
```yaml
z-image:
  primary: omni-vllm       # Tier B — preferred
  fallback: [sglang]        # Tier C — when omni occupied
```

## Hardware

- **1× RTX 4090 (24 GB)** — all GPU inference shares this single card
- **CPU** — text encoders, small models, CPU offload layers
- **RAM** — 64+ GB system memory for model storage + CPU offload

### VRAM Budget (RTX 4090: 24 GB total, ~22 GB usable)
```
OS/display              2 GB
Qwen-Image-Edit        21 GB  ← occupies nearly all VRAM
MOSS SoundEffect       13 GB  ← can share if < 22 GB
CrispASR                1.5 GB ← coexists with anything
ACE-Step                8 GB
Diffusers catch-all     8 GB
```

Containers MUST be evicted between large-model swaps (see auto-gpu-evict system).

## Docker Image Registry

All images served from `forge-reg.local:30500/tech-noir/` or upstream registries.

| Image | Tag | Source |
|-------|-----|--------|
| `vllm/vllm-omni` | `latest` (0.22.0) | Upstream. Used for working DiT models. |
| `tech-noir/vllm-omni` | `fork-v1` (0.1.dev1) | Custom build. Has Triton 3.6.0 / fork issues. |
| `tech-noir/moss` | `latest` | Python MOSS server, built from `infra/docker/Dockerfile.moss` |
| `tech-noir/asr` | `latest` | CrispASR, tagged from `ghcr.io/crispstrobe/crispasr:main-cuda-12` |
| `tech-noir/sdcpp-vace` | `latest` | Stable Diffusion.cpp for VACE |
| `lmsysorg/sglang` | `latest` | Upstream SGLang |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_omni_qwen_img_edit_fp8.sh` | Launch qwen FP8 weight-only container |
| `scripts/launch_qwen_img_edit_fp8.py` | Launcher with OmniDiffusionConfig env-override patch |
| `scripts/pipeline_qwen_image_edit_plus_patch.py` | **Critical:** monkey-patches Fp8Config → weight-only |
| `scripts/omni_patch_fork.py` | Blocks `mp.set_start_method("fork")` for fork-v1 compat |
| `scripts/run_omni_14b.sh` | Launch wan-vace 14B container |
| `scripts/run_omni_z_image_fp8.sh` | Launch z-image-turbo (BLOCKED on fork-v1) |
| `infra/docker/Dockerfile.acetep` | ACE-Step C++ build + deploy |
| `infra/docker/Dockerfile.asr` | CrispASR (thin wrapper, uses upstream image) |

## Port Allocation

| Port | Service | Container Name |
|------|---------|---------------|
| 8050 | MOSS | inference-moss |
| 8051 | CrispASR base | inference-diarization |
| 8052 | llama.cpp | inference-llama |
| 8053 | BeeLlama | inference-llama-bee |
| 8054 | ComfyUI | inference-comfyui |
| 8055 | CrispASR turbo | inference-diarization-turbo |
| 8056 | ACE-Step | inference-ace-step |
| 8093 | omni-vllm (DiT) | inference-omni-vllm |
| 8081 | SGLang | inference-sglang |
| 8095 | Diffusers | inference-diffusers |
