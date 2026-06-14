# SGLang Diffusion — Current Status

> **Date:** 2026-06-14
> **Status:** Installed and starts, but CANNOT benchmark on shared pod

---

## What We Confirmed

### ✅ SGLang Diffusion works
- Installed: sglang 0.5.13 + sgl-kernel 0.4.3 + flashinfer 0.6.12 + cache-dit 1.3.0
- Server starts and loads FLUX.1-schnell
- Recognizes `--model-type diffusion` 
- API endpoint responds on port 30010
- Health check returns OK

### ⚠️ SGLang falls back to diffusers backend for FLUX
```
"Could not resolve native configuration for model '/models/flux-schnell'. 
Falling back to diffusers backend."
"Diffusers version: 0.30.0.dev0"
```
SGLang uses its OWN bundled diffusers (0.30.0.dev0, older than our 0.37.0).
This means it's NOT using sgl-kernel optimizations for FLUX.
The 1.15-1.5x speedup may only apply to models with native SGLang support.

### ❌ Cannot benchmark on shared pod
- Forge service (Ray Serve replica) auto-restarts within seconds of being killed
- Consumes ~20-23GB VRAM immediately
- SGLang warmup fails with OOM
- Confirmed: SGLang needs its OWN container/pod (as deep research warned)

### sgl_kernel compatibility notes
- sgl_kernel has sm90 (Hopper) and sm100 (Blackwell) only
- RTX 4090 is sm89 (Ada Lovelace) — no pre-compiled kernels
- Fixed by setting LD_LIBRARY_PATH for CUDA 13 nvrtc
- Kernel loads but may not be optimized for SM89

---

## What's Needed to Benchmark SGLang

### Option 1: Dedicated SGLang pod (RECOMMENDED)
```yaml
# New Kubernetes pod with:
# - No forge/Ray Serve service
# - SGLang pre-installed
# - Exclusive GPU access
# - LD_LIBRARY_PATH configured
```

### Option 2: Disable forge in Ray Serve config
- Set forge deployment min_replicas=0 AND max_replicas=0
- Or use `ray serve shutdown` before running SGLang
- Risk: forge comes back when Ray Serve restarts

### Option 3: Run SGLang on a different machine
- Separate GPU node without Ray Serve
- Clean benchmark environment

---

## Environment Variables Required

```bash
# Required for sgl_kernel to load
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# Prevent OOM during model loading
export SAFETENSORS_DISABLE_MMAP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## SGLang Serve Command

```bash
sglang serve \
    --model-path /models/flux-schnell \
    --port 30010 \
    --host 0.0.0.0 \
    --model-type diffusion \
    --skip-server-warmup  # skip warmup (do manual warmup instead)
```

## API Endpoints (OpenAI-compatible)

```
POST /v1/images/generations    # Image generation
POST /v1/videos/generations    # Video generation  
POST /release_memory_occupation # Sleep (VRAM → 250-400MB)
POST /resume_memory_occupation  # Wake (VRAM restored in ~0.5s)
GET  /health                    # Health check
```

---

## IaC Updates Made

- `Dockerfile.wan2gp-unified`: Added `torchao`
- `Dockerfile.gpu-all`: Added `torchao`
- `pip install sglang[diffusion]` on current pod (not in Dockerfile yet)

### Still needed in IaC:
1. SGLang in a SEPARATE Dockerfile (not the forge container)
2. LD_LIBRARY_PATH in the SGLang container's entrypoint
3. CUDA 13.0 nvrtc libraries
4. Model cache volume mount (/models)

---

## 10. FIRST SUCCESSFUL SGLANG BENCHMARK (2026-06-14)

### LTX-Video 2B via SGLang Diffusion — WORKING

**Configuration:**
- Model: LTX-Video 2B (`Lightricks/LTX-Video`)
- Server: `sglang serve --model-type diffusion --server-warmup false`
- GPU: RTX 4090 (24GB)
- Generation: 25 steps, 768×512, 57 frames

**Results:**

| Metric | Cold | Warm (avg of 3) |
|--------|------|-----------------|
| Total time | 18.04s | **15.03s** |
| Inference time | 11.06s | **11.11s** |
| Peak VRAM | 12,796 MB | **9,132 MB** |
| Variance | — | ±0.13s |

**Key observations:**
1. Only 9.1GB VRAM for video generation — half the card free
2. Extremely consistent (±0.13s across 3 warm runs)
3. Only 3s cold start overhead (18s cold vs 15s warm)
4. SGLang falls back to diffusers backend (0.30.0.dev0) — no native LTX kernels
5. API is async: POST creates job, GET polls for status, GET content downloads

**Serving command that works:**
```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export SAFETENSORS_DISABLE_MMAP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
    --model-path /models/ltx-video \
    --port 30010 \
    --host 0.0.0.0 \
    --model-type diffusion \
    --server-warmup false
```

**Critical learnings:**
1. `--server-warmup false` is REQUIRED for LTX (warmup tries to pass `image` arg → IndexError)
2. `--skip-server-warmup` is for LLM server only, NOT diffusion server
3. FLUX.1 (12.5B, 23GB BF16) does NOT fit — SGLang has no layerwise offload for FLUX
4. LTX-Video (2B, ~7GB) fits easily — 9GB peak VRAM with 19GB headroom
5. SGLang's bundled diffusers (0.30.0.dev0) handles LTX via diffusers fallback
6. Killing SGLang with `kill -9` leaks VRAM — always use clean pod restart
7. The `dit_cpu_offload: true` default works for LTX on 24GB VRAM

**API Endpoints (verified):**
```
POST /v1/videos              → Create video (async, returns job ID)
GET  /v1/videos/{id}         → Poll status (pending → processing → completed/failed)
GET  /v1/videos/{id}/content → Download completed video
POST /v1/images/generations  → Image generation (FLUX etc.)
GET  /health                 → Health check
```
