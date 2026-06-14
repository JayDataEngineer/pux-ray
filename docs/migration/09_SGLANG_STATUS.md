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
