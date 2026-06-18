# Model Lifecycle — Storage, Conversion & Deployment

## Storage Layout

All models live under `/mnt/data/models/`. The directory structure encodes the
model's purpose, format, and origin.

```
/mnt/data/models/
├── audio/               # Audio models (TTS, sound effects, music)
│   ├── acestep/         # ACE-Step music generation (safetensors format)
│   ├── moss-*/          # MOSS TTS + sound effect models (replaces TangoFlux)
│   └── (tangoflux removed — superseded by moss-soundeffect-v2)
├── image-gen/           # Image generation models
│   ├── comfyui/         # ComfyUI workflow models (262 GB)
│   ├── diffusers/       # Diffusers-format image models
│   └── qwen-image-edit/ # Qwen-Image-Edit variants
│       └── 2511-fp8/    # FP8 base weights + config
├── native/              # Quantized/optimized model variants
│   ├── z-image-turbo-fp8/       # W8A8 Block FP8 (26 GB)
│   ├── z-image-base-fp8/        # W8A8 Block FP8 non-distilled (18 GB)
│   ├── ltx23-fp8-transformer/   # LTX Video ModelOpt FP8 (21 GB)
│   └── qwen2.5-vl-7b-gguf/      # GGUF quantized VL model
├── video/               # Video models
│   └── wan2.1-vace-14b-fp8-diffusers/  # Wan VACE
├── vibevoice-cpp/       # CrispASR GGUF files (25 GB)
├── llm/                 # LLM models (GGUF)
├── tts/                 # TTS output samples
└── cache/               # HuggingFace cache (31 GB)
```

## Model Registry

The `config/model_registry.yaml` file tracks every model on disk with metadata
about its source, format, and deployment status. Registry scripts:

| Script | Purpose |
|--------|---------|
| `registry/audit.py` | Scans disk vs registry, reports discrepancies |
| `registry/gc.py` | Garbage-collects orphaned cache entries |
| `registry/reconcile.py` | Syncs registry to disk state |

## Quantization Formats

### 1. Native vLLM W8A8 Block FP8
Used by: z-image-turbo, z-image-base

```
Format:   FP8 weights + FP8 activations, block-scaled (128×128 blocks)
Kernel:   _w8a8_triton_block_scaled_mm (Triton)
VRAM:     ~1 byte/param → 3B model = ~6 GB (z-image)
Status:   ❌ Broken on Triton 3.6.0 (RTX 4090)
Config:
  quant_method: "fp8"
  activation_scheme: "dynamic" or "static"
  weight_block_size: [128, 128]
```

### 2. ModelOpt FP8
Used by: ltx23-fp8-transformer

```
Format:   NVIDIA ModelOpt format, per-tensor/channel scaling
Kernel:   ModelOpt-specific code path
VRAM:     ~1 byte/param + scaling factors → larger on-disk
Status:   ✅ Works on RTX 4090
Config:
  quant_method: "modelopt"
  config_groups: { group_0: { ... } }
  quant_algo: "FP8"
```

### 3. FP8 Weight-Only (Pipeline Patch)
Used by: qwen-image-edit (via pipeline patch)

```
Format:   FP8 weights on disk, BF16 dequant at runtime
Kernel:   Standard BF16 matmul (no special kernel)
VRAM:     ~1 byte/param storage + ~2 bytes/param for dequant buffer
Status:   ✅ Works on RTX 4090 (bypasses Triton limitation)
Method:   Monkey-patches Fp8Config.get_quant_method → _Fp8WeightOnlyLinearMethod
```

### 4. GGUF (llama.cpp format)
Used by: vibevoice-cpp, llama models, qwen2.5-vl-7b

```
Format:   GGML universal format, supports multiple quant levels
Kernel:   GGML CUDA backend
Status:   ✅ Works (if model has correct tensor names)
```

### 5. BF16 / FP16 (Native)
Used by: cosmos (CPU offload), some diffusers models

```
Format:   Native PyTorch BF16/FP16
Kernel:   Standard PyTorch operations
Status:   ✅ Works but high VRAM usage
```

## Conversion Pipeline

The user's FP8 conversion pipeline transforms full-precision models into
deployable quantized formats:

```
Source model (HF hub or local safetensors)
        │
        ▼
FP8 quantize (vLLM native or ModelOpt tool)
        │
        ├──► Native FP8 (W8A8 Block) ──► Triton kernel dependency (RTX 4090 ❌)
        │
        ├──► ModelOpt FP8 ──► Works on RTX 4090 ✅
        │       │
        │       └──► Larger on-disk, more metadata
        │
        └──► Pipeline patch (FP8 weight-only) ──► Works on RTX 4090 ✅
                │
                └──► Monkey-patches at runtime, needs patch file per model
```

## Deployment Checklist

To deploy a new model to the inference system:

1. **Store model files** in appropriate `/mnt/data/models/<category>/<model-name>/`
2. **Update registry:** `config/model_registry.yaml` with path, format, metadata
3. **Create pipeline patch** (if FP8 weight-only approach): monkey-patches for
   Fp8Config override + CPU offload
4. **Create launcher script:** `scripts/run_omni_<model>.sh` with correct docker args
5. **Update pool config:** `config/inference_pools.yaml` with model entry under
   appropriate pool, including api routes, optimization flags, benchmark data
6. **Build Docker image** (if new service): `infra/docker/Dockerfile.<service>`
7. **Test:** Run container, verify health endpoint, run inference with profiling

## VRAM Allocation Strategy

For models that fit on RTX 4090 (24 GB):

| Model Size | FP8 VRAM | Can Fit? | Strategy |
|-----------|---------|----------|----------|
| < 3B params | ~3 GB | ✅ Yes | No special handling |
| 3B-8B | ~4-8 GB | ✅ Yes | With room for activations |
| 8B-14B | ~8-14 GB | ⚠️ Tight | + VAE/encoder on CPU |
| 14B-20B | ~14-20 GB | ⚠️ Requires patch | FP8 weight-only + CPU offload |
| > 20B | > 20 GB | ❌ No | Needs larger GPU or ModelOpt |

For models that DON'T fit:
1. **FP8 weight-only** — saves ~50% vs BF16, adds CPU offload for text encoder
2. **ModelOpt** — may compress further, different kernel path
3. **CPU offload** — move non-critical layers to CPU RAM
4. **Layerwise offload** — swap blocks CPU↔GPU (slow, not recommended)
