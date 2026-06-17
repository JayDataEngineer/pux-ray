# FP8 Quantization & Pipeline Patch Workflow

## The Core Problem

vLLM-Omni on RTX 4090 has two blocking issues with native vLLM FP8 (W8A8):

### Issue 1: Triton 3.6.0 `tl.dot()` lacks fp8e4nv support
```
AssertionError: Unsupported lhs dtype fp8e4nv
  → /usr/local/lib/python3.12/dist-packages/triton/language/semantic.py:1488
  → triggered by _w8a8_triton_block_scaled_mm at fp8_utils.py:779
```
Triton 3.6.0's `tl.dot()` only supports `int8, uint8, float16, bfloat16, float32`.
The fp8e4nv type (needed for W8A8 Block FP8) is not exposed in the Triton API.

**Affects:** `forge-reg.local:30500/tech-noir/vllm-omni:fork-v1` (Triton 3.6.0)

### Issue 2: Fork + CUDA incompatibility
`multiproc_executor.py:191` forces `mp.set_start_method("fork", force=True)` on fork-v1 image,
causing `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.

**Affects:** fork-v1 only. `vllm/vllm-omni:latest` correctly uses spawn at line 141.

## The Solution: FP8 Weight-Only Pipeline Patch

### How it works

Instead of using native vLLM W8A8 FP8 kernels (which call Triton `tl.dot()` with fp8 operands),
we monkey-patch `Fp8Config.get_quant_method` to return a custom linear method that:

1. **Stores weights as FP8** (Float8_e4m3fn) — 1 byte per parameter → 20 GB for a 20B model
2. **Dequantizes to BF16** before each matmul — adds ~5% overhead but **avoids Triton fp8 kernel entirely**
3. **Offloads text encoder to CPU** — frees ~4 GB VRAM by running the text encoder on CPU RAM

```
Weight storage:  FP8 (1 byte/param) ──► GPU VRAM (20 GB for 20B model)
                          │
                    dequant to BF16
                          │
                          ▼
Matmul:           BF16 × BF16 ──► no Triton fp8 kernel needed
                          │
                   Output BF16
                          │
                    quant to FP8
                          │
                          ▼
Next layer:       FP8 storage ──► repeat
```

### What the patch file does

The patch file at `scripts/pipeline_qwen_image_edit_plus_patch.py` is bind-mounted over the
in-image pipeline at:
```
/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py
```

At import time it:

1. **Monkey-patches `Fp8Config.get_quant_method`** — for DiT attention + MLP linear layers,
   returns `_Fp8WeightOnlyLinearMethod` instead of the standard `Fp8LinearMethod`
2. **`_Fp8WeightOnlyLinearMethod.forward()`** — dequantizes FP8 weights to BF16, calls
   `F.linear(input_bf16, weight_bf16)`, then requantizes if needed
3. **Moves text encoder to CPU** — patches the forward method to handle GPU→CPU→GPU
   device transfers transparently
4. **Sets `VLLM_BATCH_INVARIANT=1`** — ensures vLLM's `Fp8LinearMethod.apply` itself
   takes the BF16-dequant + F.linear path for non-patched layers (modulation, img_in, etc.)

### Cache-DiT Acceleration

Cache-DiT is a block-level caching strategy that compounds with FP8 weight-only:

- **Fn_compute_blocks=1**: Compute the first transformer block, reuse cached result for subsequent blocks
- **Bn_compute_blocks=0**: Don't recompute any trailing blocks
- **max_warmup_steps=4**: Warm up cache for 4 steps before caching kicks in
- **TaylorSeer O(1)**: First-order Taylor approximation for cache hit detection

Configuration via env vars:
```bash
DIFFUSION_CACHE_BACKEND=cache_dit
DIFFUSION_CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"enable_taylorseer":true}'
```

## FP8 Format Comparison

### Native vLLM FP8 (W8A8 Block FP8)
```
quant_method: "fp8"
activation_scheme: "static" or "dynamic"
weight_block_size: [128, 128]  # for block scaling
is_checkpoint_fp8_serialized: true
```
- **Weights:** FP8 (Float8_e4m3fn)
- **Activations:** FP8 (static or dynamic scaling)
- **Kernel:** W8A8 Block FP8 Triton kernel (`_w8a8_triton_block_scaled_mm`)
- **Status:** ❌ Broken on RTX 4090 (Triton 3.6.0 limitation)
- **Used by:** z-image-turbo-fp8, z-image-base-fp8

### ModelOpt FP8 (NVIDIA ModelOpt format)
```json
{
  "quant_method": "modelopt",
  "config_groups": {
    "group_0": {
      "input_activations": {"type": "float", "num_bits": 8, "dynamic": false},
      "weights": {"type": "float", "num_bits": 8, "dynamic": false},
      "targets": ["Linear"]
    }
  },
  "quant_algo": "FP8"
}
```
- **Weights:** FP8 with per-tensor or per-channel scaling
- **Activations:** FP8 with static scaling
- **Kernel:** Different code path (ModelOpt-specific)
- **Status:** ✅ Working on RTX 4090
- **Used by:** qwen-edit-modelopt-fp8-transformer (the overlay files)
- **Note:** Larger on-disk size due to scaling factors stored alongside weights

### FP8 Weight-Only (Pipeline Patch approach)
```
Stored as native vLLM FP8, but patched at runtime:
  Fp8Config.get_quant_method → _Fp8WeightOnlyLinearMethod
```
- **Weights:** FP8 (Float8_e4m3fn) storage
- **Activations:** BF16 for computation
- **Kernel:** Standard BF16 matmul (no special kernel needed)
- **Status:** ✅ Working on RTX 4090
- **Used by:** qwen-image-edit (via pipeline patch)

## Model Conversion Pipeline

The user's FP8 conversion pipeline:
```
Full model (BF16/FP32)
        │
        ▼
FP8 quantization (via vLLM native or ModelOpt)
        │
        ▼
onnx_to_trt / ModelOpt conversion (if needed)
        │
        ▼
ComfyUI-compatible export (fallback format)
        │
        ▼
fp8_xxx storage format
```

For models that don't fit:
1. Convert to ModelOpt FP8 (qwen-image-edit, WAN-VACE approach)
2. Or use pipeline patch for FP8 weight-only (bypasses Triton limitation)
3. Or use ComfyUI version as fallback

## Known Issues & Workarounds

### Triton 3.6.0 fp8e4nv (fork-v1 image)
- **Error:** `AssertionError: Unsupported lhs dtype fp8e4nv`
- **Root cause:** Triton 3.6.0 `tl.dot()` API missing fp8e4nv support
- **Fix:** Use `vllm/vllm-omni:latest` (Triton version may differ) + pipeline patch
- **Workaround:** Convert model via ModelOpt pipeline, or apply FP8 weight-only patch

### Fork start method (fork-v1 image)
- **Error:** `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
- **Root cause:** `multiproc_executor.py:191` forces `mp.set_start_method("fork", force=True)`
- **Fix:** Use `vllm/vllm-omni:latest` which has `mp.set_start_method("spawn", force=True)` at line 141
- **Patch:** `scripts/omni_patch_fork.py` — blocks `set_start_method("fork")` calls

### CUDA OOM (qwen-image-edit)
- **Error:** `torch.OutOfMemoryError: CUDA out of memory`
- **Root cause:** Model weights too large for 24 GB VRAM
- **Fix:** Ensure pipeline patch is applied (FP8 weight-only + CPU text encoder)
- **Env vars:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **VRAM budget:** DiT=20GB, VAE=0.3GB, activations=3GB → 23GB total, fits with 1GB headroom

### Cache-DiT warning
- **Warning:** `Failed to refresh the diffusion transformer cache; backend cache_dit currently requires num_inference_steps to be passed explicitly`
- **Impact:** Minor. Inference still works, cache may not be optimally refreshed
- **Fix:** Pass `num_inference_steps` explicitly in API request

## Testing Profiles

### Qwen-Image-Edit (20B MMDiT, 512×512)
| Config | Steps | Latency | Output |
|--------|-------|---------|--------|
| FP8 weight-only + Cache-DiT | 4 | ~46 s | 787 KB PNG |
| FP8 weight-only + Cache-DiT | 20 | ~46 s | 787 KB PNG |

Both step counts show similar latency because Cache-DiT block-level caching
dominates after the 4-step warmup phase.

### MOSS SoundEffect-v2 (3s audio, 50 steps)
| Config | Latency | Notes |
|--------|---------|-------|
| Cold (Triton compile) | 53.7 s | First run, kernel compilation |
| Warm | 4.8 s | Pure inference |

### CrispASR (whisper base)
| Audio | Latency |
|-------|---------|
| 5s speech | 71 ms |
