# Troubleshooting Guide

## Known Issues & Fixes

### 1. Triton fp8e4nv tl.dot() Crash

**Error:**
```
AssertionError: Unsupported lhs dtype fp8e4nv
  File ".../triton/language/semantic.py", line 1488, in check_dot_layout
```

**Root Cause:** Triton 3.6.0's `tl.dot()` API doesn't expose `fp8e4nv` dtype.
The W8A8 Block FP8 kernel (`_w8a8_triton_block_scaled_mm` in fp8_utils.py)
calls `tl.dot(a, b)` where both operands are fp8, which is unsupported.

**Affected Models:** Models using native vLLM W8A8 Block FP8 quantization:
- z-image-turbo-fp8
- z-image-base-fp8

**Affected Images:** `forge-reg.local:30500/tech-noir/vllm-omni:fork-v1` (Triton 3.6.0)

**Fixes:**

A. **Use pipeline patch (recommended)** — Monkey-patch Fp8Config to be weight-only:
   ```python
   # In pipeline patch file:
   from vllm.model_executor.layers.quantization.fp8 import Fp8Config
   _orig_get_quant_method = Fp8Config.get_quant_method
   def _patched_get_quant_method(self, layer):
       # Return custom weight-only linear method for target layers
       ...
   Fp8Config.get_quant_method = _patched_get_quant_method
   ```
   See `scripts/pipeline_qwen_image_edit_plus_patch.py` for example.

B. **Convert to ModelOpt FP8** — Use user's conversion pipeline to produce
   ModelOpt-format checkpoint (qwen-edit-modelopt-fp8-transformer pattern).

C. **Use latest image** — `vllm/vllm-omni:latest` may have different Triton version
   (not verified if tl.dot supports fp8e4nv there).

D. **Upgrade Triton** — Build container with Triton 3.7+ which may have fp8e4nv support.

---

### 2. CUDA Fork Re-initialization

**Error:**
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

**Root Cause:** `multiproc_executor.py:191` does `mp.set_start_method("fork", force=True)`
after CUDA has already been initialized in the parent process.

**Affected Images:** `forge-reg.local:30500/tech-noir/vllm-omni:fork-v1`

**Fixes:**

A. **Use vllm/vllm-omni:latest** (preferred) — Has `mp.set_start_method("spawn", force=True)`
   at `multiproc_executor.py:141`. This is the correct behavior.

B. **Apply omni_patch_fork.py** — Wrapper script that monkey-patches
   `multiprocessing.set_start_method` to silently ignore "fork" calls:
   ```python
   mp.set_start_method = lambda method, force=False: None if method == "fork" else _orig(method, force)
   ```
   Usage: mount `/patches/omni_patch_fork.py` and use as entrypoint instead of
   directly running api_server.

---

### 3. CUDA Out of Memory (OOM)

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate N MiB.
GPU 0 has a total capacity of 23.52 GiB of which X MiB is free.
```

**Root Cause:** Model weights + activations exceed 24 GB VRAM.

**Fixes:**

A. **Enable expandable segments:**
   ```bash
   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

B. **Enable VAE tiling + slicing** (reduces peak memory):
   ```bash
   -e DIFFUSION_VAE_USE_SLICING=1
   -e DIFFUSION_VAE_USE_TILING=1
   ```

C. **CPU text encoder offload** — Move text encoder to CPU after prefill
   (handled by pipeline patch for qwen).

D. **Reduce batch size / image size** — Smaller images = less activation memory.

E. **Use FP8 weight-only patch** — Without this, the model may load in BF16 (2 bytes/param)
   which doubles memory usage.

**VRAM Budget for Qwen-Image-Edit (20B):**
```
DiT FP8 weights        20 GB
VAE (tiled)            0.3 GB
Activations             3 GB
─────────────────────────────
Total                  23.3 GB  ← barely fits in 24 GB
```

---

### 4. MOSS "No module named 'diffusers'"

**Error:**
```
ModuleNotFoundError: No module named 'diffusers'
```

**Root Cause:** The MOSS container image was built without `diffusers` package,
but the MOSS pipeline code (`moss_soundeffect_v2`) depends on it.

**Fix:**
```bash
docker exec inference-moss pip install diffusers
```

---

### 5. MOSS "Failed to find C compiler"

**Error:**
```
RuntimeError: Failed to find C compiler. Please specify via CC environment
variable or set triton.knobs.build.impl.
```

**Root Cause:** Triton JIT compilation needs a C compiler, but the container
doesn't have `build-essential` installed.

**Fix:**
```bash
docker exec inference-moss apt-get update && apt-get install -y build-essential
```

---

### 6. CrispASR "TTS-only model" (VibeVoice)

**Error:**
```
error: 'model.gguf' is a TTS-only model (no at_enc.*/st_enc.* tensors).
Use --backend vibevoice-tts for this model
```

**Root Cause:** The vibevoice GGUFs at `/mnt/data/models/vibevoice-cpp/` are TTS
models (speech synthesis), not ASR models (speech recognition). They lack the
encoder tensors needed for recognition/diarization.

**Files on disk:**
- `vibevoice-asr-q8_0.gguf` (13 GB) — MISNAMED, is actually TTS
- `vibevoice-asr-q4_k.gguf` (9.7 GB) — MISNAMED, is actually TTS
- `vibevoice-realtime-0.5B-q8_0.gguf` (1.6 GB) — Probably also TTS

**Fixes:**
1. Use `--backend vibevoice-tts` if TTS is desired
2. Download proper ASR model: `crispasr -m auto --backend vibevoice --auto-download`
3. Use whisper backend (works out of box, auto-downloaded on first run)

---

### 7. Cache-DiT Refresh Warning

**Warning:**
```
Failed to refresh the diffusion transformer cache; backend cache_dit currently
requires num_inference_steps to be passed explicitly
```

**Root Cause:** The API request doesn't include `num_inference_steps` in the
payload, so Cache-DiT can't compute the cache refresh schedule.

**Impact:** Minor. Inference still works correctly, but the cache may not be
optimally refreshed between steps.

**Fix:** Pass `num_inference_steps` explicitly in the API request body.

---

### 8. get_hf_file_to_dict Returns None

**Error:**
```
FileNotFoundError or ValueError from enrich_config() — model_index.json not found
```

**Root Cause:** vllm-omni's `enrich_config()` method loads model metadata by
calling `get_hf_file_to_dict("model_index.json", self.model)`. When the model
path is a local directory, this function may fail if it can't find the file
or if the local file resolution is broken.

**Fix:** Patch `get_hf_file_to_dict` to check local paths first:
```python
from vllm.transformers_utils import repo_utils
_orig = repo_utils.get_hf_file_to_dict
repo_utils.get_hf_file_to_dict = lambda name, model, **kw: (
    json.load(open(os.path.join(str(model), name)))
    if os.path.isfile(os.path.join(str(model), name))
    else _orig(name, model, **kw)
)
```

This is included in `scripts/launch_qwen_img_edit_fp8.py` and `scripts/omni_patch_fork.py`.

---

### 9. ACE-Step CUDA 12.8 Build Failure

**Error:**
```
CMake Error at CMakeLists.txt:54 (add_subdirectory):
  The source directory /build/acestep/ggml does not contain a CMakeLists.txt file.
```
or
```
-- Configuring incomplete, errors occurred!
  ...CMakeDetermineCompilerId.cmake:48 (__determine_compiler_id_test)
```

**Root Causes:**

A. **Missing submodules** — `git clone` without `--recurse-submodules`
B. **Old CMake** — Ubuntu 22.04 ships CMake 3.22, which can't detect CUDA 12.8

**Fixes:**

A. Ensure `--recurse-submodules` flag is used
B. Install CMake 3.31+:
   ```dockerfile
   RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.tar.gz \
       && tar xzf cmake-3.31.5-linux-x86_64.tar.gz -C /opt \
       && ln -sf /opt/cmake-3.31.5-linux-x86_64/bin/cmake /usr/local/bin/cmake
   ```

See `infra/docker/Dockerfile.acetep.fixed` for the complete fixed Dockerfile.

---

### 10. Container Port Conflicts

**Symptom:** `docker run` fails with `port is already allocated`

**Common port collisions:**

| Port | Service | Common Issue |
|------|---------|-------------|
| 8050 | MOSS | Leftover `inference-moss` container |
| 8051 | CrispASR | Leftover `inference-diarization` container |
| 8093 | omni-vllm | Qwen or other DiT model still bound |

**Fix:**
```bash
# Check what's using the port
ss -tlnp | grep <PORT>

# Remove the conflicting container
docker rm -f <container-name>
```
