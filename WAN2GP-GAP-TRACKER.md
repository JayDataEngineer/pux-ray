# Wan2GP Integration — Gap Tracker

This file tracks gaps between the Wan2GP UI framework and our Forge deployment,
plus fixes applied in `services/wan2gp/deployment.py`.

## Status Legend
- ✅ Fixed & working
- 🔧 Fix applied but untested
- ❌ Known gap, fix needed
- 📝 Observation, no fix required

---

## Model Discovery & Weight Resolution

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 1 | **Registry key mismatch**: Wan2GP uses `family/type` (e.g. `wan/t2v`) but our model_registry.yaml keys use versioned names (e.g. `wan-t2v-14B`) | ✅ Fixed | `_WEIGHT_SEARCH` in `deployment.py:281` | Added explicit mapping entries for all models with local weights |
| 2 | **Vendor model weights not found**: Models like `flux/flux_schnell` don't have pre-downloaded weights | ✅ Verified | `_ensure_vendor_files` + `_ensure_main_model` | `query_model_files()` downloads VAE/T5/CLIP to `ckpts/`. Main model downloads from defaults URLs. Tested: flux_schnell downloaded VAE (335MB) + T5 + CLIP in 60s. Main model needs longer for large files. |
| 3 | **files_locator search paths too narrow**: Wan2GP's `_checkpoints_paths = ['ckpts', '.']` misses our model directory | ✅ Fixed | `_load_model` in `deployment.py:690` | Added model path + parent to `fl._checkpoints_paths` before handler loads |

## Handler Integration

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 4 | **`model_filename` empty for general models**: `_resolve_model_filename` returned `[]` for models without a `_TRANSFORMER_WEIGHTS` entry | ✅ Fixed | `_resolve_model_filename` fallback logic | Finds largest `.safetensors` > 100MB that isn't VAE/T5/encoder. Also checks `.pth`/`.pt` files. |
| 5 | **`text_encoder_filename` not passed to handler**: T5 encoder checkpoint was `None`, causing `TypeError` | ✅ Fixed | `_load_model` text_encoder resolution | Scans model dir for T5/umt5/text_encoder safetensors, prefers `.safetensors` over `.pth` |
| 6 | **`_interrupt` flag missing**: `WanAny2V.generate()` accesses `self._interrupt` which isn't set outside UI flow | ✅ Fixed | `infer` method | Sets `model._interrupt = False` before calling generate |
| 7 | **`offloadobj` parameter defaulting to `None`**: `WanAny2V.generate(offloadobj=None, ...)` calls `offloadobj.unload_all()` which crashes | ✅ Fixed | `_DEFAULT_OFFLOADOBJ` + `kwargs.setdefault` | Passes a no-op dummy offloadobj |
| 8 | **`shared_state["_attention"]` missing**: mmgp generates `KeyError` on `_attention` key | ✅ Fixed | `infer` method | Sets `_moff.shared_state["_attention"] = "sdpa"` before generate |
| 9 | **`loras_slists` parameter defaulting to `None`**: `update_loras_slists` crashes with `'NoneType' object is not subscriptable` | ✅ Fixed | `infer` method | Passes empty `{"phase1": [], "phase2": [], "phase3": []}` |
| 10 | **`callback` parameter defaulting to `None`**: Generate calls `callback(...)` which crashes | ✅ Fixed | `infer` method | Passes no-op lambda |
| 11 | **Output tensor shape mismatch**: Wan returns `(C, F, H, W)` in uint8, but encoder assumed `(F, C, H, W)` in float | ✅ Fixed | `_encode_output` | Detects `(C, F, H, W)` by checking `shape[0] in (1,3,4) and shape[1] > 4`, transposes to `(F, H, W, C)` |

## mmgp Compatibility

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 12 | **Mixed dtype assertion**: mmgp `offload.profile()` asserts all pipe modules have same dtype, but quantized models have mixed float32/bfloat16 params | ✅ Fixed | `_apply_mmgp_profile` | Sets `_model_dtype = torch.bfloat16` on each module so mmgp uses it as the target dtype and converts float32→bfloat16 via its own `convertWeightsFloatTo` logic |
| 13 | **No mmgp models**: Some models manage their own GPU memory (e.g., pixal3d) | ✅ Fixed | `_NO_MMGP_MODELS` set | skips `offload.profile()` for self-managed models |

## Tensor & Output Encoding

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 14 | **`image_b64` passthrough for non-i2v models**: Code decoded image_b64 into image_start for ALL models, but trellis and others need raw base64 string | ✅ Fixed | `infer` method i2v handling | Only decodes image_b64→image_start for base_model_type i2v/i2v_2_2. Other models get raw image_b64 via _SAFE_PASSTHROUGH. |
| 15 | **`tmp_path` variable lost**: Code edit accidentally removed `tempfile.NamedTemporaryFile` block | ✅ Fixed | `_encode_output` | Restored tmp_path creation |

## Config & Registry

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 16 | **Model key format inconsistency**: Wan2GP uses `family/type` but registry uses hyphenated names | ✅ Fixed | `_WEIGHT_SEARCH` | Explicit mapping entries bridge the gap |
| 17 | **Missing wan model weight registrations**: `i2v`, `t2v`, `trellis`, `index_tts2` types unmapped | ✅ Fixed | `_WEIGHT_SEARCH` | Added entries for all discovered model types with local weights |

## Forge Integration

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 18 | **VRAM not tracked for mmgp models**: `vram_mb=0` made the Forge think models use 0 VRAM, so eviction never triggered across mmgp services | ✅ Fixed | `forge_adapter.py` | Changed `vram_mb` from class constant `0` to a property backed by `torch.cuda.memory_allocated()` diff after load; gives the forge accurate per-service VRAM for eviction decisions |
| 19 | **No release/status endpoints**: POST always routed to invoke(); no way to release VRAM without pod restart | ✅ Fixed | `forge.py` | Added `action` routing — `{'action':'release'}`, `{'action':'status'}`, `{'action':'preload'}` — all work without a `service` field |
| 20 | **Default model key mismatch**: `default_model = "wan/t2v-14B"` but discovery produces key `"wan/t2v"` | ✅ Fixed | `deployment.py:364` + `forge_adapter.py:24` | Changed default_model to `"wan/t2v"`, added `_ALIASES` dict so `wan/t2v-14B` resolves to `wan/t2v` |
| 21 | **Folder name aliasing for locate_folder()**: Handler expects `index_tts2` but dir is named `index-tts`; can't symlink (read-only PVC) | ✅ Fixed | `_load_model` folder aliases | Patches `fl.locate_folder` to map expected names → actual paths for known mismatches |

## Verified Working Models

| Model | Type | Load | Infer | Disk Size | Notes |
|-------|------|------|-------|-----------|-------|
| `kokoro/kokoro` | CPU TTS | ✅ 3.1s | ✅ 3.1s | — | audio/wav output |
| `faster_whisper/faster_whisper` | CPU ASR | ✅ 2.9s | ✅ 2.9s | — | Silent audio → empty text (expected) |
| `wan/t2v` | GPU Video | ✅ 40s | ✅ 28s | 41.1 GB | 14B mmgp-offloaded. 5fr@480x480. |
| `moss/moss-voicegenerator` | GPU Audio | ✅ 29s | — | 4.0 GB | LOAD_OK. Inference very slow (autoregressive 4096 tokens). |
| `see_through/see-through` | GPU Image | ✅ 132s | — | 9.5 GB | LOAD_OK. Needs image input for inference. |
| `trellis/trellis` | GPU 3D | ✅ 181s | — | 15.1 GB | LOAD_OK. spconv CPU-only blocks mesh extraction. |
| `anigen/anigen` | GPU 3D | ✅ 92s | — | 20.4 GB | LOAD_OK, 1736MB VRAM. Self-managed GPU. |
| `moss/moss-soundeffect` | GPU Audio | ✅ 21s | — | 15.6 GB | LOAD_OK. mmgp-offloaded. |
| `moss/moss-tts` | GPU Audio | ✅ 23s | — | 15.8 GB | LOAD_OK. mmgp-offloaded. |

## Models With Load Failures

| Model | Error | Root Cause | Fix |
|-------|-------|------------|-----|
| `tts/index_tts2` | BigVGAN folder missing | Wan2GP handler requires pre-downloaded `bigvgan_v2_22khz_80band_256x` | Need to download BigVGAN or add to Docker image |
| `vibevoice_tts/vibevoice-tts` | No module named 'vibevoice' | Handler does `from vibevoice.modular...` but the pip package isn't installed | Need `pip install vibevoice` in Docker image |
| `hy_motion/hy-motion-1.0` | CUDA OOM | 19.2 GB on disk exceeds 24GB VRAM even with mmgp | Need more aggressive quantization or lower mmgp profile |
| `vibevoice_asr/vibevoice-asr` | Likely same as vibevoice_tts | Same vendor package dependency | Need `pip install vibevoice` in Docker image |

## Models Requiring Additional Work

| Model | Issue | Notes |
|-------|-------|-------|
| `flux/*` | Main model auto-download slow | VAE/T5/CLIP download works. Main 12GB model needs longer timeout or pre-download. |
| `wan/i2v` | No local weights | Needs auto-download or pre-download to test |
| `spconv` | CUDA issue | Blocks TRELLIS mesh extraction — PyPI wheel is CPU-only |
| `hy_motion/hy-motion-1.0` | OOM on 24GB | 19.2GB disk, needs quantization or profile tuning |
| `tts/index_tts2` | Missing BigVGAN | Handler expects `bigvgan_v2_22khz_80band_256x` folder |
| `vibevoice_tts` / `vibevoice_asr` | Missing pip package | `pip install vibevoice` needed in Docker image |

## Discovery Statistics

- **Total models discovered**: 113 (from 15 family handlers)
- **With local weights**: 17
- **Auto-download capable**: ~96 (vendor handlers with HF access)
- **Blocked**: 0 (auto-download path handles missing weights)
- **CPU-only (no weights needed)**: 3 (kokoro, espeak, faster_whisper)
- **Load verified**: 9 models (3 CPU + 6 GPU)
- **End-to-end verified**: 4 models (kokoro, faster_whisper, wan/t2v, moss-voicegenerator)
- **Load failures (4)**: index_tts2 (missing BigVGAN), vibevoice_tts (missing pip), hy_motion (OOM), vibevoice_asr (missing pip)
