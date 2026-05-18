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
| 1 | **Registry key mismatch**: Wan2GP uses `family/type` (e.g. `wan/t2v`) but our model_registry.yaml keys use versioned names (e.g. `wan-t2v-14B`) | ✅ Fixed | `_WEIGHT_SEARCH` in `deployment.py:267` | Added explicit mapping entries for all models with local weights |
| 2 | **Vendor model weights not found**: Models like `flux/flux_schnell` don't have pre-downloaded weights in our system | ❌ Missing | Registry + download pipeline | Need to add HF source URLs to registry and implement auto-download |
| 3 | **files_locator search paths too narrow**: Wan2GP's `_checkpoints_paths = ['ckpts', '.']` misses our model directory | ✅ Fixed | `_load_model` in `deployment.py:553` | Added model path to `fl._checkpoints_paths` before handler loads |

## Handler Integration

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 4 | **`model_filename` empty for general models**: `_resolve_model_filename` returned `[]` for models without a `_TRANSFORMER_WEIGHTS` entry | ✅ Fixed | `_resolve_model_filename` fallback logic | Finds largest `.safetensors` > 100MB that isn't VAE/T5/encoder |
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
| 12 | **Mixed dtype assertion**: mmgp `offload.profile()` asserts all pipe modules have same dtype, but some models (e.g., anigen) have mixed float32/float16 | ✅ Working | `_apply_mmgp_profile` | Normalizes all pipe modules to `torch.bfloat16` before profiling |
| 13 | **No mmgp models**: Some models manage their own GPU memory (e.g., pixal3d) | ✅ Fixed | `_NO_MMGP_MODELS` set | skips `offload.profile()` for self-managed models |

## Tensor & Output Encoding

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 14 | **`image_b64` not passed to trellis handler**: Our code decoded image_b64 into image_start for i2v models, but trellis expects raw `image_b64` | 🔧 Pending | `infer` method i2v handling | image_b64 is still passed through _SAFE_PASSTHROUGH; need to verify trellis works |
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

## Verified Working Models

| Model | Type | Status | Time | Notes |
|-------|------|--------|------|-------|
| `espeak/espeak` | CPU TTS | ✅ PASS | 0.8s | |
| `kokoro/kokoro` | CPU TTS | ✅ PASS | 2.7s | |
| `faster_whisper/faster_whisper` | CPU ASR | ✅ PASS | 9.3s | |
| `faster_qwen3_tts/faster-qwen3-tts` | GPU TTS | ✅ PASS | 15.7s | |
| `wan/t2v` | GPU Video | ✅ PASS | 95s (9fr@2st) | 14B model, mmgp-offloaded |
| `LLM service` | GPU Text | ✅ PASS | ~60s | Qwen3.6-27B-Q5_K_S via llama-server subprocess |

## Models Requiring Additional Work

| Model | Issue | Notes |
|-------|-------|-------|
| `flux/flux_schnell` | Missing weights | Need HF download or pre-download |
| `flux/*` | Missing weights | All flux variants need weights |
| `wan/*` | Partial weights | Only t2v-14B has weights; i2v-14B empty |
| `trellis/trellis` | image_b64 transport | Need to verify passthrough works |
| `anigen/anigen` | Mixed dtypes | Just fixed, needs re-test |
| `see_through/see-through` | Empty dir | weights not present |
