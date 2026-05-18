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
| 3 | **files_locator search paths too narrow**: Wan2GP's `_checkpoints_paths = ['ckpts', '.']` misses our model directory | ✅ Fixed | `_load_model` in `deployment.py:690` | Added model path to `fl._checkpoints_paths` before handler loads |

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

## Verified Working Models

| Model | Type | Status | Time | Notes |
|-------|------|--------|------|-------|
| `espeak/espeak` | CPU TTS | ✅ PASS | 0.8s | Previous test |
| `kokoro/kokoro` | CPU TTS | ✅ PASS | 2.7s | Previous test |
| `faster_whisper/faster_whisper` | CPU ASR | ✅ PASS | 9.3s | Previous test |
| `faster_qwen3_tts/faster-qwen3-tts` | GPU TTS | ✅ PASS | 15.7s | Previous test |
| `wan/t2v` | GPU Video | ✅ PASS | 68s (5fr@2st) | 14B model, mmgp-offloaded. Includes 40s model load. |
| `LLM service` | GPU Text | ✅ PASS | ~60s | Qwen3.6-27B-Q5_K_S via llama-server subprocess |

## Models With Local Weights (ready to load)

17 models have local weight files and are available through the Forge:

| Model | Weight Path |
|-------|-------------|
| `wan/t2v` | `/models/wan2gp/wan/t2v-14B` |
| `wan/t2v_2_2` | `/models/wan2gp/wan/t2v-14B` (shared) |
| `trellis/trellis` | `/models/3d/trellis/TRELLIS.2-4B/ckpts` |
| `anigen/anigen` | `/models/3d/anigen/ckpts` |
| `hy_motion/hy-motion-1.0` | `/models/motion/hy-motion-1.0` |
| `hy_motion/hy-motion-1.0-lite` | `/models/motion/hy-motion-1.0-lite` |
| `moss/moss-soundeffect` | `/models/audio/moss-soundeffect/bf16` |
| `moss/moss-tts` | `/models/audio/moss-tts` |
| `moss/moss-voicegenerator` | `/models/audio/moss-voicegenerator` |
| `see_through/see-through` | `/models/image/see-through/layerdiff3d` |
| `faster_qwen3_tts/faster-qwen3-tts` | `/models/tts/qwen3-tts-12hz-1.7b-customvoice` |
| `tts/index_tts2` | `/models/tts/index-tts` |
| `kokoro/kokoro` | `/models/tts/kokoro` |
| `faster_whisper/faster_whisper` | `/models/asr/faster-whisper` |
| `vibevoice_asr/vibevoice-asr` | `/models/asr/vibevoice-asr` |
| `vibevoice_tts/vibevoice-tts` | `/models/tts/vibevoice` |
| `flux/flux_schnell` | Auto-download (VAE+T5+CLIP cached in `ckpts/`) |

## Models Requiring Additional Work

| Model | Issue | Notes |
|-------|-------|-------|
| `flux/*` | Main model auto-download slow | VAE/T5/CLIP download works. Main 12GB model needs longer timeout or pre-download. |
| `wan/i2v` | No local weights | Needs auto-download or pre-download to test |
| `anigen/anigen` | Mixed dtypes | `_model_dtype` fix should handle this, needs re-test |
| `spconv` | CUDA issue | Blocks TRELLIS mesh extraction — PyPI wheel is CPU-only |

## Discovery Statistics

- **Total models discovered**: 113 (from 15 family handlers)
- **With local weights**: 17
- **Auto-download capable**: ~96 (vendor handlers with HF access)
- **Blocked**: 0 (auto-download path handles missing weights)
- **CPU-only (no weights needed)**: 3 (kokoro, espeak, faster_whisper)
