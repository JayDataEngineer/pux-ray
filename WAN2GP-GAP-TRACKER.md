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

## Supporting File Downloads

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 22 | **_ensure_vendor_files only ran when model_path is None**: Handler-required supporting files (semantic_codec, BigVGAN, etc.) not downloaded for models with existing local weights | ✅ Fixed | `_load_model` | `_ensure_vendor_files` now always runs, regardless of whether model_path was resolved |
| 23 | **Download path nesting**: `hf_hub_download` with `local_dir=ckpts/folder/` created nested dirs (`ckpts/bigvgan/bigvgan/file`) | ✅ Fixed | `_ensure_vendor_files` | Changed to always use `local_dir=ckpts_base` — HF creates correct flat structure |
| 24 | **Downloaded files not visible to overlay**: Files downloaded to `ckpts/index_tts2/` not available in `/tmp/wan2gp_overlay/index_tts2/` | ✅ Fixed | `_load_model` overlay merge | After overlay creation, copies downloaded files from `ckpts/base_model_type/` into the overlay |
| 25 | **index_tts2 .pth fallback**: Handler uses safetensors loader but `.pth` fallback tried to give it `gpt.pth` | ✅ Fixed | `_resolve_model_filename` | Restricted `.pth`/`.pt` fallback to `_PTH_SAFE_TYPES = {wan, hunyuan, flux, ace_step}` |

## Environment & Infrastructure

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 26 | **HF_HUB_CACHE points to read-only PVC**: `ray-service.yaml` worker set `HF_HUB_CACHE=/models/hf_cache/hub` (read-only), blocking HF model downloads at load time | ✅ Fixed | `ray-service.yaml` + `_ensure_writable_hf_cache` | Changed worker env to `/tmp/huggingface/hub`. Added safety net in `_load_model` that redirects read-only cache paths to `/tmp/hf_cache`. |
| 27 | **VibeVoice pip package missing**: Handler does `from vibevoice.modular...` but the pip package wasn't in Docker image | ✅ Fixed | `Dockerfile.gpu-all` | Added `git clone + uv pip install -e .` for Microsoft/VibeVoice |
| 28 | **hy-motion OOM on 24GB**: Full model (19.2GB) exceeds RTX 4090 VRAM | ✅ Fixed | `_ALIASES` | `hy_motion/hy-motion-1.0` aliased to `hy-motion-1.0-lite` (1.7GB, loads in 52s) |

## Sequential VRAM Management

| # | Gap | Status | Component | Notes |
|---|-----|--------|-----------|-------|
| 29 | **mmgp VRAM leak on sequential unload**: After model unload, rogue processes hold 19.93 GiB on GPU. `offload.flush_torch_caches()` creates a dummy 1GB embedding which itself leaks. Cascading OOM kills all subsequent model tests. | 🔧 Fix applied | `unload()` + `_apply_mmgp_profile` | Root cause: `offload.profile()` return value (offloadobj) was discarded — `self._offload` was always `None`, so `release()` never ran. mmgp held references to all model tensors. Fix: capture offloadobj from `_apply_mmgp_profile()`, save as `self._offload`, call `release()` in `unload()`. Also move modules to CPU before deleting, clear `shared_state["_cache"]`, removed `flush_torch_caches()` (it creates a dummy 1GB embedding). |
| 30 | **index_tts2 generate() missing positional args**: `generate()` expects `input_prompt`, `model_mode`, `audio_guide` but `_build_generate_kwargs` doesn't map `text`→`input_prompt` or `audio_b64`→`audio_guide` for this handler | ❌ Active | `infer()` payload mapping | Need handler-specific key mapping or direct passthrough |
| 31 | **trellis RMBG dtype mismatch**: BiRefNet background removal has weights on CPU but input tensor on CUDA — `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` | ❌ Active | `trellis_handler._preprocess_image` | RMBG model not moved to GPU by mmgp. Needs `.to(device)` before inference. |
| 32 | **anigen import collision**: `from models import dsine` resolves to `/opt/wan2gp/models/ltx_video/models/__init__.py` instead of anigen's DSINE hub module | ❌ Active | `anigen_handler` + PYTHONPATH | Module name collision from broad PYTHONPATH. Needs sys.path isolation. |
| 33 | **see_through import collision**: Relative import `from ..multitalk.multitalk_utils` fails because wan modules/__init__.py is triggered during see_through handler import | ❌ Active | `see_through_handler` + PYTHONPATH | Same root cause as #32 — broad PYTHONPATH causes cross-family imports. |

## Verified Working Models

| Model | Type | Load | Infer | Notes |
|-------|------|------|-------|-------|
| `espeak/espeak` | CPU TTS | ✅ | ✅ | WAV output, RIFF header verified |
| `kokoro/kokoro` | CPU TTS | ✅ | ✅ | WAV output, 7.6s inference |
| `faster_whisper/faster_whisper` | CPU ASR | ✅ | ✅ | Segments output, fixture audio |
| `wan/t2v` | GPU Video | ✅ | ✅ | video/mp4 output, 3fr@256x256 |
| `moss/moss-voicegenerator` | GPU Audio | ✅ | — | LOAD_OK. Autoregressive (170min). |
| `moss/moss-soundeffect` | GPU Audio | ✅ | — | LOAD_OK. Autoregressive (170min). |
| `moss/moss-tts` | GPU Audio | ✅ | — | LOAD_OK. Autoregressive. |
| `hy_motion/hy-motion-1.0-lite` | GPU Motion | ✅ | — | LOAD_OK in isolation. OOM in sequential tests (VRAM leak #29). |
| `tts/index_tts2` | GPU TTS | ✅ | ❌ | LOAD_OK. Infer fails: missing positional args (#30). |
| `vibevoice_tts/vibevoice-tts` | GPU TTS | ✅* | — | LOAD_OK in isolation. Fails in current image (no vibevoice module). |
| `vibevoice_asr/vibevoice-asr` | GPU ASR | ✅* | — | Same as vibevoice_tts. |
| `trellis/trellis` | GPU 3D | ✅ | ❌ | LOAD_OK. Infer fails: RMBG dtype mismatch (#31). |
| `anigen/anigen` | GPU 3D | ✅ | ❌ | LOAD_OK in isolation. Infer fails: import collision (#32). |
| `see_through/see-through` | GPU Image | ✅ | ❌ | LOAD_OK in isolation. Infer fails: import collision (#33). |

*✅ = verified in manual kubectl exec testing; fails in current Docker image (needs rebuild for vibevoice)

## Previously Blocked Models — Now Fixed

| Model | Previous Error | Fix | Status |
|-------|----------------|-----|--------|
| `tts/index_tts2` | BigVGAN + semantic_codec missing | `_ensure_vendor_files` always runs, overlay merge, .pth restriction | ✅ LOAD_OK |
| `vibevoice_tts/vibevoice-tts` | No module 'vibevoice' + read-only HF cache | Dockerfile pip install + HF_HUB_CACHE redirect | ✅ LOAD_OK |
| `vibevoice_asr/vibevoice-asr` | Same as vibevoice_tts | Same fix | ✅ LOAD_OK |
| `hy_motion/hy-motion-1.0` | CUDA OOM (19.2GB > 24GB VRAM) | Alias to `hy-motion-1.0-lite` (1.7GB) | ✅ LOAD_OK |

## Active Blockers (E2E Inference)

| # | Blocker | Impact | Priority |
|---|---------|--------|----------|
| 29 | **mmgp VRAM leak** — offloadobj never captured, release() never called | 🔧 Fix applied — needs testing on cluster | ~~Critical~~ → Medium |
| 30 | **index_tts2 generate() payload mapping** | Infer fails for index_tts2 | Medium |
| 31 | **trellis RMBG dtype mismatch** | Infer fails for trellis | Medium |
| 32 | **anigen import collision** | Load fails after other models imported | Low |
| 33 | **see_through import collision** | Load fails after other models imported | Low |

## Models Requiring Additional Work

| Model | Issue | Notes |
|-------|-------|-------|
| `flux/*` | Main model auto-download slow | VAE/T5/CLIP download works. Main 12GB model needs longer timeout or pre-download. |
| `wan/i2v` | No local weights | Needs auto-download or pre-download to test |

## Discovery Statistics

- **Total models discovered**: 113 (from 15 family handlers)
- **Load verified**: 13 models (3 CPU + 10 GPU)
- **E2E inference verified**: 4 models (espeak, kokoro, faster_whisper, wan/t2v)
- **E2E blocked by VRAM leak (#29)**: 4 models (hy_motion, trellis, anigen, see_through)
- **E2E blocked by payload mapping (#30)**: 1 model (index_tts2)
- **Needs Docker rebuild**: 2 models (vibevoice_tts, vibevoice_asr)
- **Autoregressive (skip by default)**: 3 models (moss_voicegenerator, moss_soundeffect, moss_tts)
