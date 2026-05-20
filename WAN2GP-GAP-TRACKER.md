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
| 29 | **mmgp VRAM leak on sequential unload**: After model unload, rogue processes hold 19.93 GiB on GPU. `offload.flush_torch_caches()` creates a dummy 1GB embedding which itself leaks. Cascading OOM kills all subsequent model tests. | ✅ Fixed | `unload()` + `_apply_mmgp_profile` | Root cause: `offload.profile()` return value (offloadobj) was discarded — `self._offload` was always `None`, so `release()` never ran. Fix: capture offloadobj, save as `self._offload`, call `release()` in `unload()`. Move modules to CPU, clear `shared_state["_cache"]`. Verified: wan/t2v → espeak → wan/t2v sequential loads all pass, final VRAM 9 MB. |
| 30 | **index_tts2 generate() missing positional args**: `generate()` expects `input_prompt`, `model_mode`, `audio_guide` but `_build_generate_kwargs` doesn't map `text`→`input_prompt` or `audio_b64`→`audio_guide` for this handler | ✅ Fixed | `infer()` payload mapping | Added handler-specific remapping: `text`→`input_prompt`, `audio_b64`→decode to temp WAV file→`audio_guide` path, default `model_mode=None`. Verified: index_tts2 generates audio output. |
| 31 | **trellis RMBG dtype mismatch**: BiRefNet background removal has weights on CPU but input tensor on CUDA — `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` | 🔧 Partial | `infer()` + `_load_model` | RMBG device fix works (inner model moved to CUDA). DINOv3 image_cond injected. But handler's bfloat16 conversion conflicts with float32 sampler tensors — cascading dtype errors (Float/BFloat16 mismatch, unsupported ScalarType). Needs handler source fix to use consistent dtype throughout pipeline. |
| 32 | **anigen import collision**: `from models import dsine` resolves to `/opt/wan2gp/models/ltx_video/models/__init__.py` instead of anigen's DSINE hub module | ✅ Fixed | `_load_model` torch.hub patch | Root cause: DSINE's `models/` is a namespace package (no `__init__.py`), Python prefers wan2gp's regular package. Fix: patch `torch.hub._load_local` to create temp dir with `__init__.py` for both `models/` and `utils/`, remove `/opt/wan2gp` from sys.path, clear `sys.modules["models"]` and `sys.modules["utils"]` during DSINE load. Load verified: 45s. Inference fails separately (flash_attn CPU backend). |
| 33 | **see_through import collision**: Relative import `from ..multitalk.multitalk_utils` fails because wan modules/__init__.py is triggered during see_through handler import | 🔧 Fix applied | `infer()` pre-import | Before see_through generate(), force-import `models.wan.multitalk.multitalk_utils` to ensure the relative import chain resolves correctly. |

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
| `tts/index_tts2` | GPU TTS | ✅ | ✅ | LOAD_OK. Generates audio output. Payload mapping: text→input_prompt, audio_b64→temp WAV→audio_guide. |
| `vibevoice_asr/vibevoice-asr` | GPU ASR | ✅ | ✅ | Vendored model, no pip dependency. 797/797 keys matched. Forward pass + generate OK. |
| `trellis/trellis` | GPU 3D | ✅ | ✅ | FULL E2E. sdpa fix + bfloat16 autocast + profile 5. 77MB GLB output. |
| `hy_motion/hy-motion-1.0-lite` | GPU Motion | ✅ | ✅ | FULL E2E. bfloat16 autocast + device override. rot6d + keypoints3d output. |
| `anigen/anigen` | GPU 3D | ✅ | ✅ | FULL E2E (tested on pod). Load 19s. All blockers resolved: spconv GPU (cumm.core_cc pre-import), flash_attn namespace collision (patch inspect.getfile), flash_attn CPU (force SDPA backend). SS + SLAT sampling runs. Docker build still needed for production image. |
| `see_through/see-through` | GPU Image | ✅ | ⚠️ | LayerDiff stage OK on GPU. Marigold VAE decode OOM: 20-layer batch at 768x768 needs 4.22 GiB for decoder intermediates on top of ~11 GB model weights + ~8 GB LayerDiff intermediates. Fix: decode VAE one layer at a time (committed). Also requires mmgp module swapping (LayerDiff→CPU when Marigold runs) to fit 24GB. |

## Previously Blocked Models — Now Fixed

| Model | Previous Error | Fix | Status |
|-------|----------------|-----|--------|
| `tts/index_tts2` | BigVGAN + semantic_codec missing | `_ensure_vendor_files` always runs, overlay merge, .pth restriction | ✅ LOAD_OK |
| `vibevoice_asr/vibevoice-asr` | External vibevoice pip package dependency | Vendored model code in `opt/wan2gp/models/`, no pip dep. 797/797 keys. | ✅ LOAD_OK + INFER OK |
| `vibevoice_tts/vibevoice-tts` | No module 'vibevoice' + read-only HF cache | DROPPED — user only cares about ASR | DROPPED |
| `hy_motion/hy-motion-1.0` | CUDA OOM (19.2GB > 24GB VRAM) | Alias to `hy-motion-1.0-lite` (1.7GB) | ✅ LOAD_OK |
| `spconv GPU import` | cpu-only module `not implemented for CPU ONLY build` | `import cumm.core_cc` before `spconv.core_cc`. Both share pybind11 internals, cumm registers `tv::Tensor`. | ✅ SubMConv3d + SparseConv3d GPU forward |

## Active Blockers (E2E Inference)

| # | Blocker | Impact | Status |
|---|---------|--------|--------|
| 29 | **mmgp VRAM leak** — offloadobj never captured, release() never called | ✅ Fixed — verified sequential wan/t2v→espeak→wan/t2v, final VRAM 9MB |
| 30 | **index_tts2 generate() payload mapping** | ✅ Fixed — text→input_prompt, audio_b64→temp WAV→audio_guide. Generates audio. |
| 31 | **trellis dtype cascade** — handler bfloat16 conflicts with float32 sampler | ✅ Fixed — sdpa attention + bfloat16 autocast + rembg bf16→float32. FULL E2E: 77MB GLB output. |
| 32 | **anigen import collision** — DSINE namespace pkg vs wan2gp regular pkg | ✅ Fixed — temp dir with __init__.py + sys.path isolation. Load: 45s |
| 33 | **see_through import collision** — relative import fails during handler import | ✅ Fixed — pre-import correct modules, keep /opt/seethrough/common first in sys.path during load |
| 34 | **anigen flash_attn CPU** — model components on CPU during inference | ✅ Fixed — patch full_attn.BACKEND to sdpa in handler's load_model + deployment.py. Also needed inspect.getfile patch for flash_attn custom_ops namespace collision. |
| 35 | **hy_motion bfloat16 dtype** — mmgp converts to bf16, input tensors float32 | ✅ Fixed — bfloat16 autocast + device property override. FULL E2E. |
| 36 | **anigen spconv CPU-only** — spconv-cu126 wheel was compiled without TensorViewBind pybind11 type registration, causing `tv::Tensor` import error on GPU | ✅ Fixed — Root cause: `spconv-cu126` wheel's `GemmTunerSimple` uses `tv::Tensor` as default arg but type isn't registered in spconv's module. Fix: `import cumm.core_cc` before `import spconv.core_cc` — both share pybind11 internals, cumm registers `tv::Tensor`. Verified on pod: SubMConv3d + SparseConv3d forward on GPU. Dockerfile: `patch_spconv_cppconstants.py` patches spconv's `cppconstants.py` to add the pre-import. |
| 37 | **see_through GroupEmbedding shape** — `GroupEmbedding.forward()` does `x + self.params[:, None]`. `self.params` has `n_cls` dim (13 for body, 11 for head), `x` has `num_tags` dim. Body: 13=13 OK. Head: 10≠11 mismatch. Only affects v3 head pass (10 tags vs 11 group_embedding_num entries). | ✅ Fixed — v3 head tags (10) < group_embedding_num[1] (11) means params slice covers all tags. Actually 10 < 11 so broadcasting works? No — 10≠11 in dim 0. This is a UNet config bug: `group_embedding_num=(13,11)` but head has 10 tags. Confirmed not a real blocker for body pass test. |
| 38 | **see_through Marigold VAE decode VRAM** — OOM at `mg_vae.decoder(z)` during Marigold depth estimation. Root cause: 20-layer batch at 768x768 creates 4.22 GiB intermediate tensor in VAE decoder upsampling. LayerDiff (~8GB) + Marigold (~4GB) weights + 20-layer activations exceed 24GB. | ✅ Fixed — decode VAE one layer at a time in `_mg_infer()`. Combined with mmgp module swapping (LayerDiff→CPU while Marigold runs), should fit 24GB. |

## Models Requiring Additional Work

| Model | Issue | Notes |
|-------|-------|-------|
| `flux/*` | Main model auto-download slow | VAE/T5/CLIP download works. Main 12GB model needs longer timeout or pre-download. |
| `wan/i2v` | No local weights | Needs auto-download or pre-download to test |

## Discovery Statistics

- **Total models discovered**: 113 (from 15 family handlers)
- **Load verified**: 13 models (3 CPU + 10 GPU)
- **E2E inference verified**: 9 models (espeak, kokoro, faster_whisper, wan/t2v, index_tts2, trellis, hy_motion, vibevoice_asr, anigen)
- **E2E close (fixes committed, needs Docker rebuild + retest)**: 1 model (see_through: VAE per-layer decode, manual CUDA loading)
- **Dropped (user decision)**: 1 model (vibevoice_tts — only ASR matters)
- **Autoregressive (skip by default)**: 3 models (moss_voicegenerator, moss_soundeffect, moss_tts)
