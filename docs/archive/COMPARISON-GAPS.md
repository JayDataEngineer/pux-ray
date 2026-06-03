# Wan2GP Integration Gap Analysis

Comparison between upstream Wan2GP (`vendor/wan2gp/`) and our integration
(`services/wan2gp/deployment.py` + `services/model_engine/handlers/wan2gp/__init__.py`).

---

## 1. Model Support

### Upstream Model Families (14 families, 100+ model types)

#### WAN (`models/wan/wan_handler.py`, line 53-56)
`family_handler.query_supported_types()` returns 38 model types:

| Model Type | Category | Our Coverage |
|---|---|---|
| `t2v` | WAN 2.1 Text-to-Video 14B | **Registered** as `wan/t2v-14B` |
| `t2v_2_2` | WAN 2.2 Text-to-Video 14B | MISSING |
| `t2v_1.3B` | WAN 2.1 Text-to-Video 1.3B | MISSING |
| `i2v` | WAN 2.1 Image-to-Video 14B | **Registered** as `wan/i2v-14B` |
| `i2v_2_2` | WAN 2.2 Image-to-Video 14B | MISSING |
| `i2v_2_2_multitalk` | WAN 2.2 Multi-speaker Talking Head | MISSING |
| `i2v_2_2_svi2pro` | SVI2Pro variant | MISSING |
| `flf2v_720p` | First-Last Frame to Video | MISSING |
| `fun_inp_1.3B` | Fun InPaint 1.3B | MISSING |
| `fun_inp` | Fun InPaint 14B | MISSING |
| `multitalk` | Multi-speaker Talking Head | MISSING |
| `infinitetalk` | Infinite Talk | MISSING |
| `fantasy` | Fantasy Talking | MISSING |
| `animate` | Animate (talking head, 30fps) | MISSING |
| `vace_14B` | VACE ControlNet 14B | MISSING |
| `vace_14B_2_2` | VACE ControlNet 14B v2.2 | MISSING |
| `vace_1.3B` | VACE ControlNet 1.3B | MISSING |
| `vace_multitalk_14B` | VACE MultiTalk 14B | MISSING |
| `vace_standin_14B` | VACE Stand-In 14B | MISSING |
| `vace_lynx_14B` | VACE Lynx 14B | MISSING |
| `vace_ditto_14B` | VACE Ditto 14B | MISSING |
| `standin` | Stand-In | MISSING |
| `lynx` | Lynx (T2V) | MISSING |
| `lynx_lite` | Lynx Lite | MISSING |
| `phantom_1.3B` | Phantom 1.3B | MISSING |
| `phantom_14B` | Phantom 14B | MISSING |
| `recam_1.3B` | ReCam 1.3B | MISSING |
| `alpha` | Alpha (T2V) | MISSING |
| `alpha2` | Alpha 2 (T2V) | MISSING |
| `alpha_lynx` | Alpha Lynx | MISSING |
| `chrono_edit` | Chronological Edit | MISSING |
| `ti2v_2_2` | Text+Image-to-Video 5B | MISSING |
| `lucy_edit` | Lucy Edit 5B | MISSING |
| `kiwi_edit` | Kiwi Edit 5B | MISSING |
| `mocha` | Mocha | MISSING |
| `steadydancer` | Steady Dancer (pose-driven) | MISSING |
| `wanmove` | WAN Move (trajectory-driven) | MISSING |
| `scail` | SCAIL (multi-person pose) | MISSING |
| `vista4d` | Vista4D (camera control) | MISSING |

#### Hunyuan (`models/hyvideo/hunyuan_handler.py`, line 142)
9 model types:

| Model Type | Our Coverage |
|---|---|
| `hunyuan` | **Registered** as `hunyuan/t2v` |
| `hunyuan_i2v` | MISSING |
| `hunyuan_custom` | MISSING |
| `hunyuan_custom_audio` | MISSING |
| `hunyuan_custom_edit` | MISSING |
| `hunyuan_avatar` | MISSING |
| `hunyuan_1_5_t2v` | MISSING |
| `hunyuan_1_5_i2v` | MISSING |
| `hunyuan_1_5_upsampler` | MISSING |

#### Flux (`models/flux/flux_handler.py`, lines 28-42)
12 model types:

| Model Type | Our Coverage |
|---|---|
| `flux` | **Registered** as `flux/t2i` |
| `flux2_dev` | MISSING |
| `pi_flux2` | MISSING |
| `flux2_klein_4b` | MISSING |
| `flux2_klein_9b` | MISSING |
| `flux_chroma` | MISSING |
| `flux_chroma_radiance` | MISSING |
| `flux_dev_kontext` | MISSING |
| `flux_dev_umo` | MISSING |
| `flux_dev_uso` | MISSING |
| `flux_schnell` | MISSING |
| `flux_dev_kontext_dreamomni2` | MISSING |

#### LTX-Video (`models/ltx_video/ltxv_handler.py`, line 47)
| Model Type | Our Coverage |
|---|---|
| `ltxv_13B` | MISSING |

#### LTX-2 (`models/ltx2/ltx2_handler.py`, line 183)
| Model Type | Our Coverage |
|---|---|
| `ltx2_19B` | MISSING |
| `ltx2_22B` | MISSING |

#### Kandinsky 5 (`models/kandinsky5/kandinsky_handler.py`, lines 81-86)
| Model Type | Our Coverage |
|---|---|
| `k5_lite_t2v` | MISSING |
| `k5_lite_i2v` | MISSING |
| `k5_pro_t2v` | MISSING |
| `k5_pro_i2v` | MISSING |

#### LongCat (`models/longcat/longcat_handler.py`, lines 8-9)
| Model Type | Our Coverage |
|---|---|
| `longcat_video` | MISSING |
| `longcat_avatar` | MISSING |

#### MagiHuman (`models/magi_human/magi_human_handler.py`, line 32)
| Model Type | Our Coverage |
|---|---|
| `magi_human` | MISSING |
| `magi_human_distill` | MISSING |

#### Qwen Image (`models/qwen/qwen_handler.py`, line 101)
| Model Type | Our Coverage |
|---|---|
| `qwen_image_20B` | MISSING |
| `qwen_image_edit_20B` | MISSING |
| `qwen_image_edit_plus_20B` | MISSING |
| `qwen_image_edit_plus2_20B` | MISSING |
| `qwen_image_layered_20B` | MISSING |

#### Z-Image (`models/z_image/z_image_handler.py`, lines 61-62)
| Model Type | Our Coverage |
|---|---|
| `z_image` | MISSING |
| `z_image_base` | MISSING |
| `z_image_control` | MISSING |
| `z_image_control2` | MISSING |
| `z_image_control2_1` | MISSING |

#### TTS Models (`models/TTS/`)
| Model Type | Handler | Our Coverage |
|---|---|---|
| `ace_step_v1` | `ace_step_handler.py:277` | MISSING (only v1_5 registered) |
| `ace_step_v1_5` | `ace_step_handler.py:277` | **Registered** as `ace_step/v1_5` (via vendor) |
| `ace_step_v1_5_xl` | `ace_step_handler.py:277` | MISSING |
| `index_tts2` | `index_tts2_handler.py:180` | **Registered** but BLOCKED (transformers compat) |
| `chatterbox` | `chatterbox_handler.py` | MISSING |
| `yue` | `yue_handler.py:125` | MISSING |
| `kugelaudio` | `kugelaudio_handler.py` | MISSING |
| `heartmula_oss_3b` | `heartmula_handler.py` | MISSING |
| `qwen3_tts_customvoice` | `qwen3_handler.py:14` | MISSING (we have custom handler via model_engine) |
| `qwen3_tts_voicedesign` | `qwen3_handler.py:16` | MISSING |
| `qwen3_tts_base` | `qwen3_handler.py:18` | MISSING |

### Summary: Model Coverage

- **Upstream total**: ~100 model types across 14 families
- **Our coverage**: 4 vendor models (wan/t2v, wan/i2v, hunyuan/t2v, flux/t2i) + 2 vendor TTS (ace_step v1_5, blocked index_tts2) = **6 of ~100**
- **Missing entire families**: LTX-Video, LTX-2, Kandinsky 5, LongCat, MagiHuman, Qwen Image, Z-Image
- **Missing TTS models**: Chatterbox (multilingual), YuE (music), KugelAudio, HeartMula, Qwen3 TTS (vendor version)

---

## 2. VRAM Management

### Upstream (`wgp.py` lines 2122-2162)

Upstream has a rich configuration system for mmgp profiles:

```python
server_config = {
    "profile": profile_type.LowRAM_LowVRAM,
    "video_profile": profile_type.LowRAM_LowVRAM,
    "image_profile": profile_type.LowRAM_LowVRAM,
    "audio_profile": 3.5,
    "attention_mode": "auto",
    "transformer_quantization": "int8",
    "text_encoder_quantization": "int8",
    "lm_decoder_engine": "",
    "compile": "",
    "vram_safety_coefficient": 0.8,
    "perc_reserved_mem_max": 0.0,
}
```

### Our Implementation

We hard-code profile 2 (balanced) and fixed budgets:

```python
# deployment.py line 288-298
budgets = {"transformer": 250, "text_encoder": 250, "*": 3000}
offload.profile(
    pipe,
    profile_no=MMGP_PROFILES["balanced"],  # always 2
    quantizeTransformer=False,
    budgets=budgets,
    loras=[],
    perc_reserved_mem_max=0.5,
    vram_safety_coefficient=0.9,
    coTenantsMap={},
)
```

### Gaps

| Feature | Upstream | Ours | Gap |
|---|---|---|---|
| **Per-modality profiles** | Separate video/image/audio profiles | Single profile for all | No tuning for image vs video VRAM usage |
| **Profile selection** | CLI `--profile` flag + runtime switchable | Hard-coded `balanced=2` | Cannot adjust per-request |
| **Transformers quantization** | `int8`, `fp8`, `quanto_int8` via `transformer_quantization` config | Always `int8` hard-coded | No fp8 support |
| **Text encoder quantization** | Configurable (`int8`, `fp16`, `bf16`) | Always `int8` hard-coded | No option for higher quality |
| **Attention modes** | `auto`, `sdpa`, `flash`, `sage`, `radial` via `attention_mode` | Default only | Missing sage/radial attention for long sequences |
| **Compilation** | `torch.compile` via `compile` config + per-model `compile` list | Not exposed | Missing inference speedups |
| **VAE dtype** | Configurable per model | Hard-coded `torch.float32` | No half-precision VAE option |
| **Mixed precision transformer** | `mixed_precision_transformer` flag | Not exposed | Missing option |
| **Save quantized** | `--save-quantized` CLI flag, saves int8/fp8 model to disk | Not exposed | Cannot pre-quantize models |
| **lm_decoder_engine** | `legacy`, `cg` (CUDA Graphs), `vllm` for LLM-backed TTS | Not passed through | Missing TTS speedup options |
| **Preload model** | `--preload MB` flag, partial VRAM preloading | Not exposed | No warm-start capability |
| **vram_safety_coefficient** | Configurable 0.0-1.0 | Hard-coded 0.9 | Cannot tune for different GPU sizes |
| **perc_reserved_mem_max** | CLI `--perc-reserved-mem-max` | Hard-coded 0.5 | Cannot adjust RAM allocation |

---

## 3. Generation Features

### 3.1 WAN Video (wan_handler.py)

**Upstream generate parameters** (from `wgp.py` `generate_video()` lines 5771-5881):

| Parameter | Description | Our Coverage |
|---|---|---|
| `prompt` | Text prompt | **Supported** |
| `negative_prompt` | Negative text prompt | NOT PASSED |
| `alt_prompt` | Alternate prompt (for multi-phase guidance) | NOT PASSED |
| `image_start` | Start image (i2v) | **Supported** (via `image_b64`) |
| `image_end` | End image | NOT PASSED |
| `image_refs` | Multiple reference images | NOT PASSED |
| `video_source` | Input video (for v2v) | NOT PASSED |
| `video_guide` | Guide video | NOT PASSED |
| `video_mask` | Mask video | NOT PASSED |
| `image_mask` | Mask image | NOT PASSED |
| `control_net_weight` | ControlNet strength | NOT PASSED |
| `motion_amplitude` | Motion amplitude (i2v) | NOT PASSED |
| `num_inference_steps` | Denoising steps | **Supported** (as `steps`) |
| `guidance_scale` | CFG scale | **Supported** (as `guidance`) |
| `guidance2_scale` | Secondary CFG scale (multi-phase) | NOT PASSED |
| `guidance3_scale` | Tertiary CFG scale | NOT PASSED |
| `switch_threshold` | Phase switching threshold | NOT PASSED |
| `alt_guidance_scale` | Alternate guidance | NOT PASSED |
| `flow_shift` | Flow shift parameter | NOT PASSED |
| `sample_solver` | Solver type: `unipc`, `euler`, `dpm++`, `causvid`, `lcm` | NOT PASSED (always default) |
| `seed` | Random seed | **Supported** |
| `resolution` | Width x Height | **Supported** (but cannot be overridden per-request in deployment.py; uses defaults only) |
| `sliding_window_size` | Window size for long videos | NOT PASSED |
| `sliding_window_overlap` | Window overlap | NOT PASSED |
| `audio_guide` | Audio guide (for talking head models) | NOT PASSED |
| `audio_guide2` | Second audio guide | NOT PASSED |
| `audio_source` | Audio source | NOT PASSED |
| `activated_loras` | Active LoRAs | NOT PASSED |
| `loras_multipliers` | LoRA strength multipliers | NOT PASSED |
| `skip_steps_cache_type` | TeaCache/MagCache selection | NOT PASSED |
| `skip_steps_multiplier` | Cache aggressiveness | NOT PASSED |
| `repeat_generation` | Batch generation count | NOT PASSED |
| `batch_size` | Parallel generations | NOT PASSED |
| `perturbation_switch` | Perturbation mode | NOT PASSED |
| `cfg_star_switch` | CFG* acceleration | NOT PASSED |
| `cfg_zero_step` | CFG-Zero* | NOT PASSED |
| `NAG_scale` | Negative Attention Guidance | NOT PASSED |
| `temperature` | Sampling temperature | NOT PASSED |
| `top_p` | Nucleus sampling | NOT PASSED |
| `embedded_guidance_scale` | Embedded guidance | NOT PASSED |
| `denoising_strength` | V2V denoising strength | NOT PASSED |
| `masking_strength` | Mask strength | NOT PASSED |
| `temporal_upsampling` | Temporal upscaling | NOT PASSED |
| `spatial_upsampling` | Spatial upscaling | NOT PASSED |
| `film_grain_intensity` | Film grain effect | NOT PASSED |
| `MMAudio_setting` | Auto audio generation | NOT PASSED |
| `RIFLEx_setting` | RIFLEx optimization | NOT PASSED |
| `self_refiner_setting` | Self-refinement passes | NOT PASSED |
| `image_prompt_type` | Image prompt mode selection | NOT PASSED |
| `model_mode` | Model-specific mode selection | NOT PASSED |
| `fps` | Output FPS | **Supported** (via `fps` payload key) |
| `custom_settings` | Model-specific settings dict | NOT PASSED |
| `prompt_enhancer` | Auto prompt enhancement | NOT PASSED |

**Critical bug in deployment.py**: Width/height/frames are read from `defaults` dict instead of `payload`:

```python
# deployment.py lines 436-438 — reads from defaults, not payload!
width = int(defaults.get("width", 1280))
height = int(defaults.get("height", 720))
frames = int(defaults.get("frames", 81))
```

This means resolution and frame count are always fixed to the default values and cannot be overridden per-request. The model_engine `__init__.py` handler (line 270-271) correctly reads from payload.

### 3.2 Flux Image (flux_handler.py)

Upstream supports features our integration ignores entirely:

| Feature | Upstream | Our Coverage |
|---|---|---|
| **Inpainting** | Full inpainting support with multiple methods (LanPaint, Masked Denoising) | NOT PASSED |
| **Outpainting** | `video_guide_outpainting` with ratio control | NOT PASSED |
| **ControlNets** | Per-model ControlNet support | NOT PASSED |
| **Image references** | Multiple reference image modes (KI, I, KIJ) | NOT PASSED |
| **LoRA** | Full LoRA support with multipliers | NOT PASSED |
| **NAG** | Negative Attention Guidance (NAG scale, tau, alpha) | NOT PASSED |
| **Flux2 variants** | Flux2 Dev, Klein 4B/9B, PI-Flux2 | NOT EXPOSED |
| **Chroma/Radiance** | Flux Chroma + Chroma Radiance modes | NOT EXPOSED |
| **Kontext** | Flux Dev Kontext (image-guided generation) | NOT EXPOSED |
| **Schnell** | Flux Schnell (fast inference mode) | NOT EXPOSED |

### 3.3 Hunyuan Video (hunyuan_handler.py)

| Feature | Upstream | Our Coverage |
|---|---|---|
| **I2V** | `hunyuan_i2v` variant | NOT EXPOSED |
| **Custom models** | `hunyuan_custom`, `hunyuan_custom_edit`, `hunyuan_custom_audio` | NOT EXPOSED |
| **Avatar** | `hunyuan_avatar` (talking head) | NOT EXPOSED |
| **1.5 variants** | `hunyuan_1_5_t2v`, `hunyuan_1_5_i2v`, `hunyuan_1_5_upsampler` | NOT EXPOSED |
| **MagCache** | Per-resolution MagCache ratios | NOT PASSED |
| **TeaCache** | Adaptive step skipping | NOT PASSED |
| **Embedded guidance** | Model-specific embedded guidance | NOT PASSED |
| **Sliding window** | Long video generation with overlap control | NOT PASSED |

### 3.4 ACE-Step (ace_step_handler.py)

**Our model_engine handler** (`services/model_engine/handlers/ace_step/`) is actually MORE feature-rich than the vendor path for v1.5. Our orchestrator supports:
- CoT metadata inference (5 model modes)
- Audio code generation with CFG
- Reference audio encoding (cover/timbre)
- Cover-strength blending
- ODE + SDE denoising
- VAE temporal tiling for long audio
- Custom settings (bpm, keyscale, timesignature, language)

**Upstream features we're missing** in our vendor handler path (`V2V_MODELS["ace_step/v1_5"]`):

| Feature | Upstream | Our Vendor Path |
|---|---|---|
| **Audio prompt types** | `""`, `A` (cover), `B` (timbre), `AB` (cover+timbre) | NOT PASSED |
| **LM engine selection** | `vllm`, `cg`, `legacy` | NOT PASSED (hard-coded) |
| **LM CoT modes** | 5 model modes (0-4) | NOT PASSED (in vendor path) |
| **Transformer variants** | base, sft, turbo, turbo_shift1/3, turbo_continuous, xl_turbo | NOT EXPOSED (hard-coded) |
| **LoRA** | `enabled_audio_lora: True`, per-directory | NOT PASSED |
| **v1.5 XL** | `ace_step_v1_5_xl` variant | NOT EXPOSED |
| **v1** | `ace_step_v1` with different pipeline | NOT EXPOSED |
| **Duration** | Up to 360 seconds (v1.5) | NOT EXPOSED in vendor path |

### 3.5 IndexTTS2 (index_tts2_handler.py)

Our registration is **blocked** (`blocked: True`) with reason: "Vendored transformers_generation_utils.py incompatible with transformers>=4.55".

**Upstream features** (if unblocked):

| Feature | Description |
|---|---|
| Voice cloning | Single reference audio |
| Voice + emotion | Two reference audios (AB mode) |
| Dialogue | Two speaker reference audios (AB2 mode) |
| Auto-split | Split long text into sentences with pauses |
| Temperature/top_p/top_k | Sampling parameters |
| Early stopping | `supports_early_stop: True` |
| Default emotion | Pre-set emotion instruction |
| LM engines | `legacy`, `cg`, `vllm` |
| Duration slider | 1-600 seconds |

---

## 4. Quantization

### Upstream Quantization Options

From `wgp.py` lines 2122-2126 and `shared/cli_args.py`:

| Option | Values | Description |
|---|---|---|
| `transformer_quantization` | `int8`, `fp8` | Main transformer quantization |
| `text_encoder_quantization` | `int8`, `fp16`, `bf16` | Text encoder quantization |
| `--save-quantized` | flag | Save quantized model to disk for faster next load |
| `--fp16` / `--bf16` | flags | Override transformer dtype policy |
| `mixed_precision_transformer` | bool | Per-layer mixed precision |
| `submodel_no` | int | Multi-submodel quantization (for dual-transformer models) |
| NVFP4 format | Per-model default configs | NVFP4 quantized models (ltx2, flux, wan) |
| GGUF quantization | `q4_k_m`, `q6_k`, `q8_0` | GGUF-quantized models (ltx2_distilled, ltx2_22B) |
| Nunchaku quantization | `r128_fp4`, `r256_int4` | Nunchaku quantized models (z_image, qwen_image) |

### Our Quantization

```python
# deployment.py line 279
quantizeTransformer=True,
text_encoder_quantization="int8",
```

Hard-coded. No fp8, no NVFP4, no GGUF, no Nunchaku, no mixed precision, no save-quantized.

---

## 5. Audio Models Deep Comparison

### 5.1 ACE-Step v1.5

| Parameter | Upstream (`ace_step_handler.py` lines 324-371) | Our Vendor Path | Our Model Engine Handler |
|---|---|---|---|
| `input_prompt` (lyrics) | Yes | NOT PASSED | **Supported** |
| `alt_prompt` (caption) | Yes | NOT PASSED | **Supported** |
| `duration_seconds` | 5-360 | NOT PASSED | **Supported** |
| `num_inference_steps` | 8 default | NOT PASSED | **Supported** |
| `temperature` | 0.85 default | NOT PASSED | **Supported** |
| `top_p` | 0.9 default | NOT PASSED | **Supported** |
| `top_k` | 0 default | NOT PASSED | **Supported** |
| `guidance_scale` | 1.0 (LM) | NOT PASSED | **Supported** |
| `alt_guidance_scale` | 2.5 (CFG) | NOT PASSED | **Supported** |
| `bpm` | 30-300 | NOT PASSED | **Supported** (via custom_settings) |
| `keyscale` | C major format | NOT PASSED | **Supported** (via custom_settings) |
| `timesignature` | 2,3,4,6 | NOT PASSED | **Supported** (via custom_settings) |
| `language` | ISO 639-1 code (50+ languages) | NOT PASSED | **Supported** (via custom_settings) |
| `audio_prompt_type` | `""`, `A`, `B`, `AB` | NOT PASSED | NOT SUPPORTED |
| `audio_scale` | Source audio strength | NOT PASSED | NOT SUPPORTED |
| `repeat_generation` | Batch count | NOT PASSED | NOT SUPPORTED |
| `model_modes` | 5 CoT preprocessing modes | NOT PASSED | **Supported** (as `model_mode`) |
| `transformer_variant` | base/sft/turbo/xl_turbo/etc. | NOT PASSED | NOT SUPPORTED |
| `lm_decoder_engine` | `legacy`, `cg`, `vllm` | NOT PASSED | NOT SUPPORTED |
| LoRA | Per-directory audio LoRAs | NOT PASSED | NOT SUPPORTED |

### 5.2 IndexTTS2

Blocked in our integration. Full upstream parameter set:

| Parameter | Upstream |
|---|---|
| `prompt` | Text to speak |
| `alt_prompt` | Default emotion instruction |
| `audio_prompt_type` | `A` (voice clone), `AB` (voice+emotion), `AB2` (dialogue) |
| `audio_guide` | Reference voice audio |
| `audio_guide2` | Second reference (emotion/speaker 2) |
| `temperature` | Sampling temperature |
| `top_p` | Nucleus sampling |
| `top_k` | Top-K sampling |
| `duration_seconds` | Max duration 1-600s |
| `auto_split_every_s` | Auto sentence split (5-90s) |
| `lm_decoder_engine` | `legacy`, `cg`, `vllm` |
| `pause_between_sentences` | Inter-sentence pauses |

### 5.3 Chatterbox (MISSING entirely)

Multilingual TTS with per-language models, exaggeration/pace controls.

### 5.4 Qwen3 TTS (vendor vs our model_engine)

Our model_engine has `faster_qwen3_tts` handler (custom implementation with CUDA graphs).
Upstream vendor has `qwen3_handler.py` with 3 variants: `customvoice`, `voicedesign`, `base`.
Our custom handler is likely faster but may miss upstream bug fixes and the `voicedesign` variant.

---

## 6. Missing Features (CLI, Env, Config)

### CLI Flags (`shared/cli_args.py`)

| Flag | Description | Our Equivalent |
|---|---|---|
| `--save-quantized` | Save quantized model to disk | NOT EXPOSED |
| `--profile` | mmgp profile number | Hard-coded to 2 |
| `--attention` | Attention mode (auto/flash/sdpa/sage) | NOT EXPOSED |
| `--compile` | torch.compile mode | NOT EXPOSED |
| `--preload MB` | Preload N MB into VRAM | NOT EXPOSED |
| `--vram-safety-coefficient` | VRAM safety margin | Hard-coded 0.9 |
| `--perc-reserved-mem-max` | RAM reservation | Hard-coded 0.5 |
| `--save-masks` | Save preprocessing masks | NOT EXPOSED |
| `--share` | Public URL (Gradio) | N/A (API only) |
| `--config` | Config directory | NOT EXPOSED |
| `--steps` | Default denoising steps | Per-request |
| `--frames` | Default frame count | Per-request |
| `--seed` | Default seed | Per-request |
| `--lora-preset` | Pre-load LoRA preset | NOT EXPOSED |
| `--process` | Batch process queue file | NOT EXPOSED |
| `--output-dir` | Output directory override | NOT EXPOSED |
| `--multiple-images` | Allow multiple input images | NOT EXPOSED |
| `--check-loras` | Validate LoRA files | NOT EXPOSED |
| `--gpu` | GPU device selection | Hard-coded |

### Environment Variables

| Variable | Description | Our Equivalent |
|---|---|---|
| `WAN2GP_ROOT` | Root directory for Wan2GP | Set in `_ensure_vendor()` |
| `NUMBA_THREADING_LAYER` | Threading for Linux | NOT SET |

### Post-processing Pipeline (upstream `postprocessing/`)

| Module | Description | Our Coverage |
|---|---|---|
| `mmaudio/` | Automatic audio generation for videos | NOT EXPOSED |
| `rife/` | Frame interpolation (RIFE v4) | NOT EXPOSED |
| `flashvsr/` | Video super-resolution | NOT EXPOSED |
| `film_grain.py` | Film grain effect | NOT EXPOSED |

### Pre-processing Pipeline (upstream `preprocessing/`)

| Module | Description | Our Coverage |
|---|---|---|
| `canny.py` | Canny edge detection | NOT EXPOSED |
| `depth_anything_v2/` | Depth estimation v2 | NOT EXPOSED |
| `depth_anything_v3/` | Depth estimation v3 | NOT EXPOSED |
| `dwpose/` | Pose estimation | NOT EXPOSED |
| `face_preprocessor.py` | Face detection/alignment | NOT EXPOSED |
| `flow.py` | Optical flow | NOT EXPOSED |
| `midas/` | MiDaS depth | NOT EXPOSED |
| `raft/` | RAFT optical flow | NOT EXPOSED |
| `sam3/` | SAM3 segmentation | NOT EXPOSED |
| `scribble.py` | Scribble detection | NOT EXPOSED |
| `speakers_separator.py` | Speaker diarization | NOT EXPOSED |
| `matanyone/` | Video matting | NOT EXPOSED |
| `arc/` | ARC face recognition | NOT EXPOSED |
| `extract_vocals.py` | Vocal extraction | NOT EXPOSED |

### Plugins System (upstream `plugins/`)

Upstream has 10 built-in plugins:

| Plugin | Description |
|---|---|
| `wan2gp-about` | About page |
| `wan2gp-configuration` | Settings management |
| `wan2gp-downloads` | Model download manager |
| `wan2gp-guides` | Usage guides |
| `wan2gp-models-manager` | Model management |
| `wan2gp-motion-designer` | Motion design tool |
| `wan2gp-plugin-manager` | Plugin management |
| `wan2gp-process-full-video` | Full video processing |
| `wan2gp-sample` | Quick sample generation |
| `wan2gp-video-mask-creator` | Mask creation tool |

All are Gradio UI plugins, not relevant to our API-only integration.

---

## 7. High-Impact Gaps (Prioritized)

### Priority 1 — Features that directly improve output quality

1. **Resolution/frames bug** (`deployment.py` lines 436-438): Width/height/frames read from defaults, not payload. Users cannot change resolution.
2. **Negative prompts**: Not passed to any model. Major quality lever.
3. **LoRA support**: Not wired. Users cannot use finetunes.
4. **Sample solver selection**: Always defaults. `dpm++` and `causvid` can be significantly faster.
5. **TeaCache/MagCache**: Not wired. 30-50% speedup with minimal quality loss.

### Priority 2 — Missing model variants with high user value

1. **WAN 2.2** (`t2v_2_2`, `i2v_2_2`): Newer, better quality
2. **WAN 1.3B** (`t2v_1.3B`): Fits in much less VRAM, faster generation
3. **Hunyuan 1.5** (`hunyuan_1_5_t2v/i2v/upsampler`): Major quality upgrade
4. **Flux2** variants: Newer image generation models
5. **LTX-2** (`ltx2_19B`, `ltx2_22B`): Video+audio generation
6. **Chatterbox**: Multilingual TTS with emotion control

### Priority 3 — Features that improve inference speed

1. **torch.compile**: Can provide 20-50% speedup after warmup
2. **Attention modes** (sage, radial): Better memory efficiency for long sequences
3. **lm_decoder_engine** (`cg`/`vllm`): Faster LLM inference for TTS models
4. **Save quantized**: Pre-quantize models for faster loading
5. **Per-modality profiles**: Fine-tune VRAM usage per model type

### Priority 4 — Features for advanced workflows

1. **Inpainting/outpainting** (Flux, Hunyuan): Image editing
2. **Video-to-video** (WAN, Hunyuan): Style transfer, editing
3. **Talking heads** (multitalk, infinitetalk, fantasy): Audio-driven animation
4. **Pose-driven animation** (steadydancer, scail, wanmove): Dance/motion
5. **Camera control** (vista4d): 4D camera movements
6. **Kandinsky 5**: Alternative video generation family
7. **LongCat**: Long video generation with continuation
8. **Qwen Image**: Alternative image generation family
9. **Z-Image**: Alternative image generation with control nets
10. **MagiHuman**: Human animation with audio

---

## 8. Dependency Gaps

### Upstream deps not in our Docker image

From `vendor/wan2gp/requirements.txt`:

| Package | Purpose | Likely in our image? |
|---|---|---|
| `gradio==5.29.0` | Web UI (not needed for API) | No (not needed) |
| `gradio_rangeslider` | UI component | No (not needed) |
| `pygame>=2.1.0` | Audio playback | No |
| `sounddevice>=0.4.0` | Audio I/O | No |
| `openai-whisper==20250625` | Whisper ASR (preprocessing) | No |
| `audio-separator==0.36.1` | Source separation | No |
| `pyannote.audio==3.3.2` | Speaker diarization | No |
| `speechbrain==1.0.3` | Speech processing | No |
| `torchcodec` | Video decoding | Maybe |
| `segment-anything` | SAM segmentation | No |
| `rembg[gpu]==2.0.65` | Background removal | No |
| `onnxruntime-gpu` | ONNX inference | No |
| `insightface==0.7.3` | Face detection | No |
| `taichi` | GPU compute | No |
| `vector_quantize_pytorch==1.27.19` | VQ for audio models | Maybe |
| `gguf==0.17.1` | GGUF model loading | No |
| `flash-linear-attention==0.4.1` | FLA attention | No |
| `conformer==0.3.2` | Conformer models | No |
| `spacy` + `spacy_pkuseg` | Chinese NLP | No |
| `omegaconf` + `hydra-core` | Config management | Maybe |
| `dashscope` | API client | No |
| `s3tokenizer` | Tokenizer | No |
| `misaki` | Japanese tokenizer | No |
| `wetext==0.1.2` | Text processing | No |
| `smplfitter` | SMPL body fitting (MagiHuman) | No |
| `chumpy` | SMPL dependency | No |
| `decord` | Video decoding | Maybe |
| `peft==0.17.0` | LoRA loading | Maybe |
| `nvidia-ml-py` | NVML monitoring | No |
| `stringzilla` | String processing | No |
| `mutagen` | Audio metadata | Maybe |

Most missing deps are for preprocessing, plugins, or model families we don't expose. The critical ones for extending support would be `peft` (LoRA), `gguf` (quantized models), and model-family-specific packages.
