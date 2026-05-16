# Integration List — Custom Models in Wan2GP Fork

Models integrated (or being integrated) into the Wan2GP fork, tracked against the
[WAN2GP-RULE.md](WAN2GP-RULE.md) decomposition test (7 criteria).

**Wan2GP-native models** (wan, flux, hyvideo, ltx2, qwen, ace_step, etc.) are excluded —
they ship with Wan2GP and are maintained upstream.

## Status Legend

| Mark | Meaning |
|------|---------|
| PASS | All 7 criteria met |
| PARTIAL | Meets some criteria, documented gaps |
| FAIL | Does not meet the decomposition standard |
| BLOCKED | Cannot integrate without fundamental changes |
| TODO | Not started |

## Custom Models

### moss — MOSS-SoundEffect 8B (text-to-sound)

**Status: PARTIAL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{language_model, audio_tokenizer, emb_ext, lm_heads}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PARTIAL — `MossAudioEmbedding` + `MossLMHead` authored in handler. `audio_tokenizer` loaded via `AutoModel.from_pretrained(trust_remote_code=True)` which executes upstream code from HuggingFace |
| 5 | Relative imports from source subdir | N/A — authored classes are in the handler file itself, no separate source subdir |
| 6 | Import chain terminates at pip package | PASS — `transformers.Qwen2Model`, `safetensors`, `scipy` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/moss/moss_handler.py`, `config/model_specs.yaml`
**Quant variants:** Per-module quant variant registry in `config/model_specs.yaml`.
Weights in `bf16/` subdirectory (int8 prepared for Unsloth Studio export).
Resolved via `registry/specs.py` — shared by Wan2GP handlers and the Forge.
**Notes:** Closest to the ideal pattern. The `trust_remote_code=True` on audio_tokenizer
is a grey area — it loads a model class from the HF repo at runtime, similar to how
`transformers` handles any model. The authored parts (embedding, LM heads) are clean.
First model with full quant variant hot-swap support.

---

### kokoro — Kokoro 82M TTS

**Status: FAIL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{bert, bert_encoder, predictor, text_encoder, decoder}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | FAIL — `kokoro_model.py` docstring says "All nn.Module definitions come from vendor's multitalk kokoro module" |
| 5 | Relative imports from source subdir | FAIL — uses `from models.kokoro.kokoro_model` (absolute), source files at root of model dir not in `kokoro/` subdir |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers`, `numpy` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/kokoro/kokoro_model.py`, `kokoro_phonemizer.py`
**Blockers:**
- Source is vendored from Wan2GP's multitalk module (admitted in docstring)
- Must move source into `kokoro/kokoro/` subdir and rewrite to use relative imports
- nn.Module architectures need verification that they're authored, not copied

---

### trellis — TRELLIS.2 4B (image-to-3D)

**Status: FAIL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 9 modules: `{ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024, tex_slat_flow_1024, shape_decoder, tex_decoder, image_cond, rembg}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | FAIL — `trellis2/` contains 74 Python files, 9728 lines — this is the upstream TRELLIS codebase |
| 5 | Relative imports from source subdir | FAIL — uses `sys.path.insert()` hack to make `from trellis2.*` work |
| 6 | Import chain terminates at pip package | PARTIAL — depends on `spconv`, `trimesh`, `nvdiffrast` (available in Docker image but not standard pip) |
| 7 | Zero monkeypatches | FAIL — `sys.path` manipulation is a shim |

**Files:** `opt/wan2gp/models/trellis/trellis2/` (74 files), `trellis_handler.py`
**Blockers:**
- 9728 lines of upstream TRELLIS source code in `trellis2/`
- Complex 3D pipeline with sparse convolutions, mesh processing, texture baking
- Unlike Moss (Qwen2Model + thin heads), TRELLIS has no pip package to wrap
- Would need the entire pipeline authored or accepted as a "package" dependency
- `sys.path` hack violates rule 7

---

### vibevoice_tts — VibeVoice 7B TTS (community)

**Status: FAIL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 7 modules: `{language_model, acoustic_tokenizer, semantic_tokenizer, prediction_head, acoustic_connector, semantic_connector, lm_head}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | FAIL — `blocks.py` and `diffusion.py` are structural copies of upstream `modular_vibevoice_tokenizer.py` and `modular_vibevoice_diffusion_head.py` (same class names, same config parsing, same architecture) |
| 5 | Relative imports from source subdir | PASS — `from .vibevoice_tts.blocks import ...` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers`, `diffusers` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/vibevoice_tts/vibevoice_tts/blocks.py`, `diffusion.py`
**Critical bugs:**
- Weight key nesting mismatch: upstream has triple-nested conv wrappers (`SConv1d → NormConv1d → nn.Conv1d`), authored code has double-nested (`SConv1d → nn.Conv1d`). `_load_and_strip(strict=False)` silently drops all conv weights
- Diffusion head architecture is wrong: upstream uses SwiGLU with adaLN-modulated layers, authored code uses plain `Linear → SiLU → Linear`. The 26 `model.prediction_head.*` keys won't map
- TTS `generate()` returns silence (line 202: `torch.zeros(24000)`) — LM autoregressive output is discarded
- Config typo `acostic_vae_dim` matches TTS checkpoint but masks a real issue

---

### vibevoice_asr — VibeVoice ASR 7B (microsoft)

**Status: FAIL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 4 modules: `{language_model, acoustic_tokenizer, acoustic_connector, lm_head}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | FAIL — same structural copy issue as TTS; `blocks.py` is identical to TTS version |
| 5 | Relative imports from source subdir | PASS — `from .vibevoice_asr.blocks import ...` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/vibevoice_asr/vibevoice_asr/blocks.py`
**Critical bugs:**
- Same weight key nesting mismatch as TTS (NormConv1d wrapper missing)
- Config key typo: handler reads `acostic_vae_dim` but ASR checkpoint uses correct `acoustic_vae_dim`. Falls back to default (64) by accident
- ASR `generate()` attempts real inference but untested against actual weights

---

## Not Yet Integrated

### anigen — AniGen (image-to-rigged-3D)

**Status: BLOCKED**

Not decomposable into a thin wrapper. AniGen's inference pipeline depends on:
- Custom ODE flow-matching samplers
- Sparse 3D convolution (spconv)
- Mesh processing, UV parametrization, texture baking
- Skeleton rigging, skinning weight transfer
- GLB binary export

All interdependent — no pip package provides these. The upstream source is ~5000+ lines
across `models/`, `modules/`, `representations/`, `utils/`, `pipelines/`.

**Options:**
1. Accept upstream source as a "package" dependency (like trellis2/ pattern)
2. Author all 5000+ lines from scratch
3. Skip AniGen entirely

---

### see_through — See-Through (anime layer decomposition)

**Status: BLOCKED**

Same category as AniGen — custom diffusion pipeline with no pip package equivalent.

---

### hy_motion — HY-Motion 1.0 (text-to-3D motion)

**Status: BLOCKED**

Same category as AniGen — custom motion generation pipeline, no pip package equivalent.

---

## Summary

| Model | Status | Pipe Modules | Source Authored | Weights Load | Inference Works |
|-------|--------|-------------|-----------------|-------------|----------------|
| moss | PARTIAL | 4 | Mostly | Yes | Yes |
| kokoro | FAIL | 5 | No (vendored) | Yes | Yes |
| trellis | FAIL | 9 | No (9728 LOC upstream) | Yes | Yes |
| vibevoice_tts | FAIL | 7 | No (structural copy) | No (nesting mismatch) | No (returns silence) |
| vibevoice_asr | FAIL | 4 | No (structural copy) | No (nesting mismatch) | Untested |
| anigen | BLOCKED | — | — | — | — |
| see_through | BLOCKED | — | — | — | — |
| hy_motion | BLOCKED | — | — | — | — |

**Passing: 0/8** (moss is closest but `trust_remote_code=True` on audio_tokenizer is a grey area)
