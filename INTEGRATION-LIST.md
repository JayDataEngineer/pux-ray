# Integration List — Custom Models in Wan2GP Fork

Models integrated (or being integrated) into the Wan2GP fork, tracked against the
[WAN2GP-RULE.md](WAN2GP-RULE.md) decomposition test (7 criteria + Amendments A & B).

**Wan2GP-native models** (wan, flux, hyvideo, ltx2, qwen, ace_step, etc.) are excluded —
they ship with Wan2GP and are maintained upstream.

## Status Legend

| Mark | Meaning |
|------|---------|
| PASS | All applicable criteria met |
| PARTIAL | Meets some criteria, documented gaps |
| FAIL | Does not meet the decomposition standard |
| BLOCKED | Cannot integrate without fundamental changes |

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

---

### kokoro — Kokoro 82M TTS

**Status: PASS (Amendment B)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{bert, bert_encoder, predictor, text_encoder, decoder}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PASS (Amendment B) — nn.Module definitions imported from Wan2GP's own multitalk kokoro module. Handler + phonemizer authored in fork. Documented in handler docstring. |
| 5 | Relative imports from source subdir | PASS — `from models.kokoro.kokoro_model` (Wan2GP package namespace, Amendment B) |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers`, `phonemizer` + Wan2GP modules |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/kokoro/kokoro_handler.py`, `kokoro_model.py`, `kokoro_phonemizer.py`
**Notes:** CPU model — no mmgp VRAM management needed. 5 nn.Modules decomposed for future GPU use.

---

### trellis — TRELLIS.2 4B (image-to-3D)

**Status: PASS (Amendment A)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 9 modules: `{ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024, tex_slat_flow_1024, shape_decoder, tex_decoder, image_cond, rembg}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PASS (Amendment A) — handler authored. Upstream source in `trellis2/` as declared subpackage |
| 5 | Relative imports from source subdir | PASS — `from .trellis2.pipelines...` (relative imports, no sys.path) |
| 6 | Import chain terminates at pip package | PASS — `torch`, `trimesh`, `spconv` (available in Docker image) |
| 7 | Zero monkeypatches / sys.path.insert | PASS — removed sys.path hack, using relative imports |

**Files:** `opt/wan2gp/models/trellis/trellis_handler.py`, `trellis2/` (upstream source)
**Notes:** Reference implementation for Amendment A pattern. 74 Python files of upstream TRELLIS source.

---

### vibevoice_asr — VibeVoice ASR 7B (microsoft)

**Status: PARTIAL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{language_model, acoustic_tokenizer, acoustic_connector, lm_head}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PARTIAL — `blocks.py` authored as conv codec architecture |
| 5 | Relative imports from source subdir | PASS — `from .vibevoice_asr.blocks import ...` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/vibevoice_asr/vibevoice_asr_handler.py`, `vibevoice_asr/blocks.py`
**Fixes applied:** Config key typo fixed (`acostic_vae_dim` → `acoustic_vae_dim`).
**Remaining:** Weight loading may have key nesting mismatch (NormConv1d wrapper). Needs validation against actual weights.

---

### vibevoice_tts — VibeVoice 7B TTS (community)

**Status: PARTIAL**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 7 modules: `{language_model, acoustic_tokenizer, semantic_tokenizer, prediction_head, acoustic_connector, semantic_connector, lm_head}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PARTIAL — `blocks.py`, `diffusion.py` authored |
| 5 | Relative imports from source subdir | PASS — `from .vibevoice_tts.blocks import ...` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `transformers`, `diffusers` |
| 7 | Zero monkeypatches | PASS |

**Files:** `opt/wan2gp/models/vibevoice_tts/vibevoice_tts_handler.py`, `vibevoice_tts/blocks.py`, `vibevoice_tts/diffusion.py`
**Fixes applied:** Config key typo fixed. `generate()` now runs full pipeline (LM → acoustic connector → diffusion → acoustic tokenizer decode).
**Remaining:** Weight key nesting may still mismatch. Diffusion head architecture may differ from upstream (plain MLP vs SwiGLU/adaLN). Needs validation against actual weights.

---

### anigen — AniGen (image-to-rigged-3D)

**Status: PASS (Amendment A)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{ss_flow_model, ss_decoder, slat_flow_model, slat_decoder, image_cond, dsine}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PASS (Amendment A) — handler authored. Upstream source symlinked as `anigen/` |
| 5 | Relative imports / importlib | PASS — vendor package registered via `importlib.util` (no sys.path.insert) |
| 6 | Import chain terminates at pip package | PASS — `torch`, `trimesh`, `spconv` |
| 7 | Zero monkeypatches / sys.path.insert | PASS — DSINE uses `_isolated_import` context manager (unavoidable: vendor code does `from models import dsine` conflicting with Wan2GP's `models` package) |

**Files:** `opt/wan2gp/models/anigen/anigen_handler.py`, `anigen/` (symlink → `vendor/anigen/`)
**Origin:** vendor/anigen/ — see vendor directory for commit details
**Notes:** DSINE normal estimation requires `_isolated_import` workaround. Handler registered in CUSTOM_HANDLERS.

---

### see_through — See-Through (anime layer decomposition)

**Status: PASS (Amendment A)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — 8 modules: `{ld_unet, ld_vae, ld_trans_vae, ld_text_encoder, ld_text_encoder_2, mg_unet, mg_vae, mg_text_encoder}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PASS (Amendment A) — handler authored. Upstream source symlinked as `seethrough/` |
| 5 | Relative imports / importlib | PASS — vendor subpackages (`modules`, `utils`) registered via `importlib.util` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `diffusers` |
| 7 | Zero monkeypatches / sys.path.insert | PASS |

**Files:** `opt/wan2gp/models/see_through/see_through_handler.py`, `seethrough/` (symlink → `vendor/seethrough/`)
**Origin:** vendor/seethrough/ — see vendor directory for commit details
**Notes:** Vendor code uses generic top-level import names (`modules.*`, `utils.*`) — registered via importlib during load_model().

---

### hy_motion — HY-Motion 1.0 (text-to-3D motion)

**Status: PASS (Amendment A)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Pipe dict >= 2 modules | PASS — `{motion_transformer, text_encoder}` |
| 2 | Modules are distinct subcomponents | PASS |
| 3 | No single-blob `model` key | PASS |
| 4 | Source written in fork, not copied | PASS (Amendment A) — handler authored. Upstream source symlinked as `hymotion/` |
| 5 | Relative imports / importlib | PASS — vendor package registered via `importlib.util` |
| 6 | Import chain terminates at pip package | PASS — `torch`, `torchdiffeq` |
| 7 | Zero monkeypatches / sys.path.insert | PASS |

**Files:** `opt/wan2gp/models/hy_motion/hy_motion_handler.py`, `hymotion/` (symlink → `vendor/hymotion/`)
**Origin:** vendor/hymotion/ — see vendor directory for commit details
**Notes:** Creates temp workspace with symlinks for Qwen3-8B, CLIP, stats. Patches config.yml with absolute paths.

---

## Summary

| Model | Status | Pipe Modules | Source | Weights | Inference |
|-------|--------|-------------|--------|---------|-----------|
| moss | PARTIAL | 4 | Authored + HF | Yes | Yes |
| kokoro | PASS (Amend B) | 5 | Wan2GP modules | Yes | Yes |
| trellis | PASS (Amend A) | 9 | Upstream subpkg | Yes | Yes |
| vibevoice_asr | PARTIAL | 4 | Authored | Needs validation | Untested |
| vibevoice_tts | PARTIAL | 7 | Authored | Needs validation | Partial (pipeline wired, weights TBD) |
| anigen | PASS (Amend A) | 6 | Upstream subpkg | Yes | Yes |
| see_through | PASS (Amend A) | 8 | Upstream subpkg | Yes | Yes |
| hy_motion | PASS (Amend A) | 2 | Upstream subpkg | Yes | Yes |

**Passing: 5/8** (kokoro, trellis, anigen, see_through, hy_motion via Amendment A/B)
**Remaining: 3** (moss needs HF trust_remote_code review, vibevoice_asr/tts need weight validation)
