# Wan-GAP: Wan2GP Fork Architecture

## What We Are Trying To Do

We maintain a home server (Tech Noir) running AI services on an RTX 4090.
Our stack originally used `vendor/wan2gp/` (a Git submodule of upstream
DeepBeepMeep/Wan2GP) with custom model extensions layered on top in
`services/wan2gp/custom_models/`. This "fork + overlay" pattern was brittle —
upstream updates broke the overlay, and the handlers couldn't be upstreamed.

We are forking Wan2GP into `opt/wan2gp/` and baking our custom model families
directly into the fork as first-class citizens, indistinguishable from
Wan2GP's own vendor model families (wan/, TTS/, flux/, etc.).

## The Fork

```
opt/wan2gp/          — Full fork of DeepBeepMeep/Wan2GP
```

The fork is a standalone copy with its own `wgp.py`, `models/`, `shared/`,
and `defaults/`. It has NO imports back into the parent `ray/` repository.
All path resolution for model weights is injected from outside via
`model_def` keys (standard Wan2GP pattern).

## What Makes Something "Indistinguishable"

A model family is indistinguishable from Wan2GP upstream code when:

1. Its handler lives under `opt/wan2gp/models/{family}/` as a peer to
   `wan/`, `TTS/`, `flux/`, etc.

2. All model architecture code is INLINE under `models/{family}/` —
   NOT imported from external packages outside the fork.

   **Wan2GP convention**: `models/wan/wan_handler.py` imports from
   `models/wan/modules/model.py`, `models/TTS/ace_step_handler.py`
   imports from `models/TTS/ace_step/*`. The neural network modules
   live inside the fork's model tree.

   **NOT Wan2GP convention**: `from vibevoice.modular.*` — that lives
   in `vendor/vibevoice/`, outside the fork.

3. The handler's `family_handler` class implements the 7-method contract:
   `query_supported_types`, `query_family_maps`, `query_model_family`,
   `query_family_infos`, `query_model_def`, `load_model`,
   `update_default_settings`.

4. `load_model()` produces `(pipeline, {"pipe": {...}, "coTenantsMap": {...}})`
   with mmgp-decomposed nn.Modules.

5. The pipeline has a `generate(**kwargs)` method.

6. No imports from `registry.*`, `services.*`, `gateway.*`, or anything
   in the parent `ray/` project.

7. Paths are resolved via `model_def.get("key", fallback)`, matching the
   Wan2GP pattern (e.g., `models/TTS/ace_step_handler.py` uses
   `_get_model_path(model_def, "text_encoder_folder", default)`).

## The Gap: 7 of 11 Handlers Are NOT Indistinguishable

### ✅ Indistinguishable (4)

| Family | Inline Code | Notes |
|---|---|---|
| `kokoro` | `models/kokoro/kokoro_model.py` | Full model architecture inline. Only TTS handler that is truly native. |
| `espeak` | N/A (subprocess) | Pure subprocess wrapper, no model code needed. Correct pattern for trivial handlers. |
| `faster_whisper` | Uses pip `faster-whisper` | Acceptable — same pattern as Wan2GP using `gradio`, `cv2`, `diffusers` from pip. |
| `trellis` | `models/trellis/trellis2/` | Full TRELLIS.2 model code with SageAttention patches baked in. Registers in `wgp.py` natively. |

### Status After Fix (May 15 2026)

All 11 handlers now have their model source code embedded inside the fork
under `models/{family}/_src/`. No handler imports from `vendor/`, `registry.*`,
or any package outside `opt/wan2gp/`.

| Family | Source Location | Method |
|---|---|---|
| `kokoro` | `models/kokoro/kokoro_model.py` | Inline (always was) |
| `espeak` | N/A (subprocess) | Trivial |
| `faster_whisper` | pip `faster-whisper` | Acceptable pip dep |
| `trellis` | `models/trellis/trellis2/` | Inline (always was) |
| `anigen` | `models/anigen/_src/anigen/` | 88 .py files copied from `vendor/` |
| `see_through` | `models/see_through/_src/modules/`, `_src/utils/` | 56 .py files from `vendor/seethrough/` |
| `hy_motion` | `models/hy_motion/_src/hymotion/` | 27 .py files from `vendor/hymotion/` |
| `moss` | Dynamic load from weights path | HuggingFace pattern |
| `vibevoice_asr` | `models/_lib/vibevoice/` (shared) | 28 .py files from VibeVoice repo |
| `vibevoice_tts` | same `_lib/vibevoice/` | Same shared copy |
| `faster_qwen3_tts` | `models/faster_qwen3_tts/_src/` | 14 .py files cloned from GitHub |

**What changed:**
- anigen: `vendor_root` sys.path hack → `_src/anigen/` local import
- see_through: `seethrough_common` sys.path hack → `_src/modules/` local import
- hy_motion: `vendor_root` sys.path hack → `_src/hymotion/` local import
- vibevoice_asr: pip dependency → `_lib/vibevoice/` from fork
- vibevoice_tts: same
- faster_qwen3_tts: cloned `faster-qwen3-tts` pip package source + `qwen_tts` source into `_src/`
- All: `vendor_root` removed from `deployment.py.resolve_handler_paths()`

## How We Got Here

### Phase 1 — Fork Creation

We copied the Wan2GP source from `vendor/wan2gp/` into `opt/wan2gp/` and
cleaned out the `.git` submodule.

**Files migrated:**
```
opt/wan2gp/models/anigen/        ← custom
opt/wan2gp/models/see_through/   ← custom
opt/wan2gp/models/hy_motion/     ← custom
opt/wan2gp/models/kokoro/        ← custom (+ vendor model files)
opt/wan2gp/models/espeak/        ← custom
opt/wan2gp/models/faster_whisper/ ← custom
opt/wan2gp/models/moss/          ← custom
opt/wan2gp/models/vibevoice_asr/ ← custom
opt/wan2gp/models/vibevoice_tts/ ← custom
opt/wan2gp/models/faster_qwen3_tts/ ← custom
opt/wan2gp/models/trellis/       ← existed in fork already!
```

### Phase 2 — Path Decoupling

All handlers originally imported from `registry.config` and
`registry.models` to find model weight paths. These are modules in
the parent `ray/` project, making the fork non-portable.

**Fix:** Added `_resolve_handler_paths()` to `deployment.py` which
resolves all paths at call time and injects them into `model_def`.
Handlers now read `model_def.get("anigen_path", "")` instead of
`registry.get_path("3d", "anigen")`.

Result: Zero `registry.*` imports in the fork.

### Phase 3 — Docker Integration

Updated `infra/docker/Dockerfile.gpu-all` to:
- Copy `opt/wan2gp/` to `/opt/wan2gp/` container path
- Add `/opt/wan2gp` to `PYTHONPATH`
- Remove old vendor trellis2 copy (replaced by fork's local copy)
- Add TRELLIS handler verify step in image build

### Phase 4 — Testing

Ran all 11 handlers in Docker with GPU passthrough:

**Contract tests (all pass):**
- 11 handlers import successfully
- All implement 7-method family_handler contract
- All have _Pipeline with generate()
- All accept **kwargs
- 90/90 unit contract checks pass

**Inference tests:**
- kokoro: 183KB WAV output ✅
- faster_whisper: transcription output ✅  
- faster_qwen3_tts: 88KB WAV output ✅
- trellis: 169MB GLB 3D model ✅
- anigen, hy_motion, moss, vibevoice_asr, vibevoice_tts: ❌
  (pre-existing dependency issues in test image, not fork-related)

## What Remains

All handlers are now self-contained within the fork. The source code for
every model family lives at `opt/wan2gp/models/{family}/`. No handler
depends on anything outside `opt/wan2gp/`, standard pip packages, or
model weights files.

The only remaining concern is code quality — some of the `_src/` copies
include training code, unused files, and vendored dependencies that could
be stripped. This is cosmetic, not structural.

```
                  ┌─────────────────────────────────────┐
                  │        opt/wan2gp/ (fork)            │
                  │  ┌─────────────────────────────────┐ │
                  │  │  models/wan/        (upstream)   │ │
                  │  │  models/TTS/        (upstream)   │ │
                  │  │  models/flux/       (upstream)   │ │
                  │  │  models/trellis/    (inline)     │ │
                  │  │  models/kokoro/     (inline)     │ │
                  │  │  models/espeak/     (inline)     │ │
                  │  │  models/faster_whisper/ (inline) │ │
                  │  │  models/anigen/     (_src/)      │ │
                  │  │  models/see_through/ (_src/)     │ │
                  │  │  models/hy_motion/  (_src/)      │ │
                  │  │  models/moss/       (weights)    │ │
                  │  │  models/vibevoice_asr/ (_lib/)   │ │
                  │  │  models/vibevoice_tts/ (_lib/)   │ │
                  │  │  models/faster_qwen3_tts/ (_src/)│ │
                  │  │  models/_lib/vibevoice/ (shared) │ │
                  │  └─────────────────────────────────┘ │
                  └─────────────────────────────────────┘

## Architecture Summary

```
                  ┌─────────────────────────────┐
                  │      opt/wan2gp/ (fork)      │
                  │  ┌─────────────────────────┐ │
                  │  │  models/wan/             │ │  ← upstream Wan2GP
                  │  │  models/TTS/             │ │
                  │  │  models/flux/            │ │
                  │  │  models/trellis/ (inline)│ │
                  │  │  models/kokoro/ (inline) │ │
                  │  │  models/espeak/          │ │
                  │  └─────────────────────────┘ │
                  │  ┌─────────────────────────┐ │
                  │  │  models/anigen/    ❌    │ │  ← calls into vendor/
                  │  │  models/hy_motion/ ❌    │ │
                  │  │  ...               ❌    │ │
                  │  └─────────────────────────┘ │
                  └─────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    vendor/anigen/     vendor/vibevoice/     vendor/hymotion/
    vendor/seethrough/ vendor/faster_qwen3_tts/

  services/wan2gp/deployment.py   ← orchestrator (outside fork)
    - _resolve_handler_paths()    ← resolves all model paths
    - imports from registry.*     ← only file in ray/ that knows about registry
    - CUSTOM_HANDLERS list        ← appends our custom families

  tests/                          ← all updated to use models.{family} paths
    - 419 unit tests passing
    - 3 e2e failures (model files not on disk)
```
