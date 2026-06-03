# GAPS-4: Wan2GP Fork Architecture Gaps

Audit of our Wan2GP integration — pattern inconsistencies, coupling surface, and
convergence plan. The fork is stable (10 handlers, 208 tests passing) but handlers
have drifted into inconsistent patterns that should be standardized.

## Gap 1: Path Resolution — Mixed Strategies

**Problem:** No two handlers resolve model paths the same way.

| Handler | Strategy |
|---------|----------|
| pixal3d | spec-first (`registry.specs.resolve()` → fallback `registry.get_path()`) |
| moss | spec-first (`registry.specs.resolve()` → fallback `model_def`) |
| anigen | direct `registry.get_path()` |
| see_through | direct `registry.get_path()` |
| hy_motion | `Config()`-based path resolution |
| kokoro | `model_def` paths only |
| espeak | binary path from `shutil.which` |
| faster_whisper | `model_def` paths only |
| vibevoice_asr/tts | `model_def` paths only |

**Target:** All handlers that load weights from disk should use spec-first
resolution: try `registry.specs.resolve()` first, fall back to
`registry.get_path()`, fall back to `model_def`. Same code path, same error
handling.

**Why it matters:** Spec resolution is quant-aware and gives us a single source
of truth for module paths. Without it, adding a new quant variant means editing
individual handlers.

## Gap 2: Vendor Import Patterns — Inconsistent Complexity

**Problem:** Four handlers import vendor code, each differently.

| Handler | Pattern | Complexity |
|---------|---------|------------|
| pixal3d | Clean `from .pixal3d.pipelines import ...` | Simple |
| anigen | `importlib.util` + `sys.modules` registration | Complex |
| see_through | `importlib.util` + `sys.modules` registration | Complex |
| hy_motion | `importlib.util` + `sys.modules` registration | Complex |

The importlib-based handlers register vendor packages in `sys.modules` to make
relative imports work. Pixal3D achieves the same result with just a relative
import because its symlink + `__init__.py` is set up correctly.

**Target:** All vendor-backed handlers should work like pixal3D — relative import
via the symlink, no `sys.modules` manipulation. If a vendor package needs setup,
put it in the handler's `__init__.py`, not inside `load_model()`.

## Gap 3: Weight Loading — Three Different Methods

**Problem:** Three different approaches to loading checkpoints.

| Method | Handlers |
|--------|----------|
| `safetensors.torch.load_file()` + `_load_state_dict()` | moss, vibevoice_asr, vibevoice_tts, anigen, see_through, pixal3d |
| `torch.load()` | kokoro, hy_motion |
| Model-specific constructor | faster_whisper (`WhisperModel`) |

The `_load_state_dict()` helper is copy-pasted across handlers with slight
variations. `torch.load()` is slower and less safe than safetensors.

**Target:** Common weight loading utility in `opt/wan2gp/models/_shared.py`
(or similar) that all handlers use. Safetensors-first with torch.load fallback
for legacy checkpoints. Handler just calls `load_weights(path, model)`.

## Gap 4: Subprocess Handlers Don't Fit the Pattern

**Problem:** espeak and faster_whisper return empty pipe dicts (`{}, {}`) because
they don't use nn.Modules. They wrap subprocesses (espeak-ng binary, CTranslate2).

This means:
- mmgp can't manage their memory (there's nothing to manage)
- Co-tenants concept doesn't apply
- `_Pipeline.generate()` is just a subprocess call wrapper

**Target:** Accept this as a valid handler variant. The contract already works —
they return empty pipe/coTenants and that's fine. No action needed beyond
documenting it.

## Gap 5: wgp.py Dependency is 12K Lines for One Import

**Problem:** `deployment.py` imports `wgp` once to get `family_handlers`. That
file is 12,330 lines and includes Gradio UI, model management, CLI — none of
which we use.

**Target:** Replace the `wgp` import with our own lightweight discovery that scans
`opt/wan2gp/models/*/[name]_handler.py` for `family_handler` classes. Drop the
dependency on Wan2GP's entry point entirely.

**Scope:** Only matters if Wan2GP makes breaking changes to `wgp.py`. Low urgency.

## Gap 6: No Shared Handler Base Class

**Problem:** Every handler implements the 7-method family_handler contract from
scratch. `query_supported_types()`, `query_model_family()`, `query_family_infos()`,
`query_model_def()`, `query_family_maps()`, `update_default_settings()` are
boilerplate repeated 10 times with minor variations.

**Target:** Optional base class with sensible defaults. Handler only overrides
what's unique. Not mandatory — existing handlers can keep working as-is.

## Coupling Summary

What we actually depend on from Wan2GP:

| Dependency | Used By | Replaceable? |
|-----------|---------|-------------|
| `wgp.family_handlers` list | deployment.py (1 import) | Yes — scan models/ dir |
| `mmgp` VRAM offload | deployment.py (2 imports) | Yes — `pip install mmgp` standalone |
| Wan2GP native model families | wan, hunyuan, flux, etc. | No — would need to write handlers for each |
| Wan2GP requirements.txt | Docker image base deps | Yes — but it's a curated list we'd need to maintain |

**Total coupling surface:** 3 imports. 10 custom handlers are fully self-contained.

## Convergence Priority

1. **Gap 1 (Path Resolution)** — Standardize to spec-first. Touches 8 handlers.
2. **Gap 3 (Weight Loading)** — Extract shared utility. Touches 8 handlers.
3. **Gap 2 (Vendor Imports)** — Simplify to pixal3D pattern. Touches 3 handlers.
4. **Gap 5 (wgp.py)** — Replace discovery. 1 file change.
5. **Gap 6 (Base Class)** — Optional, future cleanup.
6. **Gap 4 (Subprocess)** — No action, document only.

## Decision: Keep Fork, Don't Write Own System

The fork is stable and the coupling is thin (3 imports). Writing our own system
would be ~3,700 lines of replacement code for no functional gain. The right move
is to standardize patterns within the fork so handlers are consistent and a future
migration (if needed) would be trivial.

Migrate away from Wan2GP only if:
- Wan2GP makes a breaking change that forces fork maintenance
- We need features Wan2GP actively prevents (first-class subprocess services)
- Wan2GP development stops and Python/PyTorch move on without it
