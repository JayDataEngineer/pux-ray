# GAPS-6: Code Audit — Dead, Slop, and DRY

Audited `services/`, `gateway/`, and `scripts/` across three axes: dead code, slop (bloat/over-engineering), and DRY violations.

## A) Dead Code

### High confidence

| What | Where | Detail |
|------|-------|--------|
| `SubprocessProxyMixin` class | `services/base.py:208-275` | Defined but **never imported or used** anywhere. Only referenced in CLAUDE.md docs. Was the ComfyUI proxy base before the refactor to Forge. Safe to delete. |
| `MODELS_ROOT` constant | `services/base.py:21` | Defined at module level but never read in this file or imported elsewhere. |
| `PROJECT_ROOT` constant | `services/base.py:20` | Same — defined but never used. |

### Medium confidence

| What | Where | Detail |
|------|-------|--------|
| `quant` parameter | `services/wan2gp/deployment.py:367` | Accepted by `load()` but only forwarded to `_load_model()` — needs verification whether it's actually passed by callers. |
| Empty `__init__.py` | `services/wan2gp/__init__.py` | Zero contents. Could be removed if package discovery doesn't need it. |

---

## B) Slop Code

### High severity (real maintenance burden)

| What | Where | Fix |
|------|-------|-----|
| `_build_cmd()` — 125 lines | `services/llm/deployment.py:81-205` | Monolithic flag builder. Extract `_base_cmd()`, `_engine_flags()`, `_sampling_flags()` etc. |
| `_load_model()` — 84 lines | `services/wan2gp/deployment.py:504-588` | Does path resolution, loading, and configuration all in one method. Split into `_resolve_paths()`, `_configure_model()`, `_create_pipeline()`. |
| `_resolve_handler_paths()` — 35 lines of if/elif | `services/wan2gp/deployment.py:805-840` | Should be a lookup dict or factory, not a chain of model-type conditionals. |
| Bare `except (ImportError, RuntimeError): pass` | `services/base.py:42-44` | Silently swallows errors. At minimum, log the exception. |

### Medium severity

| What | Where | Fix |
|------|-------|-----|
| `_async_call` / `_async_call_raw` duplicate sync versions | `services/forge_subprocess.py:121-198` | Four methods (`_call`, `_call_raw`, `_async_call`, `_async_call_raw`) all do the same thing — build a URL, make a request, return the body. Consolidate with a single internal method that takes sync/async client. |
| `_do_load()` mixed responsibilities | `services/forge.py:144-169` | Estimation, allocation, and loading in one method. Split concerns. |
| Verbose test pattern | `scripts/test_services_v2.py` | Same setup/teardown repeated per test. Extract a test helper. |

---

## C) DRY Violations

### High severity (3+ files)

| What | Occurrences | Fix |
|------|------------|-----|
| **`family_handler` class boilerplate** | 10 handlers in `services/wan2gp/custom_models/*/` | Every handler defines the same 7 static methods (`query_supported_types`, `query_family_maps`, `query_model_family`, `query_family_infos`, `query_model_def`, `load_model`, `update_default_settings`). The first 5 are pure boilerplate — differ only in string constants. **Extract a `BaseFamilyHandler` base class** with the shared interface. Each handler only overrides what's unique. |
| **Process kill pattern** | `services/base.py:64-72`, `services/base.py:192-203` | Same `os.killpg` → fallback `os.kill` → catch `ProcessLookupError` duplicated. Extract to `kill_process(pid)` utility. |
| **Model download/fallback pattern** | `kokoro_handler.py`, `vibevoice_tts_handler.py`, `anigen_handler.py` | Same "check if model exists → `snapshot_download()` → continue" flow. Extract to `ensure_model_downloaded(model_type, model_name, patterns)`. |
| **base64 response wrapping** | 5+ handlers | `base64.b64encode(data).decode()` with same dict structure everywhere. Extract `build_response(data, media_type)`. |

### Medium severity (2 files, significant duplication)

| What | Files | Fix |
|------|-------|-----|
| **GPU memory utilities** | `services/base.py:26-44`, `services/forge_base.py:54-62` | Both have `actual_vram_mb()` / `torch.cuda.memory_allocated()` helpers. Move to shared `gpu_utils.py`. |
| **SubprocessMixin vs ForgeSubprocessMixin** | `services/base.py`, `services/forge_subprocess.py` | Both manage subprocess lifecycle (start/stop/health-check). Significant overlap in `kill_process`, `wait_for_health`, and health polling loops. |

---

## Priority Order (bang-for-buck)

1. **Delete `SubprocessProxyMixin`** — DONE
2. **Extract `BaseFamilyHandler`** — DONE; 10 handlers refactored, ~100 lines boilerplate removed
3. **Consolidate subprocess HTTP methods** — DONE; `_url()` helper, imports `kill_process_tree`
4. **Extract `kill_process` utility** — DONE; `forge_subprocess.py` imports from `base.py`
5. **Split `_build_cmd()`** — DONE; extracted `_base_cmd`, `_add_cache_flags`, `_add_vision_flags`, `_add_sampling_flags`
6. **Remove `MODELS_ROOT` / `PROJECT_ROOT`** — DONE
7. **Extract model download helper** — SKIPPED; only kokoro uses snapshot_download (1 occurrence)
8. **Extract base64 response builder** — DONE; `audio_response()` in `base_handler.py`, applied to 6 handlers

## Additional fixes applied

- **`_resolve_handler_paths()`** — DONE; 120-line if/elif chain replaced with `_PATH_MAP` / `_MULTI_PATH_MAP` data-driven lookup + `_resolve_hymotion_paths()` for the special case
- **`_load_model()`** — DONE; extracted `_unwrap_pipe()` and `_apply_mmgp_profile()` static methods
- **`_do_load()`** — DONE; split into `_reserve_vram()` + `_reconcile_vram()`
- **Bare `except: pass`** — DONE; now logs the exception at debug level
- **SubprocessMixin merge** — SKIPPED; the two classes serve different patterns (split start/wait vs combined) — merging would lose flexibility
- **Test helper** — SKIPPED; `test_services_v2.py` already has a clean `test()` helper
