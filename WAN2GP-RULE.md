# Wan2GP Rule

## The Fundamental Principle

Every model under `models/{name}/` MUST be authored code, not vendored code.

"Authored" means:
- Written inside this fork as a thin wrapper around standard pip packages (`torch`, `transformers`, `diffusers`, `faster-whisper`, etc.)
- Zero files copied from `vendor/` or any external repository
- Zero files relocated from another directory into the model's source subdirectory
- The only source files in `models/{name}/{name}/` are files WRITTEN for this fork

"Vendored" is forbidden:
- No `vendor/anigen/` files exist anywhere in `models/anigen/`
- No `seethrough` files exist anywhere in `models/see_through/`
- No `hymotion` files exist anywhere in `models/hy_motion/`
- No `vibevoice` files exist anywhere in `models/_lib/`, `models/vibevoice_asr/`, or `models/vibevoice_tts/`
- No `faster_qwen3_tts` or `qwen_tts` package files in `models/faster_qwen3_tts/`
- No `moss` model files (`modeling_moss_tts.py`, `configuration_moss_tts.py`, etc.) copied from a HuggingFace model card into `models/moss/moss/`

## The Real Requirement: mmgp Decomposition

The entire reason for rewriting is mmgp VRAM management. "Looks like IndexTTS2" means:

**The handler's `load_model()` must return individually swappable nn.Modules in the pipe dict.**

IndexTTS2 does this:
```python
pipe = {"gpt": gpt_module, "vqvae": vqvae_module, "bigvgan": bigvgan_module}
# mmgp can swap gpt to GPU, run it, swap it out, swap vqvae in
```

A model that returns `pipe = {"model": model}` is NOT decomposed. mmgp can only swap the entire model as one blob — nothing gained.

**Every model's source must be written with separate nn.Module classes** so the handler can extract them. These classes are authored code, not model card files. They live in `models/{name}/{name}/`.

## The Test

A model passes the decomposition test when:
1. `load_model()` returns a pipe dict with >= 2 independently swappable nn.Modules
2. Each module corresponds to a distinct subcomponent of the architecture (language_model, encoder, decoder, head, etc.)
3. No module is just `model` (the entire loaded pretrained model as one blob)
4. The source code for each module is WRITTEN inside the fork, not copied from a model card or vendor directory
5. Every import uses `.module` relative imports from the model's source subdirectory
6. Every import chain terminates at a standard pip package (`torch`, `transformers`, etc.)
7. Zero monkeypatches, zero shims, zero compatibility hacks for the installed pip versions

## Amendment A: Large Model Exception

Some models are too complex for thin-wrapper decomposition (sparse 3D convolutions,
mesh processing, custom ODE solvers, texture baking, etc.). These follow the TRELLIS
pattern instead:

**A large model passes when:**
1. `load_model()` returns a pipe dict with >= 2 independently swappable nn.Modules (same as rule 1)
2. The handler file (`{name}_handler.py`) is authored in this fork
3. Upstream source code lives as a declared subpackage within the handler directory (e.g., `models/trellis/trellis2/`)
4. The subpackage is imported via relative imports (`from .trellis2.pipelines...`), not `sys.path` hacks
5. The subpackage origin is documented (repo URL, commit hash, or version tag)
6. The handler decomposes the upstream pipeline into nn.Modules for mmgp
7. No `sys.path.insert()` calls (use relative imports or `importlib`)

**Reference implementation:** `models/trellis/trellis_handler.py` + `models/trellis/trellis2/`

**Applies to:** trellis, anigen, see_through, hy_motion — models with no pip package equivalent
where the upstream source is too large to author from scratch (>1000 LOC).

**Does NOT apply to:** moss, kokoro, vibevoice — these MUST follow the standard rules
because thin wrappers around pip packages are feasible.

## Amendment B: Wan2GP Dependency Exception

Wan2GP ships with its own model implementations under `models/wan/`, including
multitalk modules. A handler may import nn.Module definitions from Wan2GP's own
codebase (e.g., `models.wan.multitalk.kokoro`) as if they were pip packages.

This is acceptable because:
- Wan2GP is the host framework — its modules are always available
- The handler itself is still authored (orchestration, weight loading, generate logic)
- The nn.Module definitions come from the same codebase, not external vendors

Requirement: the import must be explicit and documented in the handler docstring.

## Enforcement

Before declaring any model "done":

1. Check `load_model()` return value — does it return multiple swappable modules or one blob?
2. Check source files — were they written here or copied from elsewhere? (`git log --diff-filter=A --name-only` on each file)
3. Check every import — does it trace to a pip package or to another authored file?
4. Check for monkeypatches — are there any `ProcessorMixin.__init__ =` or similar hacks?

If the answer to any of these is no, the model is NOT done. There is no shortcut.
