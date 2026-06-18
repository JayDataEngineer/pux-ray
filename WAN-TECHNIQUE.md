# Wan2GP Model Integration Technique

## How Wan2GP Integrates Complex Models (The IndexTTS2 Way)

IndexTTS2 is a massive model — GPT backbone, BigVGAN vocoder, s2mel codec,
VQVAE, semantic model, campplus speaker encoder, Qwen emotion encoder,
accelerated inference engine — over 200 source files. Yet Wan2GP integrated
it as a **native model family** with zero external dependencies, zero sys.path
hacks, zero vendor package copies.

This document explains HOW they did it, and WHY our approach was wrong.

---

## The Technique

### 1. Full Source Tree Inlining

Every model family gets a complete source directory under `models/{family}/`:

```
models/TTS/index_tts2/
├── pipeline.py         ← high-level API (what the handler calls)
├── infer_v2.py         ← inference orchestration
├── gpt/                ← full GPT model implementation
│   ├── model_v2.py
│   ├── conformer/
│   └── perceiver.py
├── BigVGAN/            ← full vocoder implementation
│   ├── bigvgan.py
│   ├── alias_free_activation/
│   ├── alias_free_torch/
│   └── nnet/
├── s2mel/              ← full speech codec implementation
│   ├── dac/
│   ├── modules/
│   └── wav2vecbert_extract.py
├── vqvae/              ← VQ-VAE decoder
├── accel/              ← accelerated inference (CUDA graphs, KV cache)
├── utils/              ← masking, feature extraction, text processing
└── configs/            ← runtime configs
```

**206 files total.** Every neural network module, every attention variant,
every activation function — all inside the fork.

### 2. Relative Imports Only

The handler at `models/TTS/index_tts2_handler.py` does:

```python
from .index_tts2.pipeline import IndexTTS2Pipeline
#        ^^^^^^^^^^^^^^^^
#        relative import into its own sibling directory
```

The pipeline at `models/TTS/index_tts2/pipeline.py` does:

```python
from .infer_v2 import IndexTTS2
from .gpt.model_v2 import UnifiedVoice
from .utils.maskgct_utils import build_semantic_model
```

Every import is a **relative import** within the model's own directory tree.
No absolute imports to external packages, no sys.path manipulation.

### 3. Standard Pip Packages Only

The handler imports from pip packages that are common ML infrastructure:

```python
import torch
import numpy as np
from omegaconf import OmegaConf
import librosa
```

Wan2GP's `requirements.txt` already lists these. No model-specific packages.

### 4. Wan2GP Framework Utilities

```python
from shared.utils import files_locator as fl
from shared.mps import mps_device_or
```

`files_locator` is Wan2GP's built-in weight discovery system. It searches
pre-configured download directories and can auto-download from HuggingFace.
This is the ONLY path resolution mechanism — no `registry.*`, no `Config()`.

### 5. Pipeline as the Public API

The source tree exports a single `Pipeline` class that the handler wraps:

```python
# models/TTS/index_tts2_handler.py
from .index_tts2.pipeline import IndexTTS2Pipeline

pipeline = IndexTTS2Pipeline(
    ckpt_root=fl.get_download_location(),
    device=runtime_device,
    gpt_weights_path=gpt_weights_path,
)

# Decompose into mmgp-managed modules
pipe = {
    "transformer": pipeline.model.gpt,
    "transformer2": pipeline.model.s2mel,
    "vocoder": pipeline.model.bigvgan,
    ...
}
```

### 6. Weight Discovery via files_locator

Weights are found by Wan2GP's standard discovery system:

```python
gpt_weights_path = fl.locate_file(INDEX_TTS2_MAIN_GPT_FILENAME, error_if_none=False)
if gpt_weights_path is None:
    raise FileNotFoundError(...)
```

No hardcoded paths. No env vars. No registry lookups. `files_locator` handles
all of it — searching download dirs, HuggingFace cache, and configured roots.

---

## Why Our Approach Was Wrong

### What We Did (The Hack)

```python
# anigen_handler.py — OUR code
vendor = paths.get("vendor_root", "")
if vendor and vendor not in sys.path:
    sys.path.insert(0, vendor)
from anigen.pipelines.anigen_image_to_3d import AnigenImageTo3DPipeline
```

```python
# see_through_handler.py — OUR code
vendor = paths.get("vendor_root", "")
...
seethrough_common = str(Path(vendor) / "seethrough" / "common")
if seethrough_common not in sys.path:
    sys.path.insert(0, seethrough_common)
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
```

Three fundamental mistakes:

**Mistake 1: sys.path manipulation.** Wan2GP never does this. Every model's
source code lives inside `models/{family}/` and uses relative imports.
Adding to sys.path is a fragile hack — it depends on import order, can
collide with pip packages, and breaks when the fork is moved.

**Mistake 2: External vendor packages.** The model source lives in `vendor/`,
NOT inside the fork. This means the fork alone is incomplete. An upstream
developer looking at `models/anigen/anigen_handler.py` sees
`from anigen.pipelines.*` and has no idea where `anigen` comes from —
it's not a pip package, it's not in the fork tree.

**Mistake 3: registry.path instead of files_locator.** Wan2GP's native
`files_locator` already handles all weight discovery — looking up
download directories, HuggingFace cache, and configured model roots.
Our handlers used `registry.get_path()` which is a `ray/`-specific API
that doesn't exist in the fork. We "fixed" this by injecting paths through
`model_def` — but the right fix is to use `files_locator` like every
other Wan2GP handler does.

---

## What Each Handler Needs

### anigen
Copy `vendor/anigen/anigen/` (the full source tree at vendor/anigen/anigen/)
into `models/anigen/_src/`. Update handler imports from `from anigen.pipelines.*`
to `from ._src.pipelines.*`. Remove sys.path manipulation.

### see_through
Copy `vendor/seethrough/common/modules/layerdiffuse/`,
`vendor/seethrough/common/modules/marigold/`, and `vendor/seethrough/common/utils/`
into `models/see_through/_src/`. Update imports from `from modules.layerdiffuse.*`
to `from ._src.modules.layerdiffuse.*`. Remove sys.path manipulation.

### hy_motion
Copy `vendor/hymotion/` into `models/hy_motion/_src/`. Update imports from
`from hymotion.*` to `from ._src.hymotion.*`. Remove sys.path manipulation.

### vibevoice_asr & vibevoice_tts
Copy `infra/repos/VibeVoice-Community/vibevoice_community/` (the actual
vibevoice package source) into `models/vibevoice_asr/_src/vibevoice/` and
`models/vibevoice_tts/_src/vibevoice/`. Update imports from
`from vibevoice.modular.*` to `from ._src.vibevoice.modular.*`.
Remove pip dependency.

### faster_qwen3_tts — REMOVED
Qwen3-TTS was removed in Session 3d cleanup (2026-06-18). MOSS VoiceGenerator
(Tier A) covers instruction-following + multilingual TTS; Kokoro (sherpa-onnx,
port 8060, CPU) covers lightweight TTS. No capability lost. See §17 in
TEST-REPORT.md for the current TTS coverage.

The old handler used `faster_qwen3_tts.utils`, `faster_qwen3_tts.model`,
and `qwen_tts.Qwen3TTSModel`, resolving to
`models/TTS/qwen3/inference/qwen3_tts_model.py`. All 5440 lines of handler
code + 8.1 GB of weights were dropped.

### moss
MOSS dynamically loads model code from the weights directory using
`importlib`. This is a valid HuggingFace pattern (model code bundled
with weights). The handler should be refactored to use `files_locator`
for finding the weights path, but the dynamic loading itself is acceptable.

---

## Verification Checklist

For a handler to be indistinguishable from Wan2GP upstream:

- [ ] Source lives in `models/{family}/` directory tree
- [ ] All imports use relative paths (`.submodule`) or pip packages
- [ ] No `sys.path.insert()` or `sys.path.append()` in the handler
- [ ] No imports from `anigen.*`, `hymotion.*`, `vibevoice.*`, `modules.*` etc.
- [ ] Weights discovered via `files_locator` or `model_def.get()` — NOT `registry.*`
- [ ] Pipeline objects have `generate(**kwargs)` interface
- [ ] Handler implements all 7 `family_handler` static methods
- [ ] `load_model()` returns `(pipeline, {"pipe": dict, "coTenantsMap": dict})`
