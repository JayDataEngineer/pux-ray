# Wan2GP Model Integration Gaps

Models that are decomposed and passing the rule sheet but have unresolved issues
preventing correct inference or weight loading.

## VibeVoice ASR/TTS — Weight Architecture Mismatch

**Severity: Critical (models produce garbage output)**

The authored `VibeVoiceAcousticTokenizer` uses a 2-layer conv stack:
```
SConv1d → nn.Conv1d
```

The actual checkpoints have a 3-layer stack:
```
SConv1d → NormConv1d → nn.Conv1d
```

`_load_and_strip(strict=False)` silently drops all conv weights because the key
names don't match. The modules exist in the pipe dict but half the parameters
are random initialization.

### TTS-specific issues
- `generate()` pipeline flow is speculative wiring (LM → connector → diffusion → decode),
  never tested against real weights
- Diffusion head (`VibeVoiceDiffusionHead`) uses plain MLP layers; upstream may use
  SwiGLU with adaLN modulation — the 26 `model.prediction_head.*` keys won't map
- Even if weights loaded correctly, output would likely be noise

### Fix
Examine the actual checkpoint's `model.acoustic_tokenizer.*` key structure and
either:
1. Add the missing `NormConv1d` wrapper to match upstream architecture
2. Or map the triple-nested keys to the double-nested authored structure

**Files:** `opt/wan2gp/models/vibevoice_asr/vibevoice_asr/blocks.py`,
`opt/wan2gp/models/vibevoice_tts/vibevoice_tts/blocks.py`,
`opt/wan2gp/models/vibevoice_tts/vibevoice_tts/diffusion.py`

---

## Duplicate Handlers in services/wan2gp/custom_models/

**Severity: Low (housekeeping)**

The following handlers were relocated to `opt/wan2gp/models/` but the originals
in `services/wan2gp/custom_models/` still exist as duplicates:

- `services/wan2gp/custom_models/anigen_handler/` → now `opt/wan2gp/models/anigen/`
- `services/wan2gp/custom_models/see_through/` → now `opt/wan2gp/models/see_through/`
- `services/wan2gp/custom_models/hy_motion/` → now `opt/wan2gp/models/hy_motion/`
- `services/wan2gp/custom_models/kokoro/` → now `opt/wan2gp/models/kokoro/`

### Fix
Delete the duplicate directories. The canonical handlers are in `opt/wan2gp/models/`.

**Directory:** `services/wan2gp/custom_models/`
