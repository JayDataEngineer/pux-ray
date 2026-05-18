# WhatDreamsCost — Original Workflow

**Source**: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
**Type**: Video generation pipeline (ComfyUI custom nodes on LTX Video 2.3)
**Architecture**: Conditioning strategies on top of LTX Video — timeline scheduling, first/last frame prompting

## What This Is

WDC is NOT a model. It is a set of **conditioning strategies** for LTX Video 2.3:
- **Backbone model**: LTX Video 2.3 (already Wan2GP built-in: models.ltx2.ltx2_handler)
- **Additional models**: LTX video VAE, audio VAE, spatial upscaler
- **Custom nodes**: LTXDirector (timeline), LTXSequencer (frame keyframes), LTXVConcatAVLatent

## Inference Workflows

### 1. LTX Director (Multi-shot timeline)
```
text prompt + segment definitions → DualCLIP encoder
  → LTXDirector (shot plan: 5 segments × independent camera)
  → LTXDirectorGuide ×2 (camera A/B guidance)
  → LTXVCropGuides (crop regions)
  → Parallel sampling: audio latent path + video latent path
  → LTXVConcatAVLatent → spatial upscale → VAE decode → video
```

### 2. FFLF 2-Stage (First frame / Last frame)
```
first_frame + last_frame → LTXSequencer (97-frame keyframes)
  → LTXVideo model (2x upscale) → video output
```

### 3. FFLF 3-Stage
```
Same as 2-stage + additional 1.5x upscale pass (3x total)
```

### 4. FFLF + Audio
```
first_frame + audio file → LTXSequencer
  → LTXVideo model → LTXVConcatAVLatent
  → audio VAE decode + video VAE decode → video with audio
```

## Key Observation

Every WDC workflow is: **LTX Video + conditioning parameters**. The
conditionings are:
- Timeline segmentation (shot boundaries, camera guides)
- First/last frame keyframe interpolation
- Audio conditioning for lip-sync / rhythmic timing
- Spatial upscaling between stages

All of these are **parameters to a single LTX Video generate() call** or
at most a 2-stage pipeline (generate at low res → generate again at high res).
