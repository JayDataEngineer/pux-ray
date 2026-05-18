# See-Through — Original Workflow

**Source**: Part of the live2d-parsing ecosystem (vendor code at `vendor/seethrough/`)
**Type**: Anime layer decomposition — separates anime images into individual layers with depth
**Architecture**: Dual-pipeline (LayerDiff + Marigold depth estimation)

## Inference Pipeline (3 Stages)

### Stage 1: LayerDiff — Body Part Extraction
```
anime image → encode with ld_text_encoder + ld_text_encoder_2 (SDXL-style dual encoders)
→ ld_vae encodes image to latent space
→ ld_unet denoises to extract body part masks
→ ld_trans_vae decodes with transparency (alpha channel)
→ output: N transparent body part images
```

### Stage 2: Marigold — Depth Estimation
```
each body part image → encode with mg_text_encoder (empty prompts)
→ mg_vae encodes to latent
→ mg_unet runs depth estimation (multiple denoising steps)
→ output: depth map per layer
```

### Stage 3: Post-processing
```
depth maps → sort layers by depth median
→ output: layered image with correct z-ordering (Live2D-ready)
```

## Components (8 nn.Modules)

**LayerDiff (5 modules):**

| Module | Role | Size | Notes |
|--------|------|------|-------|
| ld_unet | UNetFrameConditionModel — denoising for body part extraction | ~3GB | SDXL-based |
| ld_vae | VAE encoder/decoder | ~150MB | Standard SDXL VAE |
| ld_trans_vae | TransparentVAE — decode with alpha channel | ~200MB | Custom transparency decoder |
| ld_text_encoder | First text encoder | ~500MB | SDXL text encoder 1 |
| ld_text_encoder_2 | Second text encoder | ~2GB | SDXL text encoder 2 (larger) |

**Marigold (3 modules):**

| Module | Role | Size | Notes |
|--------|------|------|-------|
| mg_unet | UNet — depth estimation | ~3GB | SD-based |
| mg_vae | VAE encoder/decoder | ~150MB | Standard VAE |
| mg_text_encoder | Text encoder (empty prompts) | ~500MB | SD text encoder |

## Key Characteristics

- **Two independent diffusion pipelines** — LayerDiff runs first, then Marigold runs per-layer
- **SDXL-based architecture** — uses standard SDXL components (dual text encoders, UNet, VAE)
- **Marigold is run N times** — once per extracted body part layer
- **No exotic components** — standard diffusion UNet + VAE + text encoders
- **Peak VRAM ~8-10GB** — lighter than TRELLIS/AniGen
