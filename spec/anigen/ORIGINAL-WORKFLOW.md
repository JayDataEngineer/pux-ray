# AniGen — Original Workflow

**Source**: https://github.com/VAST-AI-Research/AniGen
**Type**: Image-to-rigged-3D (mesh + skeleton + skin weights)
**Architecture**: Multi-stage pipeline with S³ Fields (Shape, Skeleton, Skin)

## Inference Pipeline

### Stage 1: Image Preprocessing
```
image → rembg (BiRefNet, if no alpha) → DSINE normal estimation → RGB + normals
```

### Stage 2: Conditioning
```
image → image_cond model (DiNOv2-based) → condition vectors
```

### Stage 3: Sparse Structure (SS) Flow
```
random noise → ss_flow_model (N denoising steps, flow matching) → sparse structure
→ ss_decoder → binary occupancy grid + skeleton scaffold
```

### Stage 4: Structured Latent (SLAT) Flow
```
sparse structure → slat_flow_model (N denoising steps, flow matching)
→ slat_decoder → dense geometry + skeleton + skinning weights
```

### Stage 5: Post-processing & Export
```
decoded output → skin weight smoothing → geodesic filtering → GLB mesh with rig
```

## Components (6+ nn.Modules)

| Module | Role | Size | Notes |
|--------|------|------|-------|
| `ss_flow_model` | Sparse structure denoising | ~2GB | Flow matching |
| `ss_decoder` | Structure decoding | ~500MB | Continuous → binary |
| `slat_flow_model` | Dense geometry + articulation | ~4GB | Flow matching with S³ fields |
| `slat_decoder` | Latent → mesh + skeleton + skin | ~1GB | Sparse convolution |
| `image_cond` | Image feature extraction | ~1GB | DiNOv2-based (same as TRELLIS) |
| `dsine` | Normal estimation | ~300MB | Dense prediction, FP32 |

Additional: `rembg` (BiRefNet) — background removal, shared with TRELLIS.

## Key Characteristics

- **Sequential pipeline** similar to TRELLIS but with skeleton/skinning added to the latent representation
- **S³ Field representation** — shape, skeleton, and skinning as mutually consistent fields over shared spatial domain
- **DSINE normal estimation** — additional preprocessing step not in TRELLIS
- **Post-processing** — skin weight smoothing and geodesic filtering after decode
- **Confidence-decaying skeleton field** — handles geometric ambiguity at Voronoi boundaries
- **Peak VRAM ~18GB** — heavier than TRELLIS due to skeleton/skin fields

## Relationship to TRELLIS

AniGen is architecturally similar to TRELLIS (sparse structure → structured latents → decode) but adds:
- Skeleton generation (joint positions, bone connectivity)
- Skin weight prediction (vertex-to-bone binding)
- DSINE normal estimation as additional conditioning
- Post-processing for rig quality

Shares components: DiNOv2 image conditioning, BiRefNet background removal, sparse convolution decoders.
