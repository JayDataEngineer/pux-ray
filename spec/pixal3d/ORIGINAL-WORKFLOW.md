# Pixal3D — Original Workflow

**Source**: https://github.com/TencentARC/Pixal3D
**Type**: Image-to-3D (high-fidelity, near-reconstruction level)
**Architecture**: TRELLIS.2 fine-tune with projection-mode (pixel-aligned) conditioning

## Inference Pipeline (5 Stages)

### Stage 1: Image Preprocessing
```
image → rembg (BiRefNet, if no alpha) → center crop → resize
Optional: MoGe-2 for automatic FOV estimation
```

### Stage 2: Sparse Structure
```
image → projection-mode conditioning (pixel features lifted to 3D via back-projection)
random noise → ss_flow_model (N denoising steps) → sparse structure
→ ss_decoder → binary occupancy grid
```

### Stage 3: Shape SLat (Cascade 512 → 1024)
```
sparse coords → slat_flow_512 (N steps) → shape latents at 512
shape_slat_512 → upsample → high-res coords
→ slat_flow_1024 (N steps) → shape latents at 1024
```

### Stage 4: Texture SLat
```
shape_slat → tex_slat_flow_512 (N steps) → texture latents at 512
→ tex_slat_flow_1024 (N steps) → texture latents at 1024
```

### Stage 5: Decode & Export
```
shape_slat → shape_decoder → mesh (vertices + faces)
tex_slat → tex_decoder → PBR texture voxels
mesh + texture → o_voxel export → GLB bytes (trimesh fallback)
```

## Components (13 nn.Modules)

| Module | Role | Size | Notes |
|--------|------|------|-------|
| ss_flow_model | Sparse structure denoising | ~2GB | Same as TRELLIS |
| ss_decoder | Structure decoding | ~500MB | Same as TRELLIS |
| slat_flow_512 | Shape latent 512 | ~3GB | Same as TRELLIS |
| slat_flow_1024 | Shape latent 1024 | ~4GB | Same as TRELLIS |
| shape_decoder | Latent → mesh | ~1GB | Same as TRELLIS |
| tex_slat_flow_512 | Texture latent 512 | ~3GB | Pixal3D-specific |
| tex_slat_flow_1024 | Texture latent 1024 | ~4GB | Pixal3D-specific |
| tex_decoder | Latent → texture | ~1GB | Pixal3D-specific |
| image_cond_ss | DiNOv3 projection (16 grid) | ~1GB | Pixel-aligned, 16 grid |
| image_cond_shape_512 | DiNOv3 projection (32 grid) | ~1GB | Pixel-aligned, 32 grid |
| image_cond_shape_1024 | DiNOv3 projection (64 grid) | ~1GB | Pixel-aligned, 64 grid |
| image_cond_tex_1024 | DiNOv3 projection (64 grid) | ~1GB | Pixel-aligned, 64 grid |
| rembg | BiRefNet background removal | ~300MB | FP32, shared with TRELLIS |

## Key Differences from TRELLIS

- **4 separate DiNOv3 conditioners** (vs 1 in TRELLIS) — pixel-aligned at different resolutions
- **2 texture flow models** (vs 1 in TRELLIS) — cascade texture generation
- **Projection-mode conditioning** — pixel features explicitly back-projected into 3D
- **NAF upsamplers** — Neural Auxiliary Fields for conditioning upsampling
- **o_voxel export** — higher quality GLB export with PBR materials
- **13 components** (vs 8 in TRELLIS) — heavier overall

## Relationship to TRELLIS

Pixal3D is a TRELLIS.2 fine-tune. It shares:
- ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024, shape_decoder (same weights)
- BiRefNet rembg (same)
- Sparse tensor infrastructure (same)

It adds:
- 4 pixel-aligned DiNOv3 conditioners (replacing single TRELLIS image_cond)
- tex_slat_flow_512 (additional texture resolution stage)
- NAF upsamplers for conditioning
- o_voxel for higher quality export
