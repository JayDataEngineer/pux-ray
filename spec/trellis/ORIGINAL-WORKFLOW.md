# TRELLIS.2 — Original Workflow

**Source**: https://github.com/microsoft/TRELLIS
**Type**: Image-to-3D mesh generation
**Architecture**: Multi-stage pipeline — 8 nn.Modules run in sequence

## Inference Pipeline (6 Stages)

### Stage 1: Image Preprocessing
```
image → rembg (BiRefNet, if no alpha) → center crop → resize to ≤1024px → RGB
```

### Stage 2: Conditioning
```
image → image_cond model (DiNOv2-based) → condition vectors (512 + 1024 resolution)
```

### Stage 3: Sparse Structure
```
random noise [1, C, R, R, R] → ss_flow_model (N denoising steps) → z_s
z_s → ss_decoder (threshold at 0) → binary occupancy grid
→ argwhere → sparse voxel coordinates
```

### Stage 4: Shape SLat (with optional cascade)
```
# 512 resolution:
sparse_coords → slat_flow_512 (N steps) → shape latents at 512

# OR cascade 512 → 1024:
shape_slat_512 → shape_decoder.upsample(4x) → high-res coords
  → prune to max_tokens → slat_flow_1024 (N steps) → shape latents at 1024
```

### Stage 5: Texture SLat
```
shape_slat → normalize → concat with random noise
  → tex_slat_flow_1024 (N steps) → texture latents
```

### Stage 6: Decode
```
# Shape decode (GPU):
shape_slat → shape_decoder.convert_to_fp16() → mesh (vertices + faces + sub-meshes)

# Texture decode (GPU, shape intermediates on CPU):
tex_slat.to('cpu'), shape sub-meshes.to('cpu')  # free VRAM
tex_slat.to('cuda') → tex_decoder → PBR texture voxels

# Combine:
mesh + texture_voxels → MeshWithVoxel → trimesh → GLB bytes
```

## Components (8 nn.Modules)

| Module | Role | Size | Notes |
|--------|------|------|-------|
| `ss_flow_model` | Sparse structure denoising | ~2GB | 3D diffusion on occupancy grid |
| `ss_decoder` | Binary voxel decoding | ~500MB | Continuous → binary threshold |
| `slat_flow_512` | Shape latent denoising (512 res) | ~3GB | Flow matching on sparse coords |
| `slat_flow_1024` | Shape latent denoising (1024 res) | ~4GB | Higher resolution cascade |
| `tex_slat_flow_1024` | Texture latent denoising | ~4GB | Conditioned on shape latents |
| `shape_decoder` | Latent → mesh | ~1GB | Uses spconv (sparse convolution) |
| `tex_decoder` | Latent → texture | ~1GB | Uses spconv + PBR layout |
| `image_cond` | Image feature extraction | ~1GB | DiNOv2-based vision encoder |

Additional non-mmgp: `rembg` (BiRefNet) — background removal, stays FP32, excluded from mmgp pipe.

## Key Characteristics

- **All stages are strictly sequential** — no parallelism between stages
- **Intermediate tensors are large** — sparse voxel grids, latent volumes
- **Requires handler-level VRAM management** — spatial cache clearing, intermediate CPU offloading between stages
- **spconv dependency** — sparse convolution for shape/texture decoding (requires CUDA build)
- **Peak VRAM ~14GB** with staging, fits in 24GB
