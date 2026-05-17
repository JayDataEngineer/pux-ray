# LoRA Training: Making Klein 4B a Capable Image Editor

## The Goal

Train LoRAs for Flux 2 Klein 4B (Q8 GGUF, ~4 GB) to match VNCCS's Qwen Image Edit 20B quality for pose-guided character editing, at Klein's speed (4 steps) and size (1/5 the VRAM).

## The Architecture Gap

| | VNCCS (Qwen Image Edit) | Klein (Flux 2 Klein 4B) |
|---|---|---|
| Model size | 20B params | 4B params |
| Text encoder | Qwen2.5-VL 7B (vision-language) | Qwen3 4B (text-only) |
| Reference mechanism | VAE latents + vision tokens in prompt | VAE latent tokens only (`img_cond_seq`) |
| Inference steps | 20 (or 4 with Lightning LoRA) | 4 native |
| VRAM | ~24 GB | ~6-8 GB (Q8) |

Klein already has `encode_image_refs` — it VAE-encodes reference images and injects them as position-encoded tokens. It just wasn't trained *enough* on editing-style references. The LoRA biases existing attention pathways to attend more to reference content.

## What VNCCS Actually Did

### Key repos (AHEKOT / MiuProject):

| Repo | Purpose |
|---|---|
| `ComfyUI_VNCCS` | Core nodes (VNCCS_QWEN_Encoder, etc.) |
| `ComfyUI_VNCCS_Utils` | Utilities (PoseStudio, DatasetGenerator, etc.) |
| `ComfyUI-SAM3DBody_utills` | **Extract 3D body meshes from 2D images** |
| `Slimy_HMR2_keyPoint3D` | 3D pose estimation from 2D (HMR 2.0 based) |
| `QwenDatasetManager` | **Organizes instruction/result training pairs** for Qwen fine-tuning |

### Their data pipeline (the key insight):

They didn't render synthetic meshes from scratch. They extracted 3D from existing 2D:

```
Existing 2D anime image
  → SAM3DBody/HMR2 → 3D mesh + skeleton extracted from the 2D art
  → Render extracted mesh as flat image (image1 / pose reference)
  → Run OpenPose on render (image3 / skeleton)
  → Use original 2D image as character reference (image2)
  → Send triple (mesh, char, pose) to Qwen with edit instruction
  → Save (inputs + Qwen output + caption) as training pair
  → Train LoRA
```

This gave them massive diversity — every existing anime image becomes a training sample. No procedurally generated renders needed. Real art styles, real poses, real compositions.

### LoRA training details (inferred from artifacts):

| LoRA | Steps (from filename) | Size |
|---|---|---|
| `VNCCS_PoseStudioV5` | 12,800 | 1.2 GB |
| `ClothesHelperUltimateV1` | 5,100 | 1.0 GB |
| `EmotionCoreV1` | 3,000 | 1.0 GB |
| `EmotionCoreV2` | 4,700 | 1.0 GB |
| `TransferClothes` | 6,700 | 1.0 GB |
| `poser_helper_v2` | 4,200 | 1.0 GB |

- **Framework**: kohya_ss (step-number naming convention)
- **Training**: Applied to UNet only (`LoraLoaderModelOnly`)
- **Rank**: Unknown, but at 1.2 GB for 20B model, likely rank 128-256
- **Base model**: Qwen Image Edit 2511

### The VNCCS_QWEN_Encoder uses 3 image slots wired as:

| Workflow | image1 | image2 | image3 |
|---|---|---|---|
| Pose edit / sprite | Body mesh render | Character portrait | OpenPose of mesh |
| Keyframe edit | Pre-rendered mesh keyframe | Character portrait | OpenPose of keyframe |
| Clothes gen | Character sheet | Clothes reference | Same character sheet |
| Clone existing | Character front view | Different character view | Third view |
| Char sheet | Multiple character references | Another reference | Another reference |

## Training Strategy for Klein

### The teacher-student approach (distillation):

Use VNCCS Qwen 20B as teacher to generate training targets, train Klein 4B LoRA to match.

### Data generation pipeline:

```python
# For each existing character image:
1. Run SAM3DBody_utills → extract 3D mesh
2. Run HMR2 → extract skeleton keypoints
3. Render mesh as flat 2D image (pose reference)
4. Run OpenPose on render → skeleton overlay
5. Composite: (mesh, char_original, openpose) as 3-image input
6. Send to VNCCS Qwen with instruction → get target edit
7. Filter bad outputs with CLIP-score threshold
8. Save as training pair with caption
9. Repeat across diverse character images
```

### For diversity, vary:

- **Character art**: scrape/generate 500+ distinct anime characters (different artists, styles, coloring)
- **Poses**: extracted from existing art, not just MakeHuman rotations
- **OpenPose quality**: different detectors, resolutions, imperfect skeletons
- **Compositions**: full body, half body, close-up, angles

### Training config (kohya_ss):

| Parameter | Value |
|---|---|
| Base model | `flux-2-klein-4b-q8_0.gguf` |
| LoRA target | UNet only |
| Rank | 128 |
| Steps | 8K-12K |
| Batch | 1 (fits 24GB) × 4 grad accum |
| LR | 1e-4 |
| Time | ~8-12 hours on RTX 4090 |

Alternatively, $3-5 on RunPod/Vast A100 (~4 hours).

### What we need to build:

| Asset | How |
|---|---|
| 3D extraction from 2D | `ComfyUI-SAM3DBody_utills` + `Slimy_HMR2_keyPoint3D` nodes |
| Training pair organizer | `QwenDatasetManager` (or equivalent) |
| Teacher inference | VNCCS Qwen running locally on 4090 |
| LoRA trainer | kohya_ss (supports GGUF models now) |
| Klein inference pipeline | Existing 05_sprite_bodymesh_edit_klein.json + ReferenceLatent nodes |

### Quality validation:

1. **Every 500 steps**: Run fixed eval set (10 prompts × 10 image triples), compare CLIP score + LPIPS
2. **Final eval**: Side-by-side grid of "new LoRA vs no LoRA vs VNCCS result" — one screen, 2 minutes of human review
3. **Regression check**: Verify 10 text-to-image prompts (no references) don't degrade

### LoRAs to train (by priority):

| Priority | LoRA | Data | Est. steps |
|---|---|---|---|
| 1 | **Klein-PoseEdit** | (mesh, char, openpose) → edited character in pose | 12K |
| 2 | Klein-ClothesHelper | Same char, same clothes, different poses | 5K |
| 3 | Klein-EmotionCore | Same char, same pose, different expressions | 5K |
| 4 | Klein-TransferClothes | Same char, different outfits, same pose | 7K |

### What we DON'T need:
- No text encoder LoRA (Klein's Qwen3 is fine for text)
- No VAE LoRA (flux2-vae doesn't need tuning)
- No separate "vision encoder" LoRA (the PoseEdit LoRA inherently teaches reference attention through training on the 3-image pattern)

## Summary

VNCCS's advantage wasn't a magic dataset — it was a **3D extraction pipeline** that converted existing 2D anime art into structured training pairs (mesh + skeleton + original). This gave them diverse, real-art-quality data at scale without manual rendering.

For Klein, the approach is identical: extract 3D from 2D → pair with VNCCS Qwen teacher output → train LoRA. One 12-hour run on your 4090 or $5 on RunPod.
