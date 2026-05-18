# VNCCS — Original Workflow

**Source**: https://github.com/AHEKOT/ComfyUI_VNCCS
**Type**: Character creation pipeline (ComfyUI custom nodes on QWEN-Image-Edit-2511)
**Architecture**: Multi-stage pipeline — 22 custom nodes orchestrate QWEN + SDXL + LoRAs

## What This Is

VNCCS is NOT a single model. It is a **pipeline specification** consisting of:
- **Backbone model**: QWEN-Image-Edit-2511 (image-conditioned diffusion)
- **Supporting models**: SDXL/Illustrious base, Seed VR, CLIP, VAEs
- **Task LoRAs**: EmotionCore, PoseStudio, ClothesHelper, TransferClothes, poser_helper
- **Orchestration nodes**: VNCCS_Pipe (data bus), CharacterCreator, EmotionGenerator, SpriteGenerator
- **Utility nodes**: VNCCS_QWEN_Encoder, VNCCS_RMBG2, BodyMeshRenderer, OpenPose

## Inference Pipeline (5 Stages)

### Stage 1: Character Sheet Generation
```
text prompt → CharacterCreator (attributes)
  → SDXL/Illustrious base (full body, green screen background)
  → QWEN-Image-Edit refinement (face closeup + body details)
  → VNCCS_RMBG2 (background removal)
  → character sheet image + face closeups
```

Models involved: SDXL base + QWEN-Image-Edit + CLIP + VAE + poser_helper LoRA

### Stage 1.1: Clone Existing Character
```
existing character image + CharacterCreator attributes
  → QWEN-Image-Edit (re-render with new attributes)
  → two output paths: refined body + refined faces
```

Same models as Stage 1. No SDXL pass — pure QWEN edit.

### Stage 2: Clothes Generation
```
character sheet → VNCCSSheetExtractor (row extraction)
  → CharacterAssetSelectorQWEN (clothing description)
  → QWEN-Image-Edit + ClothesHelper + TransferClothes LoRAs
  → VNCCS_RMBG2 + sheet composition
```

Models involved: QWEN-Image-Edit + 2 clothes LoRAs + Seed VR + VAE

### Stage 3: Emotion Generation
```
character sheet → EmotionGeneratorV2 (emotion + costume list)
  → QWEN-Image-Edit + EmotionCore LoRA
  → one call per emotion-costume pair
  → face image + transparent sprite
```

Models involved: QWEN-Image-Edit + EmotionCoreV1 LoRA

### Stage 4: Sprite/Animation
```
character sheet → SpriteGenerator (scans Sheets/ directory)
  → CharacterSheetCropper (contour detection)
  → individual cropped sprites → SaveImage
```

Post-processing only. No GPU models.

### Stage 5: LoRA Dataset
```
character images → DatasetGenerator
  → auto-captioned .txt files → LoRA training directory
```

Data preparation only. No GPU models.

## Key Observation

Stages 1-3 are the compute-intensive ones. Every GPU operation goes through
QWEN-Image-Edit-2511 or SDXL — both already present in Wan2GP's model catalog.

The VNCCS-specific value is NOT a model — it's:
- The orchestration sequence (which model to call when)
- The reference latent injection technique (QWEN_Encoder)
- The task LoRAs (EmotionCore, PoseStudio, etc.)
- The parameter mappings (emotion tags, pose JSON, character attributes)

## Flow Diagram

```
Stage 1: Text ──→ SDXL ──→ image ──→ QWEN-Edit ──→ sheet
Stage 2: sheet ──→ QWEN-Edit + clothes LoRAs ──→ clothed sheet
Stage 3: sheet + emotions ──→ QWEN-Edit + emotion LoRA ──→ emotion set
Stage 4: sheet ──→ crop ──→ sprites
Stage 5: sprites ──→ captions ──→ LoRA dataset
```

Only Stages 1-2 involve model switching (SDXL → QWEN). Stage 3 reuses QWEN.
