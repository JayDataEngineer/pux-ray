# Tech Noir Studio — Original Workflow

**Source**: https://github.com/tech-noir-studio (internal monorepo)
**Type**: Multi-stage asset generation pipeline for 2D game sprite production
**Architecture**: YAML manifest → DAG execution → ComfyUI per stage

## What This Is

The Tech Noir Studio build system (`departments/art/`) is a declarative
pipeline that reads character YAML manifests and produces sprite sheets,
portraits, 3D models, and animation frames through a series of ComfyUI
workflow executions.

## Pipeline Stages

Each stage submits a ComfyUI workflow JSON and downloads the output:

### Stage 1: Generate
```
text prompt → Z-Image (SD-based) → character image
```
Model: Z-Image (Wan2GP built-in)
Workflow: `00_z_image_character.json`
Parameters: prompt, seed, steps, cfg, width, height

### Stage 2: Sheet (Clone)
```
character image + attributes → QWEN-Image-Edit → re-rendered character
```
Model: QWEN-Image-Edit (Wan2GP built-in via VNCCS encoder)
Workflow: `vnccs_step11_clone_existing.json`
Parameters: image, character_name, attribute overrides
Plus: optional FaceDetailer inline workflow for face refinement

### Stage 3: Emotions
```
character sheet + emotion list → QWEN-Image-Edit + EmotionCore LoRA
  → emotion portraits (one per emotion-costume pair)
```
Model: QWEN-Image-Edit
Workflow: `vnccs_step3_emotions.json`
Parameters: character, emotions_data, costumes_data

### Stage 4: Sprites (Static)
```
character sheet → VNCCS SpriteCreator → cropped individual sprites
```
Workflow: `vnccs_step4_sprites.json`
Post-processing only (SpriteGenerator + CharacterSheetCropper)

### Stage 5: Motion (HY-Motion)
```
text description → HY-Motion → NPZ motion keyframes
```
Model: HY-Motion (Wan2GP custom handler)
Workflow: `01_hymotion_keyframes.json`
Parameters: prompt, seed

### Stage 6: Sprites (Animated)
```
NPZ keyframes → BodyMeshRenderer → mesh image per frame
  → QWEN-Image-Edit + character + skeleton → posed sprite frame
```
Models: BodyMeshRenderer (CPU) + QWEN-Image-Edit
Workflow: `05_sprite_bodymesh_edit.json` per frame
Parameters: rotations_json, model_rotation_y, character image, instruction

### Stage 7: Outfit
```
character sheet + outfit description → QWEN-Image-Edit + clothes LoRAs
  → re-clothed character
```
Model: QWEN-Image-Edit
Workflow: `vnccs_step2_clothes.json`
Parameters: character, outfit description

### Stage 8: State
```
character image + condition description → QWEN-Image-Edit
  → state-modified character (e.g., beatup, injured)
```
Model: QWEN-Image-Edit
Workflow: `vnccs_step11_clone_existing.json` (same as sheet)

### Stage 9: TRELLIS 3D
```
character image → TRELLIS → GLB 3D model
```
Model: TRELLIS (Wan2GP custom handler)
Endpoint: Direct HTTP POST (not ComfyUI)
Parameters: image, name

### Stage 10: Video
```
image → LTX Video → video clip
```
Model: LTX Video (Wan2GP built-in)
Workflow: `04_ltx_video_assembly.json`

### Stage 11: LoRA Dataset
```
character sprites + faces → captioned .txt files → LoRA training set
```
Post-processing only. No model call.

## Dependency Graph

```
generate (t0)
  ├── sheet (t1) ──┬── emotions (t1.5)
  │                 ├── sprites_static (t1) ──── outfit (t2)
  │                 ├── state (t1)
  │                 └── video (t1.5)
  ├── sprites_animated (t1.5) ──┬── motion_npz (t1)
  │                             ├── trellis (t2)
  │                             └── per-frame render (t1.5)
  └── trellis (t2) ──── sprites_animated (t1.5, trellis body)
```

## Key Characteristics

- Every GPU stage uses a model already available in Wan2GP
  (Z-Image, QWEN-Image-Edit, HY-Motion, TRELLIS, LTX Video)
- The VNCCS custom nodes (Encoder, SpriteCreator, EmotionStudio, etc.)
  implement orchestration logic that can be replicated as Python code
- BodyMeshRenderer is a CPU utility, not a GPU model
- Stages are independent and cacheable (build_state.json)
