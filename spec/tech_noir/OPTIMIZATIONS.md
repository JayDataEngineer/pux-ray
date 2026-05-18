# Tech Noir Studio — Optimizations

## Integration Level: WORKFLOW (all models already in Wan2GP)

Every GPU stage in the Tech Noir pipeline uses a model that already has a
Wan2GP handler. No new handlers needed.

| Stage | Model | Wan2GP Status | Notes |
|-------|-------|---------------|-------|
| generate | Z-Image | Built-in | `svc.load("z_image")` |
| sheet | QWEN-Image-Edit | Built-in | `svc.load("qwen-image-edit")` |
| face_detailer | QWEN-Image-Edit | Built-in | Same model, different params |
| emotions | QWEN-Image-Edit | Built-in | + EmotionCore LoRA |
| sprites_static | Post-process | N/A | Sheet cropping |
| sprites_animated | BodyMesh + QWEN | Built-in + CPU utility |
| motion_npz | HY-Motion | Custom handler | `svc.load("hy-motion-1.0")` |
| outfit | QWEN-Image-Edit | Built-in | + clothes description prompt |
| state | QWEN-Image-Edit | Built-in | + condition description prompt |
| trellis | TRELLIS | Custom handler | `svc.load("trellis")` |
| video | LTX Video | Built-in | `svc.load("ltx2")` |
| lora | Post-process | N/A | Caption generation |

## What We Gain vs ComfyUI

| Aspect | ComfyUI (current) | Wan2GP Workflow (new) |
|--------|------------------|----------------------|
| Cold start | ~5min (ComfyUI boot) | ~10-30s per model |
| VRAM | Full model load | mmgp streaming |
| Per-frame sprite | ~15-20s per frame | ~5-10s via QWEN |
| Multi-stage pipeline | Serial node graph | Sequential Python calls |
| Model switching | New workflow submission | `svc.load()` with mmgp |

## Optimization Gaps

### 1. BodyMeshRenderer → QWEN quality (MEDIUM)
The composite image approach (mesh + character side-by-side) is a
simplification of the VNCCS 3-image reference latent injection. May
produce less accurate pose following. Full VNCCS_QWEN_Encoder parity
is the only true fix.

### 2. HY-Motion NPZ → per-frame mesh (LOW)
Currently we generate NPZ via HY-Motion, then need to extract
per-frame rotations for BodyMesh rendering. The NPZ → frame rotations
extraction utility (`hymotion_converter.py`) already exists in
tech-noir-studio/tools/comfyui_nodes/utils/ — needs extraction as
a standalone function.

### 3. LLM keyframe generation (DEFERRED)
The MotionDirector path (LLM generates keyframe rotations from motion
description) is not yet ported. Falls back to HY-Motion or manual poses.

### 4. Batch emotion variation (LOW)
Each emotion is a separate `svc.infer()` call. Wan2GP batch is
same-input-multi-seed only, so we loop. Could be optimized by
service-level parallelism (concurrent infer() calls for different
emotions).

## Priority Upgrade Path

1. **Published tech_noir workflow functions** — done (services/workflows/tech_noir.py)
2. **Registered as routes** — done
3. **Swap build system backend** — Replace `submit_workflow()` calls in
   stages_character/ with `workflows.tech_noir.generate()` calls
4. **Port HY-Motion converter** — Extract NPZ→frame utility
5. **VNCCS_QWEN_Encoder parity** — Only if quality gap matters
