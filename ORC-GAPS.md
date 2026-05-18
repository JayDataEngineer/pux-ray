# Orchestrator Gap Tracker

## How to Use This

Each gap is a tracked item with status, impact, and fix path. Update as
gaps are resolved. Add new gaps as they're discovered.

---

## GAP-001: VNCCS_QWEN_Encoder Reference Latent Injection

**Status**: 🟡 Workaround in place
**Impact**: HIGH — pose following quality is approximate
**Affects**: vnccs/sprite, vnccs/pose-edit, tech-noir/sprites-animated

### Problem
The VNCCS_QWEN_Encoder ComfyUI node injects VAE-encoded reference latents
at timestep zero with quadratic weighting. This is the core technique that
gives VNCCS its character consistency — it's NOT just prompt engineering.

Our approximation: composite the mesh pose image + character image
side-by-side and feed as a single `image_b64`. QWEN sees both references
in one frame with the prompt "Draw the character on the right in the pose
shown on the left."

### Fix Path
1. Study `opt/wan2gp/models/qwen/qwen_main.py` — understand how QWEN
   handles multiple reference images
2. The `input_ref_images` parameter passes through `_SAFE_PASSTHROUGH`
   and gets base64→PIL decoded in Wan2GPService.infer()
3. Determine if QWEN's `generate()` accepts a list of reference images
   and treats them as VNCCS-like reference latents
4. If yes: pass images as `reference_images: [mesh_b64, char_b64, skeleton_b64]`
5. If no: need to modify the QWEN handler to support reference latent injection

### Verification
- Compare output of ComfyUI VNCCS workflow vs workflow function
- Same prompt, same seed, same character — does the pose follow match?

---

## GAP-002: OpenPose Skeleton Extraction

**Status**: ✅ Resolved
**Impact**: Was MEDIUM — third conditioning image now available
**Affects**: vnccs/sprite, vnccs/pose-edit, tech-noir/sprites-animated

### Fix Applied
1. **Standalone DWPose utility** (`services/workflows/utils/dwpose.py`):
   - `Wholebody` class: YOLOX detection + RTMPose pose estimation via ONNXRuntime
   - `skeleton_from_image(image_np, width, height) -> np.ndarray` — skeleton overlay
   - `skeleton_from_image_b64(b64, width, height) -> str` — base64 in/out
   - `detect_poses(image_np) -> np.ndarray` — raw keypoints
   - Models auto-downloaded from HuggingFace `yzd-v/DWPose` on first use
     (yolox_l.onnx ~217MB, dw-ll_ucoco_384.onnx ~134MB)
   - COCO-18 skeleton rendering with colored bones (matching VNCCS style)

2. **Composition upgraded** (`services/workflows/vnccs.py`):
   - `_compose_images_side_by_side(*images_b64) -> str` — now accepts N images
   - `sprite()` and `pose_edit()` now use 3-image composite:
     mesh (pose guide) + character (identity) + skeleton (limb guide)
   - Instruction prompt updated to match VNCCS 3-image format

### 3-Image Chain
```
BodyMesh render (pose)          → mesh_b64
  → DWPose.skeleton_from_image  → skeleton_b64
  → compose side-by-side (mesh + character + skeleton)
  → QWEN infer with VNCCS_INSTRUCTION

VNCCS_INSTRUCTION:
  "Match the body pose shown in Picture 1 (3D body mesh).
   Picture 2 is the character to draw. Picture 3 shows the skeleton overlay.
   Replicate the exact pose, limb positions, and body orientation from Picture 1
   while maintaining the character's identity, clothing, and appearance."
```

### Dependencies
- `onnxruntime`, `cv2`, `numpy`, `huggingface_hub` (all already available)
- Models downloaded on-demand to `~/.cache/tech-noir/dwpose/`

---

## GAP-003: HY-Motion NPZ → Per-Frame Rotation Extraction

**Status**: ✅ Resolved
**Impact**: Was MEDIUM — now fully chainable
**Affects**: tech-noir/motion-npz, tech-noir/sprites-animated

### Fix Applied
1. **HY-Motion handler fixed** (`opt/wan2gp/models/hy_motion/hy_motion_handler.py`):
   `_Pipeline.generate()` now serializes `rot6d` tensor as NPZ and returns
   it as `data` (base64) in the response dict.

2. **Converter extracted** (`services/workflows/utils/motion.py`):
   - `extract_keyframes(path, num_keyframes)` — from NPZ file
   - `npz_bytes_to_keyframes(bytes, num_keyframes)` — from NPZ bytes
   - `npz_b64_to_keyframes(b64, num_keyframes)` — from base64 NPZ
   - Helper functions: rot6d_to_matrix, matrix_to_euler_xyz, convert_yup_to_zup

3. **tech_noir workflow functions updated**:
   - `motion_npz()` now returns `keyframes` in the response (pre-extracted)
   - `sprites_animated()` accepts either `poses` or `motion_npz` result containing
     `keyframes` — if the latter, uses them as per-frame pose rotations

### Full Chain
```
motion_npz(prompt="walking", duration=4.0, num_keyframes=6)
  → HY-Motion inference → rot6d tensor → NPZ bytes → base64
  → npz_b64_to_keyframes() → 6 rotation dicts
  → returned as response["keyframes"]

sprites_animated(character, motion_npz=response, directions=[...])
  → for each keyframe: render_pose(rotations) → mesh_b64
  → composite mesh + character → QWEN edit per frame
  → returns list of frame results
```

### Dependencies
- `numpy` only (motion converter has NO dependency on `anny` or `torch`)
- `anny` needed only if calling `render_pose()` on the keyframes

---

## GAP-004: LLM Keyframe Generation (MotionDirector Equivalent)

**Status**: 🔴 Not started
**Impact**: LOW — HY-Motion exists as alternative
**Affects**: tech-noir/sprites-animated (llm motion strategy)

### Problem
The MotionDirector ComfyUI node calls an LLM (OpenRouter API) to generate
keyframe rotation dicts from a motion description text. This path is used
when HY-Motion NPZ files aren't available.

### Fix Path
1. Create `services/workflows/utils/motion_llm.py`
2. Call an LLM (via Wan2GP LLM handler or direct API) with prompt:
   "Generate keyframe joint rotations for: {motion_description}"
3. Parse the JSON response into frame-by-frame rotation dicts
4. Pass to BodyMesh renderer

### Priority
Defer until HY-Motion NPZ extraction works. The HY-Motion path produces
better results.

---

## GAP-005: LoRA Cache

**Status**: 🔴 Not started
**Impact**: LOW — correctness not affected, only performance
**Affects**: All VNCCS workflows with loops (emotions, sprite)

### Problem
Each `svc.infer()` call loads LoRAs specified in `loras_selected`.
When looping over 10 emotions with the same EmotionCore LoRA, it's
reloaded each time. Wan2GP may cache internally already.

### Verification Needed
- Does Wan2GP skip LoRA loading if already in VRAM?
- If not: implement LRU LoRA cache in Wan2GPService or workflow base

---

## GAP-006: WDC Timeline Segmentation

**Status**: ✅ Resolved
**Impact**: Was LOW — now correctly generates per-segment videos
**Affects**: wdc/timeline

### Fix Applied
Investigation found that LTXDirector shot-planning lives entirely in the
ComfyUI custom node (`whatdreamscost-comfyui` / LTXDirector +
LTXDirectorGuide + LTXVCropGuides). The Wan2GP LTX handler has no
multi-shot or segment concept — `segments` was being silently ignored.

Fixed `timeline()` in `services/workflows/wdc.py`:
- Each segment is now an independent `svc.infer()` call
- Segments accept: `prompt`, `frames`, `first_frame_b64`, `last_frame_b64`
- Returns `segments` list with per-segment video bytes, prompt, frame count
- No frame-level temporal coherence between segments (no cross-fade/cut logic)

### Segment Schema
```json
{
  "prompt": "wide shot of character walking",
  "frames": 97,
  "first_frame_b64": "...",   // optional
  "last_frame_b64": "..."     // optional
}
```

### Confirmed Working
- `image_start`/`image_end` pass-through verified via `_build_generate_kwargs`
  and `_SAFE_PASSTHROUGH` / `image_end_b64` handling
- Last-frame conditioning verified: LTX2.generate() explicitly accepts
  `image_end=None` as a parameter

---

## GAP-007: FaceDetailer Equivalent

**Status**: 🟡 Workaround exists
**Impact**: LOW — QWEN detailer is a reasonable approximation
**Affects**: tech-noir/face-detailer

### Problem
The FaceDetailer ComfyUI node (from Impact Pack) does face detection →
bbox expansion → SAM masking → inpainting. Our `face_detailer()` workflow
just calls QWEN-Edit with "improve face details" — no bbox guidance, no
SAM mask. QWEN may not focus on the face specifically.

### Fix Path
- If quality gap is visible: port face detection from `ultralytics` or
  `insightface`, then pass SAM-like region via inpainting parameters
- For now: the QWEN prompt approach works acceptably for most cases

---

## GAP-008: `image_end_b64` — WDC Last-Frame Conditioning

**Status**: 🟢 Resolved
**Impact**: HIGH — was blocking WDC FFLF last-frame feature
**Affects**: wdc/ltx-fflf-2stage, wdc/ltx-fflf-3stage

### Fix Applied
Added `image_end_b64` handling in Wan2GPService.infer() at
`services/wan2gp/deployment.py:489-494`:
```python
if payload.get("image_end_b64"):
    img = Image.open(io.BytesIO(base64.b64decode(payload["image_end_b64"])))
    kwargs["image_end"] = img
```
Same pattern as `image_b64` → `image_start`. `image_end` stays in
`_BLOCKED_KEYS` (blocked from passthrough) but is set manually.

### Verification Needed
- Does the LTX Video handler's generate() actually use `image_end`?
- If not: this sets the kwarg but it's silently ignored

---

## GAP-009: BodyMeshRenderer Port

**Status**: ✅ Done
**Impact**: Was blocking ALL pose-driven workflows
**Affects**: vnccs/sprite, vnccs/pose-edit, tech-noir/sprites-animated

### Fix Applied
Complete port of BodyMeshRenderer from ComfyUI custom node to
`services/workflows/utils/body_mesh.py`. Includes:
- Anny forward pass (lazy-loaded model)
- Euler → rotation matrix
- Pyrender GPU renderer
- PIL CPU fallback renderer
- Y-axis rotation for multi-direction
- `render_pose()` and `render_pose_b64()` wrappers

### Dependencies
- `anny` (Naver Labs, Apache 2.0)
- `torch`, `numpy`
- `pyrender` + `trimesh` (optional — PIL fallback exists)

---

## GAP-010: BodyMesh → QWEN Side-by-Side Composite Quality

**Status**: 🟡 Acceptable
**Impact**: MEDIUM — pose accuracy may be lower than ComfyUI
**Affects**: vnccs/sprite, vnccs/pose-edit, tech-noir/sprites-animated

### Problem
Real VNCCS uses 3 separate reference latents with timestep-zero injection.
Our composite image approach puts mesh + character in one frame and relies
on QWEN to understand "draw the character on the right in the pose on the
left." This is less precise.

### Workaround
The instruction prompt is critical. Current best result:
"Draw the character on the right in the pose shown on the left"

### Potential Improvement
If QWEN-Edit's generate() supports `reference_images` as a list of PIL
images, we can pass separate mesh + character + skeleton images instead
of compositing. Check the QWEN handler's generate() signature and
`input_ref_images` parameter.

---

## GAP-011: Error Handling — Wan2GPService Cold Start

**Status**: 🟡 Acceptable
**Impact**: LOW — first request is slow (expected)
**Affects**: All workflows

### Problem
First call to any workflow triggers `Wan2GPService.__init__()` which runs
`discover_models()` — importing ALL Wan2GP handler modules. This is slow
(~5-15s) and happens on first request.

### Current Behavior
Singleton pattern means it happens once per process lifetime. Subsequent
requests skip discovery. Same pattern as Forge's lazy service loading.

### Possible Fix
Warm up on gateway startup via lazy import in `base.py`:

```python
def get_service():
    global _svc
    if _svc is None:
        import threading
        _svc = Wan2GPService()  # happens once
    return _svc
```

---

## GAP-012: Unreleased Models After Workflow

**Status**: ✅ Designed but not explicitly tested
**Impact**: MEDIUM — VRAM leak if models not released
**Affects**: All workflows

### Current Behavior
After a workflow function returns, the Wan2GPService singleton keeps the
last-loaded model in VRAM. This is by design (hot model for next request).
But if the workflow loaded model A then model B, model A's VRAM was freed
by `svc.load("B")` → `self.unload()` internally.

### Cleanup
- `reset_service()` in base.py forces full unload
- Gateway has `/admin/unload` endpoint via Forge
- Workflows don't explicitly release — hot model is the desired state
