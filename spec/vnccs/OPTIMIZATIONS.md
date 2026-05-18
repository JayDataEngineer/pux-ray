# VNCCS — Optimizations

## Integration Level: WORKFLOW (no new model handler needed)

VNCCS is NOT a model that needs a Wan2GP handler. It is an orchestration
pattern on top of existing Wan2GP models (QWEN-Image-Edit, SDXL/Z-Image).
All GPU operations already have Wan2GP handlers.

## What's Available

| Component | Wan2GP Model | Integration | Status |
|-----------|-------------|-------------|--------|
| SDXL base | `z_image` (or SDXL vendor handler) | Built-in | Available now |
| QWEN-Image-Edit | `qwen-image-edit` (vendor handler) | Built-in | Available now |
| CLIP text encoder | Part of QWEN/SDXL pipe dict | Built-in | Available now |
| VAE | Part of QWEN/SDXL pipe dict | Built-in | Available now |
| Task LoRAs | `loras_selected` parameter | Built-in | Available now |
| BodyMeshRenderer | CPU pyrender, not in Wan2GP | **Not available** | Needs standalone |
| OpenPose | ControlNet aux, not in Wan2GP | **Not available** | Needs standalone |

## What We Gain vs ComfyUI

| Aspect | ComfyUI (current) | Wan2GP Workflow (new) |
|--------|------------------|----------------------|
| Load time | ~5min cold start (entire ComfyUI) | ~10-30s per model (mmgp streaming) |
| VRAM | Full model in VRAM | mmgp module-level streaming |
| Batch | Loop over ComfyUI API | Loop over `svc.infer()` (same latency) |
| Warm reuse | ComfyUI keeps models hot | Wan2GPService singleton keeps hot |
| Cold start | Must boot ComfyUI subprocess | No subprocess — direct import |

## Optimization Gaps

### 1. Task LoRA Caching (MEDIUM)
Each VNCCS step loads 1-2 LoRAs. If the same LoRA is used across multiple
calls (e.g., EmotionCore for all emotion variations), Wan2GP reloads it
per call. A LoRA cache could skip redundant loading.

### 2. BodyMeshRenderer Port (LOW)
The 3D mesh renderer is a CPU pyrender script. It's small and standalone.
Could be packaged as a utility function in `services/workflows/` directly
— no Wan2GP integration needed.

### 3. VNCCS_QWEN_Encoder Replication (DEFERRED)
The encoder does reference latent injection — VAE-encode images and inject
at timestep zero. This IS unique to VNCCS and is NOT in Wan2GP. However,
replicating it outside ComfyUI requires understanding the exact QWEN-Edit
model architecture and injection mechanics. This is the hardest part.

Current assessment: the `image_b64` parameter + instruction prompt
("Draw character from image2") approximates the encoder's behavior
for most use cases. Full parity requires deeper work.

## Priority Upgrade Path

1. **Publish Wan2GPService-based workflow functions** — done (services/workflows/)
2. **Add routes** — done (gateway/routes/workflows.py)
3. **Port BodyMeshRenderer** — standalone, trivial
4. **Benchmark latency vs ComfyUI** — verify mmgp advantage
5. **VNCCS_QWEN_Encoder investigation** — only if quality gap matters
