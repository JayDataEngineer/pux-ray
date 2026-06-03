# Forge-Wan2GP Gaps

## Current State (2026-05-16)

### Model Test Results

| Model | Status | Load | Generate | Output | Notes |
|-------|--------|------|----------|--------|-------|
| eSpeak TTS | PASS | 0.0s | 0.0s | 136 KB WAV | CPU, subprocess |
| Faster-Whisper ASR | PASS | 2.6s | 2.9s | correct transcription | CPU, CTranslate2 int8 |
| Kokoro TTS | PASS | 0.6s | 0.6s | 159 KB WAV | CPU, 5 nn.Modules |
| MOSS SoundEffect | PASS | 4.8s | 17.5s | 945 KB WAV | GPU, 16.8GB peak VRAM |
| TRELLIS 3D | FIX APPLIED | — | — | — | Removed .to(dev), mmgp manages placement. Pending Docker rebuild. |
| Pixal3D | FIX APPLIED | — | — | — | Same fix as TRELLIS. Pending Docker rebuild. |

### Fixed This Session

1. **`_shared.py` Path("") bug** — `resolve_model_path()` returned `Path(".")` (current dir) when model_def was empty, because `Path("").is_dir()` is True. Fixed by checking `if raw:` before testing `is_dir()`.

2. **MOSS transformers shims** — Vendor code imports `from transformers import initialization` (removed in 4.57). Fixed by `sys.modules.setdefault("transformers.initialization", torch.nn.init)` in `_load_delay_modules()`. Also shims `MODALITY_TO_BASE_CLASS_MAPPING` and `PreTrainedConfig`.

3. **MOSS VRAM** — 8B model + audio tokenizer exceeded 24GB. Fixed by keeping audio tokenizer on CPU (saves ~2GB). Peak VRAM now 16.8GB.

4. **HF_TOKEN** — Added `HF_TOKEN` from `shared-infra` secret to Ray head and worker pods in `ray-service.yaml`. Added to `secrets.env.example`.

5. **Pipeline execution endpoint** — `POST /api/pipelines/execute` with DAG spec, output chaining (`{step.output.field}`), SSE streaming. Files: `gateway/pipeline.py`, `gateway/ingress.py`, `gateway/ingress_deployment.py`, `sdk/client.py`.

6. **spconv CUDA build** — Replaced PyPI CPU-only wheel with `cumm-cu126 + spconv-cu126` pre-built CUDA wheels (forward-compatible with CUDA 12.8 runtime). One-line install in Dockerfile.

7. **NATTEN CUDA build** — Added `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST` to natten build in `Dockerfile.gpu-all`. Pending image rebuild to verify.

8. **TRELLIS OOM on load** — Handler did `pipeline.to(dev)` loading ALL 8+ nn.Modules to GPU simultaneously before mmgp profiling. Exceeds 24GB VRAM. Fixed by removing `.to(dev)` — modules stay on CPU, mmgp swaps to GPU just-in-time during `forward()`.

9. **Pixal3D OOM on load** — Same root cause as TRELLIS. 13 nn.Modules all moved to GPU via `.to(dev)` before mmgp profiling. Fixed same way.

### Open Gaps

#### GAP-1: Docker image rebuild needed

spconv-cu126, NATTEN CUDA, TRELLIS fix, and Pixal3D fix all applied in code but need a Docker image rebuild to verify end-to-end.

**Status:** Code changes complete. Pending `docker build`.

#### GAP-2: Untested heavy GPU models

| Model | Handler | spconv | NATTEN | .to(dev) fix | Tested E2E |
|-------|---------|--------|--------|-------------|-----------|
| TRELLIS 3D | trellis_handler | cu126 OK | N/A | Fixed | No |
| Pixal3D | pixal3d_handler | cu126 OK | 0.21.0 OK | Fixed | No |
| MOSS-TTS | moss_handler | N/A | N/A | N/A | No |
| MOSS-TTSD | moss_handler | N/A | N/A | N/A | No |
| MOSS-VoiceGen | moss_handler | N/A | N/A | N/A | No |
| ACE-Step | ace_step_handler | N/A | N/A | N/A | No |
| HY-Motion | hy_motion_handler | N/A | N/A | N/A | No |
| AniGen | anigen_handler | N/A | N/A | Safe (CPU load) | No |
| See-Through | see_through_handler | N/A | N/A | Safe (dtype only) | No |

**Status:** All handlers reviewed. TRELLIS + Pixal3D fixes applied. Need Docker rebuild + GPU test.

#### GAP-3: Pipeline composing API (DONE)

Backend has `POST /api/pipelines/execute` endpoint that accepts a DAG spec and runs steps via the existing orchestrator with SSE streaming. Frontend can build visual node editors that produce the same JSON/YAML.

### Architecture

```
Ray Cluster (backend)
├── Pipeline execution — receive DAG, run steps, chain outputs
├── 20+ family_handlers — each wraps a model variant
├── Forge — VRAM-aware GPU manager
└── Single GPU image (tech-noir/gpu-all:latest, 44GB)

Frontend (tech-noir-studio)
├── Pipeline composition — visual node editor produces JSON DAG
├── Service playground — forms for each model
└── Result viewer — audio, image, 3D previews
```
