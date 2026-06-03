# Tech Noir

Home AI infrastructure — Ray Serve on consumer GPUs, orchestrated via Kubernetes.

Generates images, video, audio, 3D meshes, motion, and text through a unified API. A Wan2GP extension handles GPU model management with automatic VRAM offloading. MCP servers expose tools for agents and chat UIs. DAG pipelines compose multi-step workflows (character creation, video editing, pose transfer).

## What it does

| Capability | Models | Service |
|---|---|---|
| Image generation | Flux, ZImage, Wan2.1 | wan2gp |
| Image editing | QWEN Image Edit | wan2gp |
| Video generation | Wan2.1, CogVideoX, LTX-2 | wan2gp |
| Text-to-speech | Kokoro, Qwen3-TTS, IndexTTS, eSpeak | standalone + wan2gp |
| Speech-to-text | Faster-Whisper, VibeVoice | standalone |
| 3D mesh generation | TRELLIS, AniGen | wan2gp |
| Motion generation | HY-Motion, Kimodo | wan2gp |
| Music / sound effects | ACE-Step, MOSS | wan2gp |
| LLM inference | llama.cpp (GGUF) | forge subprocess |
| Vision analysis | YOLOv8, Florence-2, SAM2 | media-analysis MCP |
| Web research | SearXNG + crawl4ai | web-research MCP |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Traefik :30080                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ API      │  │ Studio   │  │ Editor             │  │
│  │ ingress  │  │ UI       │  │ (React SPA)        │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │                  │             │
│  ┌────▼──────────────▼──────────────────▼──────────┐ │
│  │  Ray Serve (KubeRay)                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │ │
│  │  │ Forge   │  │ CPU     │  │ Workflow Engine  │  │ │
│  │  │ (GPU)   │  │ Services│  │ (DAG pipelines)  │  │ │
│  │  └────┬────┘  └─────────┘  └─────────────────┘  │ │
│  │       │  VRAM-aware scheduling                    │ │
│  │  ┌────▼──────────────────────────────────────┐   │ │
│  │  │  Wan2GP (mmgp offloading, 12+ families)   │   │ │
│  │  └───────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  MCP Servers (standalone K8s deployments)         │ │
│  │  media-analysis · web-research · wan2gp-studio    │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Hardware:** RTX 4090 (24GB VRAM) + 64GB RAM, NVMe storage. Accessible over Tailscale.

**GPU scheduling:** The Forge claims the entire GPU and tracks VRAM per model. Multiple models coexist when they fit; the largest is evicted when a new one needs space. Wan2GP uses mmgp for module-level CPU/GPU swapping — no full model reload needed.

## Quick start

```bash
# Prerequisites: k3s, Docker, task (https://taskfile.dev)
# First-time setup
task setup

# Daily use
task boot        # Verify cluster, start services
task status      # Show all services

# Generate an image
curl -X POST http://localhost:30080/forge \
  -H "Content-Type: application/json" \
  -d '{"service":"wan2gp","model":"flux_schnell","prompt":"a cat riding a skateboard"}'

# Text-to-speech
curl -X POST http://localhost:30080/tts/kokoro/speak \
  -d '{"text":"Hello world","voice":"af_bella"}'
```

## Project structure

```
gateway/             API ingress (Starlette) + MCP app host
services/
  forge.py           VRAM-aware GPU manager (Ray Serve deployment)
  wan2gp/            Wan2GP service — unified model pool
  tts/               Kokoro, eSpeak, Qwen3-TTS, IndexTTS
  asr/               Faster-Whisper
  llm/               llama.cpp subprocess wrapper
  image/             ComfyUI subprocess proxy
  motion/            Kimodo 3D motion
  workflows/         DAG pipeline runner + VNCCS/TechNoir/WDC workflows
mcp/
  wan2gp-studio/     MCP server — Forge + Workflow tools (FastMCP)
  media-analysis/    Vision tools (YOLOv8, Florence-2, SAM2)
  web-research/      Search, scrape, extract tools
web/
  editor/            Video editor frontend (React + Vite)
config/
  workflows/         YAML workflow specs (18 pipelines)
  model_registry.yaml  Model download URLs and metadata
infra/
  k8s/               RayService YAML, MCP manifests, monitoring
  docker/            Dockerfiles (gpu-all, CPU services)
  skypilot/          Cloud burst config (SkyServe)
```

## Key APIs

All endpoints proxied through Traefik at port 30080.

| Route | Description |
|---|---|
| `POST /forge` | GPU inference — `{"service":"wan2gp","model":"...","prompt":"..."}` |
| `POST /v1/run` | DAG pipeline runner — `{"pipeline":"vnccs/char-sheet","params":{...}}` |
| `POST /tts/<engine>/speak` | Text-to-speech |
| `POST /asr/whisper/transcribe` | Speech-to-text |
| `POST /llm/v1/chat/completions` | LLM chat (OpenAI-compatible) |
| `GET /mcp/media/*` | Vision analysis MCP |
| `GET /mcp/web/*` | Web research MCP |
| `GET /wf/*` | Workflow Engine (list specs, start runs, step execution) |

Auth: `X-API-Key` header or `?api_key=` query param.

## VNCCS pipelines

Character creation workflows matching the original VNCCS ComfyUI nodes:

| Pipeline | Description |
|---|---|
| `vnccs/char-sheet` | Attributes → character sheet (PoseGen → SD → QWEN refine → face detailer) |
| `vnccs/pose-edit` | Re-pose character from reference image or joint rotations |
| `vnccs/clone` | Clone character with modified attributes |
| `vnccs/emotions` | Generate emotion variations |
| `vnccs/sprite` | Generate animation sprite frames |

## Build & deploy

```bash
# Build GPU image and push to Forge Registry
bash infra/k8s/build_and_import.sh

# Deploy/update Ray Service
kubectl apply -f infra/k8s/ray-service.yaml

# Build and deploy MCP servers
bash infra/k8s/build_mcp.sh

# Deploy monitoring stack
task infra:monitor
```

**GPU image:** Multi-stage Dockerfile based on Wan2GP (CUDA 12.8, PyTorch 2.10, Python 3.10). Does not reinstall torch/torchaudio/flash-attn.

**Source mount:** Code is mounted via hostPath into the GPU pod — edits take effect immediately, but Python module changes need a Ray Serve redeploy (`serve.delete('forge')` then redeploy).

## Testing

```bash
task test                                        # Unit tests
python tests/test_vnccs_e2e.py char_sheet        # E2E char_sheet pipeline
python tests/test_vnccs_e2e.py pose_edit         # E2E pose_edit pipeline
python scripts/test_services_v2.py               # Integration tests
```

## Documentation

| File | Description |
|---|---|
| `CLAUDE.md` | Full system reference (services, architecture, conventions) |
| `AGENTS.md` | AI agent context for coding assistants |
| `ENDPOINTS.md` | API endpoint reference |
| `DEVELOPMENT.md` | Development workflow on constrained hardware |
| `WORKFLOW.md` | Flux GitOps workflow |
| `docs/archive/` | Historical gap analysis docs |

## License

Private repository.
