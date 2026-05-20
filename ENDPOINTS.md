# API Endpoints Reference

All endpoints served through Traefik at port **30080** (`http://100.86.69.57:30080`).

Traefik routes Ray Serve, MCP servers, monitoring, and external services. The `forge-ingress` IngressRoute in `ai-services` namespace sends `PathPrefix('/')` → `tech-noir-ray-serve-svc:8000`.

Auth: `X-API-Key` header or `?api_key=` query param. Unset = no auth (dev mode).

---

## Ray Serve Deployments

Three Ray Serve applications handle routing. Ray matches most-specific `route_prefix` first:

| App | route_prefix | Description |
|-----|-------------|-------------|
| `forge` | `/forge` | VRAM-aware GPU scheduler |
| `playground` | `/playground` | Interactive service UI |
| `api-ingress` | `/` | Catch-all gateway (everything else) |

---

## Unified Interface

### POST /v1/run

Single endpoint for all GPU and pipeline operations. Three payload shapes:

**Single service invocation:**
```json
{
  "service": "trellis",
  "model": "trellis",
  "image": "<base64>"
}
```

**Named pipeline (registered workflow function):**
```json
{
  "pipeline": "tech-noir/generate",
  "params": {"prompt": "a warrior", "seed": 42}
}
```

**Inline pipeline steps (DAG with output chaining):**
```json
{
  "steps": [
    {"name": "gen", "service": "wan2gp", "model": "z_image", "params": {"prompt": "a cat"}},
    {"name": "3d", "service": "trellis", "depends_on": ["gen"], "params": {"image": "{gen.image}"}}
  ]
}
```

### GET /v1/run/catalog

Discover available pipelines and services. Returns:
```json
{
  "pipelines": [{"id": "tech-noir/generate", "description": "..."}],
  "services": [{"name": "trellis", "label": "TRELLIS.2", "category": "3D"}]
}
```

---

## OpenAI-Compatible Endpoints

### GET /v1/models

List all models across all services. Optional `?category=image` filter.

Response: `{"object": "list", "data": [{"id": "...", "category": "...", "output_type": "..."}]}`

### POST /v1/chat/completions

OpenAI-compatible chat. Routes through Forge to LLM service.

```json
{
  "model": "qwen3.6-27b-ud-q4_k_xl",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### POST /v1/audio/speech

Text-to-speech. Routes to appropriate TTS service based on `model` field.

```json
{
  "model": "tts-01-kokoro",
  "input": "Hello world",
  "voice": "af_bella"
}
```

### POST /v1/audio/transcriptions

Speech-to-text (multipart form data).

```
audio: <file>
model: whisper-1
language: en
```

### POST /v1/llm/configure

Set LLM model, engine, startup flags, session defaults.

```json
{
  "model": "qwen3.6-27b-ud-q4_k_xl",
  "gpu_layers": 99,
  "context_size": 8192
}
```

---

## TNAP Service Endpoints

### POST /v1/{service}/generate

Generic service invocation. All registered services are accessible here. Examples:

- `POST /v1/kokoro/generate` — Kokoro TTS
- `POST /v1/espeak/generate` — eSpeak TTS
- `POST /v1/faster_whisper/generate` — Faster-Whisper ASR
- `POST /v1/index_tts/generate` — IndexTTS v2
- `POST /v1/faster_qwen3_tts/generate` — Qwen3-TTS (CUDA graphs)
- `POST /v1/trellis/generate` — TRELLIS 3D
- `POST /v1/ace_step/generate` — ACE-Step music
- `POST /v1/moss_soundeffect/generate` — MOSS sound effects
- `POST /v1/anigen/generate` — AniGen 3D
- `POST /v1/see_through/generate` — See-Through layer decomposition
- `POST /v1/hy_motion/generate` — HY-Motion 3D motion

GPU services route through the Forge (VRAM-tracked). CPU services route directly to their Ray deployments.

---

## Service Discovery

### GET /v1/services

List all registered services with metadata.

### GET /v1/services/{service}

Info about a specific service (default model, aliases, description).

---

## Forge (VRAM-Aware GPU)

### POST /forge

GPU service invocation with VRAM tracking. Auto-loads and evicts as needed.

```json
{"service": "trellis", "image": "<base64>", "seed": 1}
```

**Actions** (also via `action` field):

| Action | Body | Description |
|--------|------|-------------|
| (default) | `{"service": "name", ...}` | Invoke service (auto-load if needed) |
| `preload` | `{"action": "preload", "service": "name", "model": "..."}` | Pre-load a service without inference |
| `release` | `{"action": "release", "service": "name"}` | Unload a specific service |
| `release` | `{"action": "release"}` | Unload all services |
| `status` | `{"action": "status"}` | VRAM allocations + GPU info |

### GET /forge

Returns current Forge status (loaded services, VRAM allocations, GPU info).

**Available Forge services:** `wan2gp`, `comfyui`, `llm`, `avatar`, `kimodo_demo`, `trellis`, `ace_step`, `hy_motion`, `moss_soundeffect`, `anigen`, `see_through`, `index_tts`, `faster_qwen3_tts`, `vibevoice_microsoft`, `vibevoice_community_tts`, `phi4mm`

---

## Workflows (Multi-Model Orchestration)

### GET /v1/workflows

List all registered workflow functions.

### GET /v1/workflows/{workflow_id}

Get metadata for a specific workflow.

### POST /v1/workflows/{workflow_id}

Execute a workflow through the Forge's VRAM ledger.

```json
{"prompt": "a warrior character", "seed": 42}
```

**Registered workflows:**

| ID | Description |
|----|-------------|
| `vnccs/char-sheet` | Text → character base sheet (SD + QWEN refine) |
| `vnccs/emotions` | Character sheet → emotion variation set |
| `vnccs/sprite` | Character sheet + poses → animation frames |
| `vnccs/pose-edit` | Character image + pose → posed character |
| `vnccs/clone` | Reference character → cloned variant |
| `vnccs/detailer` | Face/hand region refinement |
| `wdc/ltx-fflf-2stage` | Image-to-video with 2-stage FFLF |
| `wdc/ltx-fflf-3stage` | Image-to-video with 3-stage FFLF + upscale |
| `wdc/ltx-audio` | Image-to-video with audio conditioning |
| `wdc/timeline` | Multi-shot timeline video |
| `tech-noir/generate` | Z-Image character generation |
| `tech-noir/sheet` | Clone/re-edit character sheet |
| `tech-noir/face-detailer` | Face refinement via QWEN |
| `tech-noir/emotions` | Emotion variation set |
| `tech-noir/sprites-static` | Sprite extraction from sheet |
| `tech-noir/sprites-animated` | Animated sprite frames |
| `tech-noir/motion-npz` | HY-Motion motion generation |
| `tech-noir/outfit` | Outfit variant via QWEN |
| `tech-noir/state` | Condition state variant |
| `tech-noir/trellis` | TRELLIS 3D model generation |
| `tech-noir/video` | LTX Video assembly |
| `tech-noir/lora-dataset` | LoRA dataset preparation |

---

## Pipeline Executor (DAG with SSE Streaming)

### POST /api/pipelines/execute

Execute a multi-step inference pipeline with output chaining between steps.

```json
{
  "steps": [
    {"name": "gen", "service": "wan2gp", "params": {"prompt": "a cat"}},
    {"name": "upscale", "service": "trellis", "depends_on": ["gen"], "params": {"image": "{gen.image}"}}
  ]
}
```

Returns Server-Sent Events (SSE) stream with events: `pipeline_started`, `step_started`, `step_completed`, `step_error`, `pipeline_completed`, `pipeline_error`.

Step `params` support `{step_name.field}` references that resolve to outputs from previous steps.

---

## Proxy Endpoints

### /llm/** — LLM Proxy

Proxies all paths to llama.cpp server (auto-loaded via Forge on first request).

| Path | Method | Description |
|------|--------|-------------|
| `/llm` | GET | LLM root |
| `/llm/v1/chat/completions` | POST | Chat (native llama.cpp) |
| `/llm/v1/models` | GET | List loaded GGUF models |
| `/llm/{path:path}` | * | Any llama.cpp endpoint |

### /comfyui/** — ComfyUI Proxy

Proxies all paths to ComfyUI subprocess (auto-loaded via Forge on first request).

| Path | Method | Description |
|------|--------|-------------|
| `/comfyui` | GET | ComfyUI Web UI |
| `/comfyui/prompt` | POST | Submit workflow |
| `/comfyui/{path:path}` | * | Any ComfyUI API endpoint |

---

## Admin

### POST /admin/load

Pre-load a GPU service.

```json
{"service": "trellis", "model": "trellis"}
```

### POST /admin/unload

Unload all GPU services (frees all VRAM).

---

## Dashboard

| Path | Method | Description |
|------|--------|-------------|
| `/dashboard` | GET | GPU metrics dashboard (HTML) |
| `/dashboard/api/gpu` | GET | Current GPU snapshot + Forge state + processes |
| `/dashboard/api/gpu/history` | GET | Rolling 5-minute GPU samples (sparkline data) |
| `/dashboard/api/services` | GET | All Ray Serve deployment statuses |

---

## Studio Switcher (Ray Serve)

GPU app launcher for switching between loaded services. Part of the Ray Serve ingress.

| Path | Method | Description |
|------|--------|-------------|
| `/studio` | GET | Studio app launcher (HTML) |
| `/studio/api/apps` | GET | All services with status + UI metadata |
| `/studio/api/switch` | POST | Swap GPU to a service: `{"service": "comfyui"}` |
| `/studio/api/release` | POST | Release GPU (unload everything) |

---

## Playground (Interactive Service UI)

| Path | Method | Description |
|------|--------|-------------|
| `/playground` | GET | Interactive service testing UI (HTML) |
| `/playground/api/services` | GET | Services with input field metadata for dynamic forms |

---

## Poser (Pose Presets)

| Path | Method | Description |
|------|--------|-------------|
| `/poser/presets` | GET | List all pose presets (name, description, tags) |
| `/poser/presets/{name}/render` | GET | Render pose skeleton as PNG. Query params: `width`, `height`, `line_width`, `point_radius` |

**Available presets:** `standing_neutral`, `t_pose`, `sitting`, `walking`, `running`, `dancing`, `waving`

---

## System

### GET /health

Health check. Always returns `{"status": "ok"}`. No auth required.

### GET /status

System status: Forge VRAM state, GPU memory info, Ray resources. No auth required.

---

## Wan2GP Studio (MCP + Web UI, port 30080)

Browser-accessible GPU generation studio powered by MCP tools and an LLM chat interface. Runs as standalone K8s deployments in the `mcp` namespace — completely separate from Ray.

**Architecture:**
```
Browser → Traefik (:30080) → Wan2GP Studio Web (Next.js)
                                    │
                                    ├─ Chat → LLM (via Ray /llm/v1) → tool calls
                                    │
                                    └─ Direct Generate → MCP Client
                                            │
                                    Wan2GP Studio MCP (FastMCP)
                                            │
                                    Forge (Ray Serve /forge)
                                            │
                                    GPU (RTX 4090)
```

### Wan2GP Studio Web (Next.js)

Access: `http://100.86.69.57:30080/studio/`

| Path | Method | Description |
|------|--------|-------------|
| `/studio/` | GET | Main UI — sidebar generation form + LLM chat interface |
| `/studio/api/health` | GET | Health check: `{"status": "ok", "service": "wan2gp-studio-web"}` |
| `/studio/api/generate` | POST | Direct tool invocation: `{"tool": "generate_image", "args": {...}}` |
| `/studio/api/chat` | POST | LLM chat with tool use (streaming). OpenAI-compatible messages format |

The web UI has two interaction modes:
- **Generate tab** — direct form-based tool calls (pick type, model, params, generate)
- **Chat tab** — natural language chat where the LLM picks the right MCP tool automatically

### Wan2GP Studio MCP Server (FastMCP)

Access: `http://100.86.69.57:30080/mcp/wan2gp-studio/mcp`

MCP endpoint for AI assistants (Claude Desktop, ChatGPT, etc.) and the web UI's MCP client. Wraps the Forge GPU gateway. Streamable HTTP transport.

**6 MCP tools:**

#### generate_video

Generate video from text or image input.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `model` | string | `wan/t2v` | Model: `wan/t2v`, `wan/i2v`, `hunyuan/t2v`, `hunyuan/i2v`, `ltx2` |
| `image_b64` | string | null | Base64 input image (required for i2v models) |
| `width` | int | 768 | Output width (256-1920) |
| `height` | int | 512 | Output height (256-1920) |
| `frames` | int | 81 | Number of frames (8-200) |
| `fps` | int | 24 | Frames per second (8-60) |
| `steps` | int | 30 | Denoising steps (1-100) |
| `guidance` | float | 5.0 | CFG guidance scale (0-20) |
| `seed` | int | -1 | Random seed (-1 for random) |
| `negative_prompt` | string | null | What to avoid |

#### generate_image

Generate image from text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description |
| `model` | string | `flux` | Model: `flux`, `flux_schnell`, `flux2_dev`, `flux2_klein_4b`, `qwen-image-edit` |
| `image_b64` | string | null | Base64 input image (for editing models) |
| `width` | int | 1024 | Output width (256-2048) |
| `height` | int | 1024 | Output height (256-2048) |
| `steps` | int | 24 | Denoising steps (1-100) |
| `guidance` | float | 3.5 | CFG guidance scale (0-20) |
| `seed` | int | -1 | Random seed (-1 for random) |
| `negative_prompt` | string | null | What to avoid |

#### generate_3d

Convert image to 3D mesh model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_b64` | string | required | Base64-encoded input image |
| `model` | string | `trellis` | Model: `trellis` or `anigen` |
| `steps` | int | 50 | Generation steps (1-200) |

#### generate_audio

Generate speech, sound effects, or music.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text for TTS or sound description |
| `model` | string | `moss-soundeffect` | Model: `moss-soundeffect`, `kokoro`, `espeak`, `vibevoice_cpp_gpu`, `vibevoice_cpp_cpu` |
| `voice` | string | null | Voice name (for TTS models) |
| `language` | string | null | Language code (e.g. en, zh) |
| `duration_seconds` | float | null | Target duration (for music/SFX) |

#### list_models

Discover available GPU model families and their variants. Returns families grouped by type (video, image, 3d, audio, music) with live GPU status from the Forge.

#### forge_status

Check GPU status, VRAM usage, and currently loaded services.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detailed` | bool | false | Include per-service VRAM breakdown and GPU node info |

**Internal routing:** The MCP server talks to the Forge at `http://tech-noir-ray-serve-svc.ai-services:8000/forge` (K8s cluster-internal). All GPU inference goes through the Forge's VRAM ledger.

---

## MCP Services (Standalone K8s, port 30080)

All MCP servers run in the `mcp` namespace as standard K8s Deployments — separate from Ray. Accessible via Traefik.

| Route | Service | Description |
|-------|---------|-------------|
| `/mcp/wan2gp-studio/mcp` | Wan2GP Studio MCP | GPU generation tools (6 tools, FastMCP) |
| `/studio/` | Wan2GP Studio Web | Browser UI for GPU generation (Next.js + assistant-ui) |
| `/mcp/media/*` | Media Analysis MCP | YOLOv8, Florence-2, SAM2, OCR, transcription |
| `/mcp/web/*` | Web Research MCP | Search, scrape, extract, crawl |
| `/mcp/equibles/*` | Equibles Financial MCP | SEC filings, stocks, insider trades, congressional trades |

---

## Cloud Burst (Overflow Proxy)

| Route | Method | Description |
|-------|--------|-------------|
| `/overflow/*` | * | Local → SkyServe cloud fallback when local GPU is overloaded |

---

## External Services (Direct Access)

| Service | URL | Description |
|---------|-----|-------------|
| Ray Dashboard | `http://100.86.69.57:30080/ray-dashboard/` | Ray cluster monitoring (via Traefik) |
| ComfyUI UI | `http://100.86.69.57:30080/comfyui/` | ComfyUI web UI (auto-loaded via Forge) |
| Grafana | `http://100.86.69.57:30080/grafana` | Metrics dashboards (via Traefik) |
| Flux Operator | `http://100.86.69.57:30090` | Flux CD GitOps UI (NodePort) |
| Forge Registry | `http://100.86.69.57:30500` | Container image registry (NodePort) |
