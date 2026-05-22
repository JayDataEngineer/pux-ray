# Forge API Endpoints

Base URL (dev): `http://localhost:30080`
Base URL (prod): `http://100.86.69.57:30080`

Auth: `X-API-Key` header or `api_key` query param (empty = no auth in dev).

---

## Forge Deployment (Ray Serve `POST /`)

The Forge deployment itself is an HTTP endpoint at Ray Serve's internal routing.
These are the actions the ingress dispatches to.

| Action | Method | Endpoint | Body | Response |
|--------|--------|----------|------|----------|
| Invoke service | `POST` | `/` | `{"service": "...", ...payload}` | result dict |
| Preload service | `POST` | `/` | `{"action": "preload", "service": "...", "model": "...", "quant": "..."}` | `{"status": "loaded", "service": "...", ...}` |
| Release | `POST` | `/` | `{"action": "release", "service": "..."}` | `{"status": "released", "service": "..."}` |
| Release all | `POST` | `/` | `{"action": "release"}` | `{"status": "released", "services": ["..."]}` |
| Status | `GET` | `/` | — | full status with loaded services, VRAM, GPU info |
| Status | `POST` | `/` | `{"action": "status"}` | same as GET |
| Pipeline | `POST` | `/` | `{"action": "run_pipeline", "pipeline_id": "...", "params": {...}}` | pipeline result |

---

## Ingress API (Gateway, port 30080)

### Health & Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check. Returns `{"status": "ok"}` |
| `GET` | `/status` | Full status: loaded services, VRAM, GPU info, resources |

### Service Discovery

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/services` | List all registered services with metadata |
| `GET` | `/v1/services/{name}` | Info about a specific service |
| `GET` | `/v1/models` | OpenAI-compatible model list (supports `?category=` filter) |

### Service Invocation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/{service}/generate` | Generic TNAP generate — routes through Forge for GPU services |
| `POST` | `/v1/run` | Unified endpoint: single service, named pipeline, or inline steps |

Payload shapes for `/v1/run`:

```json
// Single service
{"service": "wan2gp", "model": "z_image", "params": {...}}

// Named pipeline
{"pipeline": "tech-noir/generate", "params": {"prompt": "..."}}

// Inline steps (returns SSE)
{"steps": [{"name": "gen", "service": "wan2gp", "params": {...}}]}
```

### LLM

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/llm/configure` | Set model, engine, startup flags |
| `GET` | `/llm` | LLM proxy root |
| `POST` | `/llm` | LLM proxy POST |
| `GET` | `/llm/{path:path}` | LLM proxy sub-path |
| `POST` | `/llm/{path:path}` | LLM proxy sub-path POST |

### Audio

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/speech` | Text-to-speech (model-based dispatch) |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (form-based, multipart) |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/admin/load` | Preload a service. Body: `{"service": "...", "model": "..."}` |
| `POST` | `/admin/unload` | Release all loaded services |

### ComfyUI Proxy

| Method | Path | Description |
|--------|------|-------------|
| `GET/POST/PUT/DELETE` | `/comfyui` | ComfyUI proxy root |
| `GET/POST/PUT/DELETE` | `/comfyui/{path:path}` | ComfyUI proxy sub-path (preserves raw paths) |

### Pipelines (SSE)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/pipelines/execute` | Execute multi-step inference pipeline (returns SSE) |

### Workflows

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/workflows` | List available workflows |
| `GET` | `/v1/workflows/{workflow}` | Workflow metadata |
| `POST` | `/v1/workflows/{workflow}` | Execute workflow |
| `GET` | `/v1/run/catalog` | Discover available pipelines and services |

### Web UIs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard` | GPU metrics dashboard page |
| `GET` | `/dashboard/api/gpu` | GPU metrics (current) |
| `GET` | `/dashboard/api/gpu/history` | GPU metrics (time series) |
| `GET` | `/dashboard/api/services` | Dashboard service list |
| `GET` | `/studio` | Studio page |
| `GET` | `/studio/api/apps` | Studio app list |
| `POST` | `/studio/api/switch` | Switch loaded GPU app |
| `POST` | `/studio/api/release` | Release current GPU app |
| `GET` | `/playground` | Playground page |
| `GET` | `/playground/api/services` | Playground service list |

### Poser

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/poser/presets` | List pose presets |
| `GET` | `/poser/presets/{name}/render` | Render a pose preset as skeleton image |

---

## Service Registry (Forge-managed GPU services)

These services are managed by the Forge VRAM scheduler:

| Service Key | Label | Category | Output | VRAM |
|-------------|-------|----------|--------|------|
| `wan2gp` | Wan2GP Pool | creative | video | mmgp (self-managed) |
| `comfyui` | ComfyUI | image | proxy | declared |
| `llm` | LLM (llama.cpp) | llm | json | declared |
| `avatar` | Avatar Pipeline | avatar | video | staged |
| `kimodo_demo` | Kimodo Demo | motion | proxy | declared |

Additional services route through Wan2GP:

| Service Key | Label | Category | Output |
|-------------|-------|----------|--------|
| `kokoro` | Kokoro TTS | tts | audio |
| `espeak` | eSpeak TTS | tts | audio |
| `faster_whisper` | Faster-Whisper | asr | json |
| `index_tts` | IndexTTS | tts | audio |
| `moss_soundeffect` | MOSS-SoundEffect | audio | audio |
| `ace_step` | ACE-Step | audio | audio |
| `vibevoice_asr` | VibeVoice ASR | asr | json |
| `vibevoice_tts` | VibeVoice TTS | tts | audio |
| `trellis` | TRELLIS 3D | 3d | model_3d |
| `anigen` | AniGen 3D | 3d | model_3d |
| `z_image` | Wan2GP Image | image | image |
| `faster_qwen3_tts` | Faster Qwen3-TTS | tts | audio |
| `hy_motion` | HY-Motion | motion | motion |
| `see_through` | See-Through | creative | image |
| `kimodo` | Kimodo Motion | motion | motion |

---

## Persistence Levels

| Level | Value | Behavior |
|-------|-------|----------|
| `TRANSIENT` | 0 | Evicted freely between calls |
| `PERSISTENT` | 1 | Evicted only if no transient services remain |
| `PIPELINE_LOCKED` | 2 | Never evicted (set during pipeline execution) |
