# Media Analysis MCP Server

Standalone MCP server for image, audio, and video analysis. Runs entirely on CPU — no GPU required (but supports one).

## Quick Start

```bash
git clone <repo-url> media-analysis-mcp
cd media-analysis-mcp
docker compose up -d --build
```

Server starts at `http://localhost:8001/mcp`.

### GPU Mode (NVIDIA)

```bash
# One command — GPU torch + CUDA runtime
TORCH_VARIANT=cu124 MEDIA_DEVICE=cuda \
  docker compose up -d --build
```

Requires: NVIDIA GPU, NVIDIA Container Toolkit, Docker Compose v2.

Uncomment the `devices` section in `docker-compose.yml` for GPU resource reservation.

## Tools (16 total)

### Smart Router
| Tool | Description |
|------|-------------|
| `process(query, media_url)` | Routes to the right tool automatically |

### Image Tools
| Tool | Model | What it does |
|------|-------|-------------|
| `analyze_image` | Florence-2 (~900MB) | Captions, OCR, object detection, dense captions |
| `detect_objects` | YOLOv8-nano (~6MB) | Bounding box object detection |
| `tag_image` | WD14 (~300MB) | Content tags and categories |
| `extract_colors` | ColorThief | Dominant color + palette extraction |
| `read_barcodes` | pyzbar | QR code and barcode reading |
| `extract_exif` | Pillow | Camera, GPS, timestamps, metadata |
| `detect_faces` | InsightFace (~350MB) | Face detection, landmarks, embeddings |
| `classify_nsfw` | NudeNet (~100MB) | NSFW content scoring |
| `segment_image` | SAM 2 (~200MB) | Image segmentation masks |

### Audio Tools
| Tool | Model | What it does |
|------|-------|-------------|
| `transcribe_audio` | Parakeet TDT v3 (~300MB) | Speech-to-text (36x realtime) |
| `classify_audio` | PANNs (~200MB) | Sound event detection (AudioSet 527 classes) |
| `fingerprint_audio` | Chromaprint | Audio fingerprint for identification |
| `diarize_audio` | Pyannote 3.1 (~1GB) | Speaker diarization (who speaks when) |

### Video Tools
| Tool | Model | What it does |
|------|-------|-------------|
| `check_video` | FFmpeg + SSIM | Keyframe extraction, temporal consistency |
| `detect_scenes` | PySceneDetect | Shot boundary and scene change detection |

## Tool Profiles

Control which tools are available via `MEDIA_PROFILE`:

```bash
# No ML models at all — instant startup, minimal RAM
MEDIA_PROFILE=minimal docker compose up -d

# Core ML models (Florence-2, YOLOv8, WD14, ASR) + easy wins (default)
MEDIA_PROFILE=standard docker compose up -d

# All ML models including heavy ones (InsightFace, SAM 2, NudeNet, PANNs)
MEDIA_PROFILE=full docker compose up -d

# Everything including Pyannote (requires HF token)
MEDIA_PROFILE=all MEDIA_PYANNOTE_TOKEN=hf_xxx docker compose up -d
```

Override individual tools regardless of profile:
```bash
# Standard profile but also enable face detection
MEDIA_PROFILE=standard MEDIA_FACE_ENABLED=true docker compose up -d
```

## Auto Spin-Down

Models that haven't been used in 30 minutes automatically unload from memory.
They re-load on the next request (lazy loading). Configure or disable:

```bash
# Change timeout to 10 minutes
MEDIA_IDLE_TIMEOUT=600 docker compose up -d

# Disable auto-unload entirely
MEDIA_IDLE_TIMEOUT=0 docker compose up -d
```

## Memory Budget

| Profile | Startup RAM | Max RAM (all models loaded) |
|---------|------------|---------------------------|
| minimal | ~200MB | ~200MB |
| standard | ~500MB | ~2.2GB |
| full | ~500MB | ~4.1GB |

With lazy loading + idle timeout, actual usage stays well under the 4GB container limit.

## Configuration

All settings use `MEDIA_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIA_PROFILE` | `standard` | Tool profile (minimal/standard/full/all) |
| `MEDIA_DEVICE` | `cpu` | Compute device (cpu or cuda) |
| `MEDIA_IDLE_TIMEOUT` | `1800` | Auto-unload idle models after N seconds |
| `MEDIA_PORT` | `8001` | Server port |
| `MEDIA_ROUTER_ENABLED` | `true` | Enable FunctionGemma smart router |
| `MEDIA_PYANNOTE_TOKEN` | | HuggingFace token for speaker diarization |

## Connecting from Claude CLI

```bash
# Add to ~/.claude/settings.json MCP servers:
{
  "mcpServers": {
    "media": {
      "type": "http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

Or across Tailscale:
```bash
# If served via Tailscale funnel at gtek.tailb1e597.ts.net
"url": "https://gtek.tailb1e597.ts.net/media/mcp"
```

## Examples

```bash
# Analyze an image
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": 1,
    "method": "tools/call",
    "params": {
      "name": "process",
      "arguments": {
        "query": "describe this image",
        "media_url": "https://example.com/photo.jpg"
      }
    }
  }'

# Extract color palette
curl -X POST http://localhost:8001/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"extract_colors","arguments":{"imageSource":"https://example.com/logo.png","color_count":8}}}'

# Transcribe audio
curl -X POST http://localhost:8001/mcp \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"transcribe_audio","arguments":{"audioSource":"https://example.com/speech.mp3"}}}'
```

## Tech Stack

- **FastMCP 3.x** — MCP server framework
- **PyTorch** (CPU or CUDA) — Florence-2, SAM 2
- **ONNX Runtime** — WD14, YOLOv8, InsightFace, NudeNet, PANNs
- **llama-cpp-python** — FunctionGemma GGUF router
- **UV** — Fast Python package management
- **Docker** — Multi-stage build, ~2GB image (CPU)
