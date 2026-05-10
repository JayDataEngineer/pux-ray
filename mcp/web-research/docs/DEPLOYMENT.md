# MCP Server Deployment Guide

Everything you need to stand up both MCP servers on a new machine.

## Prerequisites

- Ubuntu 22.04+ (or any Linux with Docker)
- Docker + Docker Compose v2
- Tailscale (for remote access)
- At least 4GB RAM (8GB recommended for both servers running simultaneously)
- Two NVMe slots (or one — both servers can share a single disk)

## Hardware Overview

| Component | This Build |
|-----------|-----------|
| Machine | GMKtec NucBox K10 |
| CPU | Intel Raptor Lake-P (13th gen) |
| RAM | 32GB DDR5 |
| NVMe 0 | Samsung 970 EVO Plus 2TB (Linux/Ubuntu, LUKS encrypted) |
| NVMe 1 | Crucial CT1000 1TB (unused Windows install) |
| GPU | Intel Iris Xe (integrated, no eGPU currently attached) |
| OCuLink | Yes — PCIe 4.0 Graphics Port available for eGPU |

## Project Locations

| Project | Path | Repo |
|---------|------|------|
| Research MCP | `/home/user/projects/host/local-web-mcp` | `github.com/JayDataEngineer/local-web-mcp` |
| Media Analysis MCP | `/home/user/projects/host/media-analysis-mcp` | `github.com/JayDataEngineer/media-analysis-mcp` |

## Research MCP Server

### Stack
- **FastMCP 3.x** with Streamable HTTP (stateless, 2 workers)
- **PostgreSQL** — domain tracking, learned scraping preferences
- **Redis** — URL-level scrape cache (24h TTL), search cache (5min)
- **SearXNG** — meta search engine
- **Celery + Redis** — background tasks
- **Caddy** — reverse proxy (Tailscale handles TLS now)
- **VPN proxy** — ProtonVPN SOCKS5 for outbound scraping
- **Florence-2** — vision model (analyze_image tool)

### Key Architecture Decisions
- `stateless_http=True` — eliminates session 404s with multiple workers
- URL-level Redis cache in `scrape_service.py` — prevents re-scraping same URL
- FunctionGemma is NOT used here (unlike media server) — routing is simpler
- VPN proxy rotates IPs to avoid scraping blocks

### Deployment
```bash
cd /home/user/projects/host/local-web-mcp
docker compose up -d --build
```

### Key Files
| File | Purpose |
|------|---------|
| `src/mcp_sse.py` | Main server, lifespan, tool registration, middleware |
| `src/tools/web_tools.py` | All web tools (research, search, scrape, extract, etc.) |
| `src/services/scrape_service.py` | URL-level Redis caching + fallback routing |
| `src/settings.py` | All config (`MCP_` env prefix) |
| `docker-compose.yml` | Full stack: postgres, redis, searxng, celery, vpn, mcp |

### Environment Variables (key ones)
- `MCP_REDIS_HOST`, `MCP_REDIS_PORT` — Redis for caching
- `MCP_POSTGRES_*` — PostgreSQL for domain tracking
- `MCP_SEARXNG_URL` — SearXNG instance URL
- `MCP_PROXY_URL` / `MCP_PROXY_URLS` — VPN proxy for outbound scraping

### Health Check
```bash
curl http://localhost:8000/health
```

## Media Analysis MCP Server

### Stack
- **FastMCP 3.x** with Streamable HTTP (stateless, 2 workers)
- **FunctionGemma 270M GGUF** — smart tool router (~300MB, loads at startup)
- **Florence-2** (~900MB) — image analysis
- **YOLOv8-nano** (~6MB) — object detection
- **WD14** (~300MB) — image tagging
- **Parakeet TDT v3** (~300MB) — speech-to-text
- **InsightFace** (~350MB) — face detection
- **NudeNet** (~100MB) — NSFW detection
- **PANNs** (~200MB) — audio classification
- **SAM 2** (~200MB) — image segmentation
- **Pyannote 3.1** (~1GB) — speaker diarization (disabled by default)
- **ColorThief, pyzbar, Pillow EXIF, PySceneDetect, Chromaprint** — no ML, instant

### Profiles (MEDIA_PROFILE)
| Profile | What loads | Startup RAM | Max RAM |
|---------|-----------|-------------|---------|
| `minimal` | No ML models | ~200MB | ~200MB |
| `standard` | Florence-2, YOLOv8, WD14, ASR, video | ~500MB | ~2.2GB |
| `full` | + InsightFace, NudeNet, PANNs, SAM 2 | ~500MB | ~4.1GB |
| `all` | + Pyannote (needs HF token) | ~500MB | ~5GB |

### GPU/CPU Toggle
- **CPU** (default): `docker compose up -d --build`
- **GPU** (NVIDIA): `TORCH_VARIANT=cu124 MEDIA_DEVICE=cuda docker compose up -d --build`

### Idle Auto-Unload
Models spin down after 30 min idle, re-load on next request.
- Configure: `MEDIA_IDLE_TIMEOUT=1800` (default, 30 min)
- Disable: `MEDIA_IDLE_TIMEOUT=0`

### Deployment
```bash
cd /home/user/projects/host/media-analysis-mcp
docker compose up -d --build
```

### Key Files
| File | Purpose |
|------|---------|
| `src/server.py` | FastMCP server, lifespan, tool registration |
| `src/settings.py` | Profiles, device toggle, `is_enabled()` |
| `src/router/function_router.py` | FunctionGemma + fallback routing |
| `src/tools/media_tools.py` | All 16 tools + `process()` dispatch |
| `src/services/*.py` | One per model/capability |
| `src/services/idle_watcher.py` | Auto-unload background task |

## Tailscale Integration

### Current Setup
- **Hostname**: `gtek`
- **Tailnet**: `gtek.tailb1e597.ts.net`
- **Funnel**: Research server exposed at `https://gtek.tailb1e597.ts.net/mcp`

### Funnel Commands
```bash
tailscale funnel status
tailscale funnel --bg --https=443 http://127.0.0.1:8327   # Research server
tailscale serve --set-path /media http://127.0.0.1:8001     # Media server path
```

### Claude CLI Config
```json
{
  "mcpServers": {
    "research": {
      "type": "http",
      "url": "https://gtek.tailb1e597.ts.net/mcp"
    },
    "media-analysis": {
      "type": "http",
      "url": "http://gtek:8001/mcp"
    }
  }
}
```

## Migration Checklist

1. **Pull both repos** on the new machine
2. **Copy `.env` files** (if any) — they contain secrets, NOT in git
3. **Research server**: `docker compose up -d --build` (needs .env for DB passwords, VPN creds)
4. **Media server**: `docker compose up -d --build` (works out of the box, no .env needed)
5. **Tailscale**: Install and auth on new machine, update funnel if hostname changes
6. **Claude CLI**: Update MCP server URLs if IP/hostname changed
7. **Docker volumes**: Model cache (HuggingFace), Redis data, Postgres data — these rebuild on first run
8. **Test**: `curl http://localhost:8000/health` (research) and `curl -X POST http://localhost:8001/mcp -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'` (media)

## Known Issues / Lessons Learned

- **Session 404s**: Fixed with `stateless_http=True` — never go back to stateful
- **WD14 tagger**: Model expects NHWC format `[1, 448, 448, 3]`, NOT NCHW
- **ASR model**: Use alias `nemo-parakeet-tdt-0.6b-v3`, not full HF repo ID
- **ASR audio format**: onnx-asr requires WAV, not OGG/MP3 — ffmpeg conversion needed
- **Florence-2**: Pinned to `transformers==4.45.2` for compatibility
- **FunctionGemma**: Use `unsloth/functiongemma-270m-it-GGUF` (not meetkai, which 404s)
- **PyTorch CPU**: Must force-reinstall after `uv sync` via `--index-url https://download.pytorch.org/whl/cpu`
- **VPN proxy**: Research server uses ProtonVPN SOCKS5 for outbound scraping — works fine, don't blame it
- **Wikipedia user-agent**: Wikipedia blocks our default User-Agent — use different test images
