# Tech Noir MCP Hub

Single entry point for all MCP services on the Tech Noir server.

## Endpoints

| MCP Server | Hub URL | Transport |
|---|---|---|
| Media Analysis | `https://cloud.tailb1e597.ts.net/mcp/media` | Streamable HTTP |
| Web Research | `https://cloud.tailb1e597.ts.net/mcp/web` | Streamable HTTP |

LAN / Tailnet direct (same machine):

| MCP Server | URL |
|---|---|
| Media Analysis | `http://192.168.1.184:30080/mcp/media` or `http://100.86.69.57:30080/mcp/media` |
| Web Research | `http://192.168.1.184:30080/mcp/web` or `http://100.86.69.57:30080/mcp/web` |

## Connecting from Claude Code

```bash
# Via Funnel (works from anywhere on the internet)
claude mcp add media --transport http https://cloud.tailb1e597.ts.net/mcp/media
claude mcp add web --transport http https://cloud.tailb1e597.ts.net/mcp/web

# Via Tailnet (other machines on your tailscale network)
claude mcp add media --transport http http://100.86.69.57:30080/mcp/media
claude mcp add web --transport http http://100.86.69.57:30080/mcp/web
```

## Connecting from Claude Desktop

In `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "media": {
      "type": "http",
      "url": "https://cloud.tailb1e597.ts.net/mcp/media"
    },
    "web": {
      "type": "http",
      "url": "https://cloud.tailb1e597.ts.net/mcp/web"
    }
  }
}
```

## Connecting from Other MCP Clients (Cursor, Continue.dev, etc.)

Any SSE/Streamable HTTP client. Use the URL format above with `type: http`.

## Architecture

```
Client → https://cloud.tailb1e597.ts.net/mcp/media
  └─ Tailscale Funnel :443 → http://127.0.0.1:30080
       └─ k3s NodePort :30080 → Traefik pod :80
            └─ rewrite /mcp/media → /mcp
                 └─ K8s Endpoint → 192.168.1.184:8001
                      └─ Docker: media-analysis-mcp
```

Routes:
- `/mcp/media/*` → Media Analysis MCP (image/audio/video analysis)
- `/mcp/web/*`   → Web Research MCP (web scraping, search)
- `/llm/*`          → LLM (chat, vision)
- `/tts/*`          → Text-to-speech services
- `/forge/*`        → Master router (3D, music, image gen)
- `/ray-dashboard/*` → Ray cluster dashboard
- `/*`              → Ray Serve catch-all

## Available Tools

### Media Analysis MCP (19 tools)
`process`, `analyze_image`, `detect_objects`, `tag_image`, `extract_colors`,
`read_barcodes`, `extract_exif`, `detect_faces`, `classify_nsfw`,
`segment_image`, `transcribe_audio`, `classify_audio`, `fingerprint_audio`,
`diarize_audio`, `check_video`, `detect_scenes`, `detect_objects_text`,
`phi4_vision`, `kosmos_ocr`

### Web Research MCP (20+ tools)
`research`, `search`, `scrape_url`, `extract`, `list_schemas`, `map_domain`,
`crawl_site`, `process_html`, `analyze_image`, `docs_list_sources`,
`docs_fetch_docs`, `domains`, `stats`, `clear_blacklist`, `proxy_status`,
and more.
