# Connecting to MCP Server via Tailscale

## Quick Connect

The MCP server is accessible via your Tailscale network at:

```
http://100.102.244.97:8000/mcp
```

Or via Tailscale DNS:

```
http://dockerhost-1.tailb1e597.ts.net:8000/mcp
```

## Verify Connection

Test from your terminal:

```bash
# Health check
curl http://100.102.244.97:8000/health

# Should return:
# {"status":"healthy","server":"mcp-research-server"}
```

## Claude Desktop Configuration

Add to your Claude Desktop config (`claude_desktop_config.json`):

### On macOS/Linux: `~/Library/Application Support/Claude/claude_desktop_config.json`
### On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "dockerhost-mcp": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/inspector"],
      "env": {
        "MCP_SERVER_URL": "http://100.102.244.97:8000/mcp"
      }
    }
  }
}
```

**OR use SSE transport (recommended for reliability):**

```json
{
  "mcpServers": {
    "dockerhost-mcp": {
      "url": "http://100.102.244.97:8000/mcp"
    }
  }
}
```

## Clearing the Blacklist

If domains are incorrectly blocked, use the `clear_blacklist` tool:

```bash
# Via curl (requires session ID from initialize)
curl -X POST http://100.102.244.97:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "clear_blacklist", "arguments": {}}
  }'
```

Or directly in database (plus clear Redis cache for immediate effect):

```bash
# Unblacklist all domains in database
docker exec mcp-postgres psql -U postgres -d mcp_server -c \
  "UPDATE domains SET is_blacklisted = false, failure_count = 0;"

# Clear FastMCP response cache (required for immediate effect)
docker exec mcp-redis redis-cli FLUSHDB
```

## Troubleshooting

### "MCP server is offline" Error

1. **Check if host is online:**
   ```bash
   tailscale ping dockerhost
   ```

2. **Verify Tailscale IP hasn't changed:**
   ```bash
   tailscale status --json | grep "TailscaleIPs"
   ```

3. **Test direct connection:**
   ```bash
   curl -v http://100.102.244.97:8000/health
   ```

### Connection Refused

1. **Ensure containers are running:**
   ```bash
   cd /path/to/docker/lang-tools
   docker compose ps
   ```

2. **Restart if needed:**
   ```bash
   docker compose up -d
   ```

### Port Already in Use

If port 8000 is already in use on the host:

```bash
# Find what's using port 8000
sudo lsof -i :8000

# Or change the port in docker-compose.yml:
# mcp-server:
#   ports:
#     - "8001:8000"  # Use 8001 instead
```

## Architecture

```
┌─────────────────┐
│  Your Client    │
│ (Claude Desktop) │
└────────┬────────┘
         │ Tailscale VPN
         ▼
┌─────────────────────────────────────┐
│  Tailscale Node (dockerhost)        │
│  ├─ Caddy (host network, port 80)   │
│  └─ MCP Server (bridge, port 8000)  │
│     └─ 8GB RAM, 5 concurrent browsers│
└─────────────────────────────────────┘
```

## Available Tools

Once connected, these tools are available:

- `search_web` - Search multiple engines
- `scrape_url` - Scrape any URL with auto method selection
- `scrape_structured` - Extract structured JSON data
- `map_domain` - Discover URLs from sitemaps
- `crawl_site` - Deep crawl with BFS strategy
- `list_schemas` - List available extraction schemas
- `get_domains` - View tracked domains & methods
- `clean_database` - Reset domain tracking

## Performance Notes

- **Concurrent browsers**: Max 5 Crawl4AI + 10 Selenium
- **Per-domain rate limit**: 3 concurrent requests
- **Typical scrape time**: 3-8 seconds
- **Memory limit**: 8GB (plenty of headroom)

## Current Tailscale Info

```
Node: dockerhost-1
IP: 100.102.244.97
DNS: dockerhost-1.tailb1e597.ts.net
Status: Online
```
