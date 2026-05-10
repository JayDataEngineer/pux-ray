# MCP Research Server

FastMCP-based research server with MCP (Model Context Protocol) support for unified search, scraping, and clean LLM-ready output. Accessible remotely via Tailscale VPN with automatic HTTPS.

## Features

- **FastMCP Server**: SSE transport for Claude Desktop, Claude Code, and other MCP clients
- **Tailscale Integration**: Automatic HTTPS via MagicDNS (e.g., `https://mcp-server.tailb1e597.ts.net`)
- **Multi-page Search**: SearXNG (Brave, Bing, DuckDuckGo, Ask) with 10-page pagination
- **Smart Scraping**: Crawl4AI (fast) → SeleniumBase (stealth fallback) → Blacklist
- **PDF Support**: Download and extract text from PDF files using PyMuPDF
- **Domain Rate Limiting**: Redis-backed concurrent request limiting
- **Clean Output**: ContentCleaner with priority extraction for LLM-ready markdown
- **Domain Learning**: PostgreSQL tracks which method works per domain
- **Documentation Tools**: Native FastMCP tools for fetching any URL as clean Markdown
- **Redis Caching**: ResponseCachingMiddleware for search, scrape, and docs
- **Error Handling**: Input validation, ToolError exceptions, masked internal errors
- **VPN/Proxy Gateway**: Gluetun container routes all scraping through WireGuard VPN tunnel
- **Proxy Management**: Runtime tools for proxy status, testing, and rotation
- **Caddy Reverse Proxy**: Professional deployment with automatic TLS

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Caddy (ports 80/443)          │
                    │  Tailscale TLS via shared socket      │
                    └──────────────┬──────────────────────────┘
                                   │ Docker DNS (mcp_net)
                    ┌──────────────▼──────────────────────────┐
                    │         mcp-server (SSE on :8000)      │
                    │  ┌────────────────────────────────┐   │
                    │  │ Web Tools:                     │   │
                    │  │ • search_web                   │   │
                    │  │ • scrape_url                   │   │
                    │  │ • get_domains                  │   │
                    │  │ • clean_database               │   │
                    │  └────────────────────────────────┘   │
                    │  ┌────────────────────────────────┐   │
                    │  │ Docs (native, namespace: docs_) │   │
                    │  │ • docs_list_sources            │   │
                    │  │ • docs_fetch_docs              │   │
                    │  └────────────────────────────────┘   │
                    └──────────────┬──────────────────────────┘
                                   │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  PostgreSQL  │    │  Redis       │    │  SearXNG     │
    │  (domains)   │    │  (cache)     │    │  (search)    │
    └──────────────┘    └──────────────┘    └──────────────┘
                                  │
                    ┌───────────────┼──────────────┐
                    ▼               ▼               ▼
            ┌──────────┐   ┌──────────┐   ┌──────────┐
            │  Worker │   │  Beat    │   │  Flower  │
            │(Celery) │   │(schedule)│   │(monitor) │
            └──────────┘   └──────────┘   └──────────┘


    External Traffic (scrape/search):
    ┌──────────┐   HTTP    ┌──────────┐  WireGuard  ┌───────────┐
    │  MCP /   │ ────────► │ Gluetun  │ ──────────► │  Internet │
    │  Celery  │ :8888     │  (VPN)   │             │           │
    └──────────┘           └──────────┘             └───────────┘

    Internal Traffic (postgres, redis, searxng):
    ┌──────────┐  direct   ┌──────────────┐
    │  MCP /   │ ────────► │  Internal    │
    │  Celery  │           │  services    │
    └──────────┘           └──────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Tailscale account (for remote access)

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your values:
# - TS_AUTHKEY: Get from https://login.tailscale.com/admin/settings/keys
# - TAILNET_DOMAIN: Your tailnet domain (e.g., tailb1e597.ts.net)
# - TAILNET_MACHINE_NAME: Your Tailscale machine name (e.g., mcp-server)
```

### 3. Start Services

```bash
docker compose up -d
```

### 4. Connect via Tailscale

Once Tailscale is running, your MCP server is accessible at:

```
https://<TAILNET_MACHINE_NAME>.<TAILNET_DOMAIN>/sse
```

Example: `https://gtek.tailb1e597.ts.net/sse`

---

## Connecting Clients

### Claude Code (CLI)

```bash
claude mcp add --transport http research https://gtek.tailb1e597.ts.net/sse
```

Replace the URL with your actual Tailscale MagicDNS URL.

### Claude Desktop

Add to your Claude Desktop config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "research": {
      "transport": "sse",
      "url": "https://gtek.tailb1e597.ts.net/sse",
      "timeout": 120000
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can connect via SSE transport to your Tailscale HTTPS URL.

**Local access** (same machine only):
- MCP server: `http://localhost:8327/sse`
- Flower dashboard: `http://localhost:5921`
- VPN proxy: `http://localhost:8328`

---

## Available Tools

### Web Research (no prefix)

| Tool | Description |
|------|-------------|
| `search_web` | Search the web using multiple search engines |
| `scrape_url` | Scrape a URL and extract clean markdown |
| `map_domain` | Discover URLs from sitemaps/Common Crawl (URL discovery) |
| `crawl_site` | Deep crawl with BFS strategy (follows links) |
| `scrape_structured` | Extract structured JSON data using pre-built schemas |
| `list_schemas` | List available extraction schemas |
| `get_domains` | List tracked domains with preferred methods |
| `clean_database` | Clear all domain tracking data |

### Documentation (prefix: `docs_`)

| Tool | Description |
|------|-------------|
| `docs_list_sources` | List available documentation libraries |
| `docs_fetch_docs` | Fetch documentation from any URL (cached, cleaned to Markdown) |

### Proxy Management

| Tool | Description |
|------|-------------|
| `proxy_status` | Show current proxy configuration and rotation stats |
| `proxy_test` | Test proxy connectivity and report exit IP vs real IP |
| `proxy_rotate` | Manually rotate to the next proxy in the rotation list |

---

## Structured Data Extraction

The `scrape_structured` tool extracts structured JSON from web pages using pre-built CSS schemas—no LLM required, much faster and cheaper.

### Schema Types

| Schema | Extracts |
|--------|----------|
| `ecommerce` | Products (name, price, rating, availability, image, url, sku) |
| `news` | Articles (headline, author, date, content, category, summary) |
| `jobs` | Listings (title, company, location, salary, description, type) |
| `blog` | Posts (title, author, date, content, tags, excerpt) |
| `social` | Social posts (username, content, timestamp, likes, shares) |
| `products` | Product catalog multi-item extraction |

### Usage Examples

```python
# Extract products from an e-commerce page
scrape_structured(
    url="https://shop.example.com/products",
    schema_type="ecommerce"
)
# Returns: {"items": [{"title": "...", "price": "$99", "rating": "4.5", ...}]}

# Extract job listings
scrape_structured(
    url="https://jobs.example.com",
    schema_type="jobs"
)
# Returns: {"items": [{"title": "Engineer", "company": "...", "salary": "...", ...}]}

# Extract with custom selector
scrape_structured(
    url="https://news.example.com",
    schema_type="news",
    custom_selector=".article-list article"  # Override base selector
)
```

### When to Use scrape_structured vs scrape_url

| Use scrape_structured when... | Use scrape_url when... |
|-------------------------------|------------------------|
| You need structured JSON data | You need full page content |
| Extracting specific fields (products, jobs) | Reading articles, documentation |
| Building datasets/analysis | General purpose scraping |
| Page has consistent structure | Page structure is unknown |

---

## Map and Crawl Workflow

The new `map_domain` and `crawl_site` tools enable intelligent site exploration:

### 1. Map Domain (URL Discovery)

```python
# Discover all blog posts from a site
map_domain(
    domain="blog.example.com",
    source="sitemap",           # or "cc" for Common Crawl, "sitemap+cc" for both
    pattern="*/posts/*",        # Filter by URL pattern
    max_urls=1000,              # Maximum URLs to return
    extract_head=True           # Extract metadata (slower but richer)
)

# Relevance-based discovery
map_domain(
    domain="docs.example.com",
    source="sitemap",
    query="API reference endpoints",
    score_threshold=0.5,        # Minimum BM25 relevance score
    scoring_method="bm25"
)
```

**Sources:**
- `sitemap`: Fast XML sitemap parsing (100-1000 URLs/second)
- `cc`: Common Crawl dataset (historical index)
- `sitemap+cc`: Both sources for maximum coverage

### 2. Crawl Site (Deep Crawling)

```python
# Crawl documentation with depth limit
crawl_site(
    url="https://docs.example.com",
    max_depth=2,                # Follow links 2 levels deep
    max_pages=50,               # Maximum pages to crawl
    pattern="*/api/*",          # Filter by URL pattern
    word_count_threshold=100    # Skip short pages
)
```

**Strategy:**
- BFS (Breadth-First Search) for systematic exploration
- Respects `max_depth` (link levels) and `max_pages` (total pages)
- Can filter by URL pattern and word count
- Returns full content for each crawled page

### Typical Workflow

```python
# Step 1: Discover URLs
result = map_domain(domain="python.langchain.com", pattern="*/docs/*")

# Step 2: Review discovered URLs
for url_info in result["urls"]:
    print(url_info["url"], url_info.get("title", ""))

# Step 3: Deep crawl for full content
crawl_result = crawl_site(
    url="https://python.langchain.com/docs/",
    max_depth=2,
    max_pages=20
)

# Step 4: Process crawled pages
for page in crawl_result["pages"]:
    print(page["title"], len(page.get("content", "")))
```

---

## Docker Services

| Container | Purpose | Resources |
|-----------|---------|-----------|
| mcp-caddy | Reverse proxy with automatic TLS | - |
| mcp-server | FastMCP server with SSE transport | 512MB limit |
| mcp-celery-worker | Scraping worker (10 parallel browsers) | 3GB limit, 2 CPUs |
| mcp-celery-beat | Periodic task scheduler | 256MB limit |
| mcp-flower | Celery monitoring (localhost:5555) | 256MB limit |
| mcp-postgres | Domain tracking database | 512MB limit |
| mcp-redis | Cache + rate limiting | 256MB limit |
| mcp-searxng | Multi-engine search | 512MB limit |
| mcp-vpn | Gluetun VPN gateway (HTTP proxy) | 256MB limit |
| mcp-ts | Tailscale sidecar (host network) | - |

---

## VPN / Proxy Setup

All search and scrape traffic routes through a WireGuard VPN tunnel via a [Gluetun](https://github.com/qdm12/gluetun) container. Your real IP is never exposed to scraped sites.

**How it works:**

1. Gluetun connects to a VPN server via WireGuard
2. It exposes an HTTP proxy at `gluetun:8888` on the Docker network
3. The MCP server and Celery worker route all external HTTP traffic through this proxy
4. Internal services (Postgres, Redis, SearXNG) connect directly — they're excluded from the proxy

If the VPN tunnel drops, Gluetun's firewall blocks all outbound traffic. There is no fallback to your real IP.

### Setup

**1. Get WireGuard credentials from your VPN provider**

Most providers (Proton VPN, Mullvad, Surfshark, NordVPN, etc.) let you download a WireGuard `.conf` file. It looks like:

```ini
[Interface]
PrivateKey = ABC123abc...
Address = 10.2.0.2/32
DNS = 10.2.0.1

[Peer]
PublicKey = XYZ789xyz...
Endpoint = 1.2.3.4:51820
AllowedIPs = 0.0.0.0/0
```

**2. Add the values to your `.env`**

```bash
WIREGUARD_PRIVATE_KEY=ABC123abc...       # from [Interface] PrivateKey
WIREGUARD_ADDRESS=10.2.0.2/32            # from [Interface] Address
WIREGUARD_PUBLIC_KEY=XYZ789xyz...        # from [Peer] PublicKey
WIREGUARD_ENDPOINT_IP=1.2.3.4             # from [Peer] Endpoint IP
WIREGUARD_ENDPOINT_PORT=51820             # from [Peer] Endpoint port
```

**3. Start the stack**

```bash
docker compose up -d
```

**4. Verify it's working**

Use the `proxy_test` MCP tool — it reports your proxy exit IP and your real IP. If `ip_different: true`, your traffic is going through the VPN.

### Proxy Rotation

To rotate across multiple proxies (e.g., different VPN servers or a proxy pool), set `MCP_PROXY_URLS` instead:

```bash
# Comma-separated list — rotates round-robin by default
MCP_PROXY_URLS=http://gluetun:8888,http://proxy2:8080,socks5://proxy3:1080
```

Use `proxy_rotate` to manually advance to the next proxy, or `proxy_status` to see current rotation state.

### Advanced Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_PROXY_URL` | `http://gluetun:8888` | Single proxy URL (HTTP or SOCKS5) |
| `MCP_PROXY_URLS` | _(empty)_ | Comma-separated list for rotation (overrides `MCP_PROXY_URL`) |
| `MCP_PROXY_ROTATION` | `round-robin` | Rotation strategy: `round-robin` or `random` |
| `MCP_PROXY_EXCLUDE` | `searxng,postgres,redis,localhost,127.0.0.1` | Hostnames to bypass proxy |

### Disabling the VPN

To run without a proxy, comment out the `gluetun` service in `docker-compose.yml` and set `MCP_PROXY_URL=` (empty) in your `.env`. All traffic will go direct.

---

## Environment Variables

See `.env.example` for all configurable values:

```bash
# Tailscale
TS_AUTHKEY=tskey-auth-<your-key>
TAILNET_DOMAIN=your-tailnet.ts.net
TAILNET_MACHINE_NAME=mcp-server

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_DB=mcp_server
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# SearXNG
SEARXNG_SECRET=<generate with: openssl rand -hex 32>

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# API
MCP_PORT=8000
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# VPN (WireGuard credentials from your VPN provider)
WIREGUARD_PRIVATE_KEY=<from your WireGuard config>
WIREGUARD_ADDRESS=10.2.0.2/32
WIREGUARD_PUBLIC_KEY=<from your WireGuard config>
WIREGUARD_ENDPOINT_IP=<server_ip>
WIREGUARD_ENDPOINT_PORT=51820

# Caching (seconds)
SEARCH_CACHE_TTL=300
SCRAPE_CACHE_TTL=3600
DOCS_CACHE_TTL=3600
```

---

## Scraping Flow

```
1. Check Cache → Return cached if available
2. Check Rate Limit → Wait if too many concurrent to domain
3. Check Blacklist → Reject if blacklisted
4. PDF? → Extract text with PyMuPDF
5. Reddit? → Use Reddit JSON API
6. Check Database → Use learned preference
7. Try Crawl4AI (3x retry, fast, JS-enabled)
8. If failed → Try Selenium (3x retry, stealth mode)
9. If both failed → Blacklist domain
10. Clean HTML → Waterfall strategy for universal scraping
```

**Content Extraction Priority (Waterfall Strategy):**
1. CSS selector (if provided)
2. **Waterfall** (Selectolax aggressive pruning + semantic targeting)
   - Works on ALL page types: articles, SaaS, landing pages, SPAs
   - Aggressive junk tag removal (script, style, nav, footer, form, etc.)
   - Semantic targeting (`<main>`, `<article>`, `#content`)
   - Full body fallback for chaotic layouts
3. Trafilatura (article-only fallback for news/blogs)
4. BeautifulSoup (nuclear option)

---

## Error Handling

The server implements FastMCP-compliant error handling:

- **Input Validation**: All parameters use `Annotated[Field]` with constraints
  - `min_length`/`max_length` for strings
  - `ge`/`le` for numeric ranges
  - `Literal` for enum choices
- **User-Facing Errors**: `ToolError` for client-visible messages
- **Internal Errors**: Masked from clients (security), logged server-side
- **HTTP-Specific Messages**: 404, 403, 500 errors return helpful context

---

## Caching

Redis-backed caching with configurable TTL:

```bash
# Cache TTL (seconds)
SEARCH_CACHE_TTL=300      # 5 minutes for search results
SCRAPE_CACHE_TTL=3600     # 1 hour for scraped content
DOCS_CACHE_TTL=3600       # 1 hour for documentation
```

Cached responses include metadata (method used, timestamp) and bypass expensive operations.

---

## Documentation Sources

The server includes official documentation for:

- **LangGraph** - Agent framework
- **LangChain** - LLM framework (python.langchain.com, docs.langchain.com)
- **DeepAgents** - LangChain agent patterns
- **FastAPI** - Web framework
- **Pydantic** - Data validation (docs.pydantic.dev, ai.pydantic.dev)
- **FastMCP** - MCP framework (gofastmcp.com)
- **Docker** - Container platform
- **Next.js** - React framework
- **Vercel AI** - AI SDK for React (ai-sdk.dev)

Add more in `docs_config.yaml`. Domains linked in llms.txt files are automatically discovered and allowed.

---

## Tech Stack

- **FastMCP** - MCP server framework with SSE transport
- **Caddy** - Reverse proxy with automatic TLS
- **PostgreSQL** - Domain tracking and learning
- **Celery + Redis** - Task queue and rate limiting
- **SearXNG** - Multi-engine search
- **Crawl4AI** - Fast JS-enabled scraping, URL seeding, and deep crawling
- **SeleniumBase** - Stealth scraping fallback
- **ContentCleaner** - Multi-strategy HTML→Markdown conversion
- **Gluetun** - VPN gateway with WireGuard and HTTP proxy
- **Tailscale** - VPN + MagicDNS
