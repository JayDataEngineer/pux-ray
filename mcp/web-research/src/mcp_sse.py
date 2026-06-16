"""MCP Server with Streamable HTTP transport

Web Research Tools:
- research: Search + scrape top results in one call (recommended)
- search: Search using multiple search engines
- scrape: Scrape a URL and extract clean markdown
- extract: Extract structured JSON data using pre-built schemas
- list_schemas: List available extraction schemas
- map: Discover URLs from sitemaps/Common Crawl
- crawl: Deep crawl with BFS or Best-First strategy
- process_html: Convert raw HTML to clean markdown

Vision Tools:
- analyze_image: Analyze images using Florence-2 vision model (CPU)

Admin Tools:
- domains: List tracked domains with preferred methods
- stats: View scrape statistics and metrics
- reset: Clear all domain tracking data
- clear_blacklist: Clear all blacklisted domains

Documentation:
- docs_list_sources: List available documentation libraries
- docs_fetch_docs: Fetch documentation from llms.txt sources

Proxy:
- proxy_status, proxy_test, proxy_rotate
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from urllib.parse import urlparse

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan
from fastmcp.server.providers import LocalProvider
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.exceptions import ToolError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from loguru import logger
from pydantic import Field, HttpUrl, ValidationError, field_validator, BeforeValidator
import json
import os


# ========== CONFIGURATION ==========

from .settings import get_settings

settings = get_settings()

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in settings.allowed_origins.split(",")
    if origin.strip()
]

REDIS_HOST = settings.redis_host
REDIS_PORT = settings.redis_port
REDIS_PASSWORD = settings.redis_password

# Cache TTL from settings
SEARCH_CACHE_TTL = settings.search_cache_ttl
SCRAPE_CACHE_TTL = settings.scrape_cache_ttl


# ========== REDIS STORE SETUP ==========

def _create_redis_store():
    """Create namespaced Redis store for FastMCP state and caching"""
    from key_value.aio.stores.redis import RedisStore
    from key_value.aio.wrappers.prefix_collections import PrefixCollectionsWrapper

    base_store = RedisStore(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,  # Use database 0 for FastMCP state
    )

    # Namespace to avoid conflicts with other apps using same Redis
    return PrefixCollectionsWrapper(
        key_value=base_store,
        prefix="mcp-server"
    )


# ========== LIFESPAN ==========

@lifespan
async def service_lifespan(server: FastMCP):
    """Initialize and cleanup services on server startup/shutdown

    Yields a dict with services that becomes accessible via ctx.lifespan_context
    """
    from .services.search_service import get_search_service
    from .services.scrape_service import get_scrape_service
    from .services.crawl_service import get_map_crawl_service
    from .services.content_cleaner import get_content_cleaner
    from .services.vision_service import get_vision_service
    from .db.database import get_db

    logger.info("Initializing services...")
    search_service = get_search_service()
    scrape_service = get_scrape_service()
    crawl_service = get_map_crawl_service()
    cleaner = get_content_cleaner()
    vision_service = get_vision_service()

    # Database is optional — domain tracking improves over time but is not
    # required for core search/scrape functionality.  If Postgres is
    # unreachable (e.g. DNS flake during pod startup) we serve degraded
    # without it rather than crash-looping.
    db = None
    try:
        db = await get_db()
    except Exception as e:
        logger.warning(f"Database unavailable — serving without domain tracking: {e}")

    try:
        yield {
            "search_service": search_service,
            "scrape_service": scrape_service,
            "crawl_service": crawl_service,
            "cleaner": cleaner,
            "vision_service": vision_service,
            "db": db,
        }
    finally:
        logger.info("Shutting down services...")
        await search_service.close()
        await scrape_service.close()
        await crawl_service.close()
        await vision_service.close()
        if db is not None:
            await db.close()
        logger.info("Shutdown complete")


# ========== FASTMCP SERVER ==========

# Create Redis store for session state and caching
redis_store = None
try:
    redis_store = _create_redis_store()
except ImportError:
    logger.warning("py-key-value-aio[redis] not available, using in-memory session state")
except Exception as e:
    logger.warning(f"Failed to initialize Redis store: {e}")

mcp = FastMCP(
    name="mcp-research-server",
    instructions=(
        "Provides web research tools. "
        "Use 'research' to search and read top results in one call (recommended for most queries). "
        "Use 'search' for lightweight result lists (titles/snippets only). "
        "Use 'scrape' to read a single page. "
        "Use 'extract' for structured JSON extraction from pages. "
        "Use 'map' to discover URLs from sitemaps. "
        "Use 'crawl' for deep site crawling. "
        "The server learns which scraping method works best for each domain."
    ),
    lifespan=service_lifespan,
    session_state_store=redis_store,
    mask_error_details=True,  # Hide internal errors from clients for security
)

# Add middleware in order: error handling first, then caching
mcp.add_middleware(ErrorHandlingMiddleware(
    include_traceback=True,  # Log full tracebacks server-side
    transform_errors=True,    # Convert exceptions to MCP errors
))

# Add response caching middleware with Redis backend
# Note: Streaming endpoints are automatically excluded by FastMCP
# Note: Using explicit allowlist for tools - dynamic operations (map, crawl)
#       are excluded to ensure fresh data on each call
if redis_store:
    try:
        mcp.add_middleware(ResponseCachingMiddleware(
            cache_storage=redis_store,
            call_tool_settings={
                "enabled": True,
                "ttl": SCRAPE_CACHE_TTL,
                "included_tools": [  # Explicit allowlist - cache only stable operations
                    "research",
                    "search",
                    "scrape",
                    "extract",
                    "docs_fetch_docs",
                    "docs_list_sources",
                    "list_schemas",
                    "domains",
                    "clear_blacklist",
                ],
                # Excluded from caching (fresh results each call):
                # - map: Sitemaps change frequently, need fresh discovery
                # - crawl: Dynamic link discovery, content changes
                # - reset: Must always execute
            },
            list_tools_settings={"enabled": True, "ttl": SEARCH_CACHE_TTL},
        ))
        cached_tools = ", ".join([
            "research", "search", "scrape", "extract",
            "docs_fetch_docs", "docs_list_sources", "list_schemas", "domains", "clear_blacklist"
        ])
        logger.info(f"Response caching enabled ({SCRAPE_CACHE_TTL}s)")
        logger.info(f"Cached tools: {cached_tools}")
        logger.info(f"Uncached: map, crawl, reset (always fresh)")
    except Exception as e:
        logger.warning(f"Failed to add caching middleware: {e}")


# ========== CORS & HEALTH CHECK ==========

_original_http_app = mcp.http_app


def http_app_with_middleware(**kwargs):
    """Add CORS and health check to the underlying Starlette app

    Note: We don't add custom streaming headers middleware because:
    1. FastMCP already handles HTTP streaming headers correctly
    2. Custom middleware on top of streaming causes buffering issues
    """
    app = _original_http_app(**kwargs)

    # Add CORS if not already present
    if not any(m.cls == CORSMiddleware for m in app.user_middleware):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add health check endpoint
    async def health_check(request):
        return JSONResponse({"status": "healthy", "server": "mcp-research-server"})

    app.add_route("/health", health_check, methods=["GET"])
    return app


mcp.http_app = http_app_with_middleware


# ========== TOOL REGISTRATION ==========

# Import tool functions from modular structure
from .tools.web_tools import research, search, scrape, extract, list_schemas, process_html
from .tools.crawl_tools import map, crawl
from .tools.docs_tools import docs_list_sources, docs_fetch_docs
from .tools.admin_tools import domains, stats, reset, clear_blacklist
from .tools.proxy_tools import proxy_status, proxy_test, proxy_rotate
from .tools.vision_tools import analyze_image

# Register all tools with FastMCP
mcp.add_tool(research)
mcp.add_tool(search)
mcp.add_tool(scrape)
mcp.add_tool(extract)
mcp.add_tool(list_schemas)
mcp.add_tool(map)
mcp.add_tool(crawl)
mcp.add_tool(docs_list_sources)
mcp.add_tool(docs_fetch_docs)
mcp.add_tool(domains)
mcp.add_tool(stats)
mcp.add_tool(reset)
mcp.add_tool(clear_blacklist)
mcp.add_tool(proxy_status)
mcp.add_tool(proxy_test)
mcp.add_tool(proxy_rotate)
mcp.add_tool(process_html)
mcp.add_tool(analyze_image)


# ========== ASGI APP (for uvicorn --workers) ==========

app = mcp.http_app(stateless_http=True)


# ========== SERVER ENTRY POINT ==========

if __name__ == "__main__":
    port = settings.port
    host = settings.host

    logger.info(f"MCP HTTP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")
    logger.info(f"Via Caddy+Tailscale: http://<your-tailscale-ip>/mcp")
    logger.info(f"Via MagicDNS+Caddy: https://<hostname>.<tailnet>.ts.net/mcp")
    logger.info(f"Session state: Redis @ {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Caching: enabled (search: {SEARCH_CACHE_TTL}s, scrape: {SCRAPE_CACHE_TTL}s)")
    logger.info(f"Web tools: research, search, scrape, extract, list_schemas, map, crawl")
    logger.info(f"Vision tools: analyze_image (Florence-2, lazy-loaded)")
    logger.info(f"Admin tools: domains, stats, reset, clear_blacklist")
    logger.info(f"Docs tools: docs_list_sources, docs_fetch_docs")
    logger.info(f"Proxy tools: proxy_status, proxy_test, proxy_rotate")

    # Log proxy status
    from .utils.proxy import get_proxy_manager
    proxy_mgr = get_proxy_manager()
    proxy_stats = proxy_mgr.get_stats()
    if proxy_stats["enabled"]:
        logger.info(f"Proxy: ENABLED ({proxy_stats['proxy_count']} proxy(ies), rotation={proxy_stats['rotation']})")
    else:
        logger.info("Proxy: disabled")

    mcp.run(transport="http", host=host, port=port)
