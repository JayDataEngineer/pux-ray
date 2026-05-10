"""Services module for MCP research server

Core services used by the MCP server:
- search_service: SearXNG multi-engine search
- scrape_service: Unified scraping with method routing
- crawl_service: URL mapping and deep crawling
- content_cleaner: HTML to Markdown conversion

Legacy services (used by Celery tasks, not core MCP server):
- rate_limit_service: Redis-based rate limiting (replaced by utils.rate_limiter)
- cache_service: Redis-based caching (replaced by FastMCP middleware)
"""

from .search_service import get_search_service
from .scrape_service import get_scrape_service
from .crawl_service import get_map_crawl_service
from .content_cleaner import get_content_cleaner

# Note: Legacy services are kept for Celery tasks but no longer used by MCP tools
# Import directly if needed: from ..services.cache_service import get_cache_service
# Import directly if needed: from ..services.rate_limit_service import get_rate_limit_service

__all__ = [
    "get_search_service",
    "get_scrape_service",
    "get_map_crawl_service",
    "get_content_cleaner",
]
