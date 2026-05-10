"""MCP Server Tools - Modular organization

This package exports all MCP tools organized by functionality:
- Web tools: research, search, scrape, extract, list_schemas
- Crawl tools: map, crawl
- Docs tools: docs_list_sources, docs_fetch_docs
- Admin tools: domains, stats, reset, clear_blacklist
- Proxy tools: proxy_status, proxy_test, proxy_rotate
"""

from .web_tools import research, search, scrape, extract, list_schemas
from .crawl_tools import map, crawl
from .docs_tools import docs_list_sources, docs_fetch_docs
from .admin_tools import domains, stats, reset, clear_blacklist
from .proxy_tools import proxy_status, proxy_test, proxy_rotate

__all__ = [
    # Web tools
    "research",
    "search",
    "scrape",
    "extract",
    "list_schemas",
    # Crawl tools
    "map",
    "crawl",
    # Docs tools
    "docs_list_sources",
    "docs_fetch_docs",
    # Admin tools
    "domains",
    "stats",
    "reset",
    "clear_blacklist",
    # Proxy tools
    "proxy_status",
    "proxy_test",
    "proxy_rotate",
]
