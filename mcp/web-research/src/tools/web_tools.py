"""Web Research Tools

Tools for searching, scraping, and researching the web.
- research: Search + scrape top results in one call (recommended)
- search: Search using multiple search engines
- scrape: Scrape a URL and extract clean markdown
- extract: Extract structured JSON data using pre-built schemas
- list_schemas: List available extraction schemas
"""

import asyncio
from typing import Annotated, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from loguru import logger
from pydantic import Field


# Cache TTL from settings
from ..settings import get_settings
_settings = get_settings()
SEARCH_CACHE_TTL = _settings.search_cache_ttl
SCRAPE_CACHE_TTL = _settings.scrape_cache_ttl


def _is_url_blacklisted(url: str) -> bool:
    """Check if a URL is blacklisted for security reasons.

    Blocks access to:
    - localhost and loopback addresses
    - Private network IPs (RFC 1918)
    - Link-local addresses
    - AWS metadata service
    - Other internal services

    Returns True if the URL should be blocked.
    """
    from urllib.parse import urlparse
    import ipaddress

    try:
        parsed = urlparse(url)
        netloc = parsed.netloc

        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]

        # Remove www. prefix for hostname checking
        hostname = netloc[4:] if netloc.startswith("www.") else netloc

        # Block localhost variants
        blocked_hostnames = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "ip6-localhost",
            "ip6-loopback",
        }
        if hostname.lower() in blocked_hostnames:
            return True

        # Block AWS metadata service
        if hostname == "169.254.169.254":
            return True

        # Try to parse as IP address
        try:
            ip = ipaddress.ip_address(hostname)

            # Block private IP ranges (RFC 1918)
            if ip.is_private:
                return True

            # Block link-local addresses
            if ip.is_link_local:
                return True

            # Block reserved addresses
            if ip.is_reserved:
                return True

            # Block loopback addresses (in case hostname was an IP)
            if ip.is_loopback:
                return True

        except ValueError:
            # Not an IP address, continue checking
            pass

        # Block internal TLDs
        if hostname.endswith(".local") or hostname.endswith(".internal"):
            return True

        return False

    except Exception:
        # If we can't parse the URL, err on the side of caution and block
        return True


async def research(
    query: Annotated[str, Field(
        description="Research query — searches the web and scrapes the top results",
        min_length=1,
        max_length=500
    )],
    max_results: Annotated[int, Field(
        description="Number of search results to scrape (1-5)",
        ge=1,
        le=5
    )] = 3,
    depth: Annotated[Literal["quick", "deep"], Field(
        description="quick=1 search page (~10 results), deep=2 pages (~20 results)"
    )] = "quick",
    rerank: Annotated[bool, Field(
        description="Apply flash re-ranking to prioritize relevant results. Set false to get raw search engine ordering."
    )] = True,
    ctx: Context | None = None
) -> dict:
    """Search the web and scrape the top results in one call.

    This is the fastest way to research a topic. It searches multiple
    engines, then scrapes the top results concurrently so you get
    actual page content — not just snippets.

    Returns results with title, url, snippet, and page content (capped at ~5K chars per result).
    Use 'scrape' on a specific URL if you need the full page content.

    Args:
        query: What to search for
        max_results: How many results to scrape (default 3)
        depth: "quick" for 1 page of results, "deep" for 2 pages

    Returns:
        Dictionary with query and list of results, each containing
        title, url, snippet, and scraped content.
    """
    if ctx:
        await ctx.info(f"Researching: {query}")

    search_svc = ctx.lifespan_context.get("search_service")
    scrape_svc = ctx.lifespan_context.get("scrape_service")
    if not search_svc or not scrape_svc:
        raise ToolError("Search or scrape service not available")

    # Step 1: Search
    pages = 1 if depth == "quick" else 2
    search_result = await search_svc.search(
        query=query,
        pages=pages,
        exclude_blacklist=True,
        top_k=max_results,
        rerank=rerank,
    )

    if not search_result.results:
        return {"query": query, "results": []}

    # Step 2: Scrape top results in parallel
    from ..models.unified import ScrapeRequest

    async def _scrape_one(result_item) -> dict:
        try:
            req = ScrapeRequest(url=result_item.url, text_only=True)
            resp = await scrape_svc.scrape(req)
            content = resp.content if resp.success else None
            # Cap individual results at ~10K chars to protect context windows.
            # PruningContentFilter already removes noise; this catches edge cases.
            if content and len(content) > 10000:
                content = content[:10000] + f"\n\n... ({len(resp.content)} chars total, use scrape for full page)"
            return {
                "title": result_item.title,
                "url": result_item.url,
                "snippet": result_item.snippet,
                "content": content,
                "error": resp.error if not resp.success else None,
            }
        except Exception as e:
            logger.debug(f"research: failed to scrape {result_item.url}: {e}")
            return {
                "title": result_item.title,
                "url": result_item.url,
                "snippet": result_item.snippet,
                "content": None,
                "error": str(e)[:200],
            }

    scraped = await asyncio.gather(
        *[_scrape_one(r) for r in search_result.results[:max_results]]
    )

    # Filter to results that actually got content
    successful = [r for r in scraped if r["content"]]

    if ctx:
        await ctx.info(
            f"Scraped {len(successful)}/{len(scraped)} results"
        )

    return {"query": query, "results": list(scraped)}


async def search(
    query: Annotated[str, Field(
        description="Search query string",
        min_length=1,
        max_length=500
    )],
    pages: Annotated[int, Field(
        description="Number of search result pages to fetch (1-5)",
        ge=1,
        le=5
    )] = 1,
    exclude_blacklist: Annotated[bool, Field(
        description="Exclude blacklisted domains from results"
    )] = True,
    top_k: Annotated[int | None, Field(
        description="Maximum number of results to return (None = all results)"
    )] = 5,
    rerank: Annotated[bool, Field(
        description="Apply flash re-ranking based on query relevance"
    )] = True,
    time_filter: Annotated[Literal["day", "week", "month", "year"] | None, Field(
        description="Filter results by time: day (24h), week (7d), month (30d), year (365d)"
    )] = None,
    ctx: Context | None = None
) -> dict:
    """Search the web using multiple search engines

    Returns titles, URLs, and short snippets. Use the 'research' tool
    instead if you want full page content along with results.

    Args:
        query: Search query string (1-500 characters)
        pages: Number of search result pages to fetch (default: 1)
        exclude_blacklist: Exclude blacklisted domains from results
        top_k: Maximum number of results to return (default: 5)
        rerank: Apply flash re-ranking to prioritize relevant results
        time_filter: Filter by time - day (24h), week (7d), month (30d), year (365d)

    Returns:
        Dictionary with query and list of results (title, url, snippet)
    """
    if ctx:
        await ctx.info(f"Searching for: {query}")

    search_svc = ctx.lifespan_context.get("search_service")
    if not search_svc:
        raise ToolError("Search service not available")

    result = await search_svc.search(
        query=query,
        pages=pages,
        exclude_blacklist=exclude_blacklist,
        top_k=top_k,
        rerank=rerank,
        time_filter=time_filter
    )

    if ctx:
        await ctx.info(f"Found {result.total_results} results")

    return {
        "query": query,
        "total_results": result.total_results,
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in result.results
        ]
    }


async def scrape(
    url: Annotated[str, Field(description="URL to scrape")],
    method: Annotated[Literal["httpx", "crawl4ai", "selenium", "pdf"] | None, Field(
        description="Force specific scraping method"
    )] = None,
    css_selector: Annotated[str | None, Field(
        description="Optional CSS selector for targeted content extraction"
    )] = None,
    text_only: Annotated[bool, Field(
        description="Disable images for faster loading (Crawl4AI only)"
    )] = False,
    ctx: Context | None = None
) -> dict:
    """Scrape a URL and extract clean markdown content

    The server learns which scraping method works best per domain and
    automatically uses it on future requests.

    Args:
        url: URL to scrape (must be a valid HTTP/HTTPS URL)
        method: Force specific scraping method (crawl4ai, selenium, pdf)
        css_selector: Optional CSS selector for targeted content extraction
        text_only: Disable images for faster loading (Crawl4AI only)

    Returns:
        Dictionary with url, title, content (markdown), success, and error

    Security:
        Internal and private IPs are blocked (localhost, 127.0.0.1, 10.*,
        172.16-31.*, 192.168.*, 169.254.*). Only public URLs can be scraped.
    """
    # Validate URL
    if not url.startswith(("http://", "https://", "file://")):
        raise ToolError("URL must start with http://, https://, or file://")

    # Security: Check for blacklisted URLs (internal/private IPs)
    if url.startswith(("http://", "https://")) and _is_url_blacklisted(url):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        raise ToolError(
            f"URL is not allowed for security reasons: {parsed.netloc} "
            f"appears to be a private or internal address. "
            f"Only public URLs can be scraped."
        )

    if ctx:
        await ctx.info(f"Scraping: {url}")

    from ..models.unified import ScrapeRequest, ScrapingMethod

    scrape_svc = ctx.lifespan_context.get("scrape_service")
    if not scrape_svc:
        raise ToolError("Scrape service not available")

    try:
        request = ScrapeRequest(
            url=url,
            force_method=ScrapingMethod(method) if method else None,
            css_selector=css_selector,
            text_only=text_only
        )
    except ValueError as e:
        raise ToolError(f"Invalid scraping method: {e}")

    try:
        result = await scrape_svc.scrape(request)

        response = {
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "success": result.success,
        }

        if not result.success and result.error:
            response["error"] = result.error
            if ctx:
                await ctx.error(f"Scrape failed: {result.error}")

        return response
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error in scrape for {url}: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")

        error_msg = str(e) if len(str(e)) < 200 else "Internal error during scraping"

        if ctx:
            await ctx.error(f"Scrape error: {error_msg}")

        return {
            "url": url,
            "success": False,
            "title": None,
            "content": "",
            "error": error_msg
        }


async def extract(
    url: Annotated[str, Field(description="URL to scrape with structured extraction")],
    schema_type: Annotated[Literal["ecommerce", "news", "jobs", "blog", "social", "products"], Field(
        description="Pre-built schema type for extraction"
    )] = "ecommerce",
    custom_selector: Annotated[str | None, Field(
        description="Custom CSS selector to override the base selector"
    )] = None,
    bypass_cache: Annotated[bool, Field(
        description="Bypass cache and fetch fresh data"
    )] = True,
    ctx: Context | None = None
) -> dict:
    """Extract structured data from a web page using pre-built schemas

    Uses CSS extraction schemas to extract structured JSON data from web
    pages WITHOUT using LLMs. Much faster and cheaper than LLM-based
    extraction.

    SCHEMA TYPES:
    - ecommerce: Products (name, price, rating, availability, image, url)
    - news: Articles (headline, author, date, content, category, summary)
    - jobs: Listings (title, company, location, salary, description)
    - blog: Posts (title, author, date, content, tags, excerpt)
    - social: Social posts (username, content, timestamp, likes, shares)
    - products: Product catalog multi-item extraction

    Args:
        url: URL to extract data from
        schema_type: Pre-built schema type for extraction
        custom_selector: Custom CSS selector to override the base selector
        bypass_cache: Bypass cache and fetch fresh data

    Returns:
        Dictionary with extracted items as JSON array
    """
    from ..services.crawl_service import StructuredScrapeConfig

    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise ToolError("URL must start with http:// or https://")

    # Security: Check for blacklisted URLs
    if _is_url_blacklisted(url):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        raise ToolError(
            f"URL is not allowed for security reasons: {parsed.netloc} "
            f"appears to be a private or internal address."
        )

    if ctx:
        await ctx.info(f"Extracting {schema_type} data from: {url}")

    crawl_svc = ctx.lifespan_context.get("crawl_service")
    if not crawl_svc:
        raise ToolError("Crawl service not available")

    config = StructuredScrapeConfig(
        schema_type=schema_type,
        custom_selector=custom_selector,
        bypass_cache=bypass_cache,
    )

    result = await crawl_svc.scrape_structured(url, config)

    if ctx:
        if result.success:
            await ctx.info(f"Extracted {result.item_count} items")
        else:
            await ctx.error(f"Extraction failed: {result.error}")

    return {
        "url": result.url,
        "success": result.success,
        "schema_type": result.schema_type,
        "item_count": result.item_count,
        "items": result.items,
        "title": result.title,
        "error": result.error,
    }


async def list_schemas(ctx: Context | None = None) -> dict:
    """List all available structured extraction schemas

    Returns information about pre-built schemas available for
    the extract tool, including field counts and descriptions.

    Returns:
        Dictionary with list of available schemas
    """
    from ..services.extraction_schemas import list_schemas as list_available_schemas

    schemas = list_available_schemas()

    return {
        "total": len(schemas),
        "schemas": schemas,
    }


async def process_html(
    html: Annotated[str, Field(
        description="Raw HTML content to clean and convert to markdown"
    )],
    url: Annotated[str, Field(
        description="Source URL for context (used by cleaning heuristics)"
    )] = "",
    css_selector: Annotated[str | None, Field(
        description="Optional CSS selector for targeted content extraction"
    )] = None,
    ctx: Context | None = None
) -> dict:
    """Process raw HTML content and convert to clean markdown.

    For AI browser agents: send the page HTML from your browser and get back
    clean, LLM-ready markdown. Uses the same cleaning pipeline as the scrape
    tool (noise removal, content extraction, 50K char cap).

    Args:
        html: Raw HTML content from the page DOM
        url: Source URL (helps the cleaner make better extraction decisions)
        css_selector: Optional CSS selector for targeted extraction

    Returns:
        Dictionary with content (clean markdown), success, and content_length
    """
    if ctx:
        await ctx.info(f"Processing HTML: {len(html)} chars from {url or 'unknown'}")

    if not html or not html.strip():
        raise ToolError("html parameter cannot be empty")

    from ..scrapers.base import postprocess_markdown

    scrape_svc = ctx.lifespan_context.get("scrape_service")
    if not scrape_svc:
        raise ToolError("Scrape service not available")

    cleaner = scrape_svc.cleaner
    clean_markdown = cleaner.clean(html, url=url, css_selector=css_selector)
    clean_markdown = postprocess_markdown(clean_markdown)

    return {
        "success": bool(clean_markdown),
        "content": clean_markdown,
        "content_length": len(clean_markdown),
        "url": url,
    }
