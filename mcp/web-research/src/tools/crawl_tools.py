"""Crawling Tools

Tools for discovering URLs and deep crawling websites.
- map: Discover URLs from sitemaps/Common Crawl
- crawl: Deep crawl with BFS or Best-First strategy
"""

from typing import Annotated, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field
import json


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


async def map(
    domain: Annotated[str, Field(
        description="Domain to map (e.g., 'example.com' or 'https://example.com')",
        min_length=3
    )],
    source: Annotated[Literal["sitemap", "cc", "sitemap+cc"], Field(
        description="URL source: sitemap (fast), cc (Common Crawl), or sitemap+cc (both)"
    )] = "sitemap+cc",
    pattern: Annotated[str, Field(
        description="URL pattern filter (e.g., '*/blog/*' for blog posts, '*' for all)"
    )] = "*",
    max_urls: Annotated[int | None, Field(
        description="Maximum URLs to return (None = unlimited)"
    )] = None,
    extract_head: Annotated[bool, Field(
        description="Extract metadata from <head> section (slower but richer)"
    )] = False,
    query: Annotated[str | None, Field(
        description="Optional search query for BM25 relevance scoring"
    )] = None,
    score_threshold: Annotated[float | None, Field(
        description="Minimum BM25 relevance score (0.0-1.0) when using query"
    )] = None,
    ctx: Context | None = None
) -> dict:
    """Discover URLs from a domain using sitemaps or Common Crawl

    This is a URL DISCOVERY tool - it finds URLs without crawling them.
    Use this BEFORE scraping to understand a site's structure.

    WORKFLOW:
    1. Call map_domain to discover URLs (e.g., all blog posts)
    2. Filter URLs by pattern, metadata, or relevance score
    3. Call scrape_url or crawl_site on selected URLs

    USE CASES:
    - Find all documentation pages: pattern="*/docs/*"
    - Discover blog posts: pattern="*/blog/*"
    - Find product pages: pattern="*/product/*"
    - Relevance search: query="python tutorial" with score_threshold=0.3

    SOURCES:
    - sitemap: Fast XML sitemap parsing (100-1000 URLs/second)
    - cc: Common Crawl dataset (50-500 URLs/second)
    - sitemap+cc: Both sources for maximum coverage

    Returns:
        Dictionary with domain, total URLs found, and list of URLs with metadata

    Note:
        Results are NOT cached - each call performs fresh sitemap discovery.
        This ensures you get the latest URL structure as sitemaps change frequently.
    """
    from ..services.crawl_service import MapConfig

    if ctx:
        await ctx.info(f"Mapping domain: {domain} (source={source})")

    crawl_svc = ctx.lifespan_context.get("crawl_service")
    if not crawl_svc:
        raise ToolError("Crawl service not available")

    config = MapConfig(
        source=source,
        pattern=pattern,
        extract_head=extract_head,
        max_urls=max_urls,
        query=query,
        scoring_method="bm25" if query else None,
        score_threshold=score_threshold,
        filter_nonsense=True,
    )

    result = await crawl_svc.map_domain(domain, config)

    # Format output
    urls_summary = []
    for url_entry in result.urls[:50]:  # Limit output to first 50
        url_info = {"url": url_entry.get("url", "")}
        if url_entry.get("relevance_score") is not None:
            url_info["score"] = round(url_entry["relevance_score"], 3)
        if url_entry.get("head_data"):
            head = url_entry["head_data"]
            if head.get("title"):
                url_info["title"] = head["title"]
            if head.get("meta", {}).get("description"):
                url_info["description"] = head["meta"]["description"][:100]
        urls_summary.append(url_info)

    if ctx:
        await ctx.info(f"Discovered {result.valid_urls} valid URLs (total: {result.total_urls})")

    return {
        "domain": result.domain,
        "source_used": result.source_used,
        "total_urls": result.total_urls,
        "valid_urls": result.valid_urls,
        "urls": urls_summary,
        "_note": f"Showing first {len(urls_summary)} URLs. Use smaller max_urls or pattern filters for targeted discovery." if result.valid_urls > 50 else "",
    }


async def crawl(
    url: Annotated[str, Field(
        description="Starting URL to crawl"
    )],
    max_depth: Annotated[int, Field(
        description="Maximum depth to crawl (1-5)"
    )] = 2,
    max_pages: Annotated[int, Field(
        description="Maximum pages to crawl (1-200)"
    )] = 50,
    include_external: Annotated[bool, Field(
        description="Follow links to external domains"
    )] = False,
    pattern: Annotated[str | None, Field(
        description="Optional URL pattern filter (e.g., '*/docs/*')"
    )] = None,
    word_count_threshold: Annotated[int, Field(
        description="Minimum word count for pages (50-1000)"
    )] = 100,
    # Filter chain options
    include_patterns: Annotated[Optional[str], Field(
        description="URL patterns to include (comma-separated: '*api*,*reference*')"
    )] = None,
    exclude_patterns: Annotated[Optional[str], Field(
        description="URL patterns to exclude (comma-separated: '*v1*,*old*')"
    )] = None,
    # Best-First strategy options
    strategy: Annotated[Literal["bfs", "best_first"], Field(
        description="Crawling strategy: bfs (systematic) or best_first (prioritize relevant pages)"
    )] = "bfs",
    keywords: Annotated[Optional[str], Field(
        description="Keywords for relevance scoring (comma-separated: 'api,tutorial')"
    )] = None,
    ctx: Context | None = None
) -> dict:
    """Deep crawl a site following links (BFS or Best-First strategy)

    This is a DEEP CRAWL tool - it discovers and crawls pages by following links.
    Use this AFTER map_domain when you need actual page content.

    WORKFLOW:
    1. Call map_domain to discover URLs (optional but recommended)
    2. Call crawl_site with starting URL to crawl linked pages
    3. Review crawled pages and extract specific URLs of interest

    STRATEGIES:
    - bfs (default): Systematic breadth-first exploration
    - best_first: Prioritize pages matching keywords (requires keywords parameter)

    FILTERING:
    - pattern: Simple URL pattern (e.g., '*/docs/*')
    - include_patterns: Multiple patterns to include (e.g., ['*api*', '*reference*'])
    - exclude_patterns: Multiple patterns to exclude (e.g., ['*deprecated*', '*v1*'])

    USE CASES:
    - Crawl documentation with filters: url="https://docs.example.com", include_patterns=["*api*"]
    - Best-First for specific topics: url="https://docs.example.com", strategy="best_first", keywords=["api", "tutorial"]
    - Exclude old versions: url="https://docs.example.com", exclude_patterns=["*v1*", "*deprecated*"]

    Returns:
        Dictionary with crawl stats and list of crawled pages with content

    Warning:
        Deep crawling is resource-intensive. Start with low max_depth (2)
        and max_pages (20) for testing, then increase as needed.

    Note:
        Results are NOT cached - each call performs fresh crawling.
        This ensures you get the latest content, but be aware that repeated
        calls to the same URL will re-crawl the site.
    """
    from ..services.crawl_service import CrawlConfig

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

    # Robust parameter parsing - handle LLMs that serialize lists as JSON strings
    def _parse_flex(value: str | None) -> list[str] | None:
        """Parse flexible list parameter from string.

        Handles:
        - JSON arrays: '["a","b"]' -> ['a', 'b']
        - Comma-separated: 'a,b' -> ['a', 'b']
        - Single value: 'a' -> ['a']
        - None: None
        """
        if value is None:
            return None

        value = value.strip()

        # Try parsing as JSON array first (LLM bug scenario)
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass

        # Fall back to comma-separated
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

        # Single value
        return [value]

    include_patterns = _parse_flex(include_patterns)
    exclude_patterns = _parse_flex(exclude_patterns)
    keywords = _parse_flex(keywords)

    if ctx:
        await ctx.info(f"Deep crawling: {url} (strategy={strategy}, max_depth={max_depth}, max_pages={max_pages})")

    crawl_svc = ctx.lifespan_context.get("crawl_service")
    if not crawl_svc:
        raise ToolError("Crawl service not available")

    # Validate best_first requires keywords
    if strategy == "best_first" and not keywords:
        raise ToolError("best_first strategy requires keywords parameter")

    # Ensure word_count_threshold has a valid value
    if word_count_threshold is None:
        word_count_threshold = 100

    config = CrawlConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        pattern=pattern,
        only_text=True,
        word_count_threshold=word_count_threshold,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        strategy=strategy,
        keywords=keywords,
    )

    result = await crawl_svc.crawl_site(url, config)

    # Format output - limit content size
    pages_summary = []
    for page in result.pages:
        page_info = {
            "url": page["url"],
            "success": page["success"],
            "depth": page.get("depth", 0),
        }
        if page.get("title"):
            page_info["title"] = page["title"]
        if page.get("content"):
            content = page["content"]
            # Truncate long content
            if len(content) > 2000:
                page_info["content"] = content[:2000] + f"... (truncated, was {len(content)} chars)"
                page_info["truncated"] = True
            else:
                page_info["content"] = content
        pages_summary.append(page_info)

    if ctx:
        await ctx.info(f"Crawled {result.total_crawled} pages ({result.successful} successful, {result.failed} failed)")

    # Check for blocking - if all pages failed with block errors, provide clear message
    if result.failed > 0 and result.pages:
        block_errors = [p.get("error") for p in result.pages if p.get("error")]
        if block_errors and any("blocked" in e.lower() or "captcha" in e.lower() or "rate" in e.lower() for e in block_errors if e):
            # Count block types
            blocked_count = sum(1 for e in block_errors if e and ("blocked" in e.lower() or "captcha" in e.lower()))
            rate_limited = sum(1 for e in block_errors if e and "rate" in e.lower())

            if blocked_count > 0 or rate_limited > 0:
                error_msg = "Unable to crawl - "
                if blocked_count > 0:
                    error_msg += f"site is blocking automated crawlers (CAPTCHA/access denied). "
                if rate_limited > 0:
                    error_msg += f"rate limiting detected. "

                error_msg += "Try scrape_url instead which has SeleniumBase fallback."
                if ctx:
                    await ctx.info(error_msg)

                return {
                    "start_url": url,
                    "total_crawled": result.total_crawled,
                    "successful": result.successful,
                    "failed": result.failed,
                    "pages": pages_summary,
                    "block_detected": True,
                    "error_message": error_msg.strip(),
                }

    return {
        "start_url": url,
        "total_crawled": result.total_crawled,
        "successful": result.successful,
        "failed": result.failed,
        "pages": pages_summary,
    }
