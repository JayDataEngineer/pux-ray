"""Map and crawl service using Crawl4AI's URL seeding and deep crawling

Provides:
- AsyncUrlSeeder: Fast URL discovery from sitemaps/Common Crawl
- BFSDeepCrawlStrategy: Controlled deep crawling with max depth/pages
- JsonCssExtractionStrategy: Structured data extraction from pages

Anti-detection features:
- Stealth mode enabled (anti-fingerprinting)
- Random user agents
- Delays between requests
- Block detection (403, CAPTCHA, rate limits)
"""

from dataclasses import dataclass, field
from typing import Literal
from loguru import logger
import json
import re


# Block detection patterns
BLOCK_PATTERNS = {
    "captcha": re.compile(r"captcha|challenge|prove.?human|robot.?check", re.I),
    "blocked": re.compile(r"access.?denied|forbidden|blocked|unavailable", re.I),
    "rate_limit": re.compile(r"rate.?limit|too.?many.?requests|429", re.I),
}


def detect_blocking(page_content: str, status_code: int = None) -> str | None:
    """Detect if we've been blocked by the website

    Returns:
        Error message describing the block type, or None if not blocked
    """
    if status_code:
        if status_code == 403:
            return "Blocked: HTTP 403 Forbidden"
        if status_code == 429:
            return "Rate limited: Too many requests"
        if status_code >= 500:
            return f"Server error: HTTP {status_code}"

    content_lower = page_content.lower()[:2000]  # Check first 2000 chars
    for block_type, pattern in BLOCK_PATTERNS.items():
        if pattern.search(content_lower):
            if block_type == "captcha":
                return "Blocked: CAPTCHA challenge detected"
            if block_type == "blocked":
                return "Blocked: Access denied"
            if block_type == "rate_limit":
                return "Rate limited: Too many requests"

    return None


@dataclass
class MapConfig:
    """Configuration for domain mapping (URL discovery)"""
    source: Literal["sitemap", "cc", "sitemap+cc"] = "sitemap+cc"
    pattern: str = "*"  # URL pattern filter (e.g., "*/blog/*")
    extract_head: bool = False  # Extract metadata from <head>
    live_check: bool = False  # Verify URLs are accessible
    max_urls: int | None = None  # Maximum URLs to return (None = unlimited)
    concurrency: int = 10  # Parallel workers
    query: str | None = None  # Search query for BM25 scoring
    scoring_method: str | None = None  # "bm25" for relevance scoring
    score_threshold: float | None = None  # Minimum BM25 score
    filter_nonsense: bool = True  # Filter utility URLs


@dataclass
class CrawlConfig:
    """Configuration for deep crawling"""
    max_depth: int = 2  # Maximum depth to crawl
    max_pages: int = 50  # Maximum pages to crawl
    include_external: bool = False  # Follow external links
    pattern: str | None = None  # URL pattern filter
    only_text: bool = True  # Extract only text content
    word_count_threshold: int = 100  # Minimum word count

    # Filter chain options
    include_patterns: list[str] | None = None  # URL patterns to include (e.g., ["*api*", "*reference*"])
    exclude_patterns: list[str] | None = None  # URL patterns to exclude (e.g., ["*deprecated*", "*v1*"])

    # Best-First strategy options
    strategy: Literal["bfs", "best_first"] = "bfs"  # Crawling strategy
    keywords: list[str] | None = None  # Keywords for relevance scoring (best_first only)


@dataclass
class MapResult:
    """Result from domain mapping"""
    domain: str
    total_urls: int
    valid_urls: int
    urls: list[dict] = field(default_factory=list)
    source_used: str = ""


@dataclass
class CrawlResult:
    """Result from deep crawling"""
    domain: str
    total_crawled: int
    successful: int
    failed: int
    pages: list[dict] = field(default_factory=list)


@dataclass
class StructuredScrapeConfig:
    """Configuration for structured scraping"""
    schema_type: Literal["ecommerce", "news", "jobs", "blog", "social", "products"] = "ecommerce"
    custom_selector: str | None = None  # Override baseSelector
    bypass_cache: bool = True  # Bypass cache for fresh data


@dataclass
class StructuredResult:
    """Result from structured scraping"""
    url: str
    success: bool
    schema_type: str
    items: list  # List of extracted items
    item_count: int
    error: str | None = None
    title: str | None = None
    raw_html: str | None = None  # For debugging


class MapCrawlService:
    """Service for mapping and crawling websites"""

    def __init__(self):
        self._seeder = None

    async def _get_seeder(self):
        """Get or create AsyncUrlSeeder instance"""
        if self._seeder is None:
            from crawl4ai import AsyncUrlSeeder
            self._seeder = AsyncUrlSeeder()
        return self._seeder

    async def map_domain(
        self,
        domain: str,
        config: MapConfig,
    ) -> MapResult:
        """Discover URLs from a domain using sitemap/Common Crawl

        Args:
            domain: Domain to map (e.g., "example.com" or "https://example.com")
            config: Configuration for URL discovery

        Returns:
            MapResult with discovered URLs
        """
        from crawl4ai import SeedingConfig
        from urllib.parse import urlparse
        from ..scrapers.base import scrape_selenium
        from ..utils import extract_domain
        import xml.etree.ElementTree as ET

        # Normalize domain to URL format
        if not domain.startswith(("http://", "https://")):
            domain_url = f"https://{domain}"
        else:
            domain_url = domain
            parsed = urlparse(domain)
            domain = parsed.netloc

        logger.info(f"Mapping domain: {domain} with source={config.source}")

        seeder = await self._get_seeder()

        # Build SeedingConfig
        seeding_config = SeedingConfig(
            source=config.source,
            pattern=config.pattern if config.pattern != "*" else None,
            extract_head=config.extract_head,
            live_check=config.live_check,
            max_urls=config.max_urls,
            concurrency=config.concurrency,
            query=config.query,
            scoring_method=config.scoring_method,
            score_threshold=config.score_threshold,
            filter_nonsense_urls=config.filter_nonsense,
            verbose=True,
        )

        try:
            # Discover URLs
            urls = await seeder.urls(domain, seeding_config)

            # Process results
            # Accept URLs with status="valid" or status="unknown"
            # "unknown" means not verified via live_check, but still usable
            valid_urls = [u for u in urls if u.get("status") in ("valid", "unknown")]

            logger.info(f"Discovered {len(urls)} total URLs, {len(valid_urls)} valid")

            # Apply max_urls limit if set
            limited_urls = valid_urls[:config.max_urls] if config.max_urls is not None else valid_urls

            return MapResult(
                domain=domain,
                total_urls=len(urls),
                valid_urls=len(valid_urls),
                urls=limited_urls,
                source_used=config.source,
            )

        except Exception as e:
            logger.warning(f"Crawl4AI seeder failed for {domain}: {e}")

            # Fallback: Try to fetch sitemap.xml directly using SeleniumBase
            if config.source in ("sitemap", "sitemap+cc"):
                logger.info(f"Falling back to SeleniumBase sitemap fetch for {domain}")
                return await self._map_domain_selenium_fallback(domain, config)

            # If Common Crawl only requested, no fallback available
            logger.error(f"Cannot map {domain} - Common Crawl failed and no sitemap fallback")
            return MapResult(
                domain=domain,
                total_urls=0,
                valid_urls=0,
                urls=[],
                source_used=config.source,
            )

    async def _map_domain_selenium_fallback(
        self,
        domain: str,
        config: MapConfig,
    ) -> MapResult:
        """Fallback: Fetch and parse sitemap.xml using SeleniumBase

        Args:
            domain: Domain to map
            config: Configuration for URL discovery

        Returns:
            MapResult with discovered URLs
        """
        from ..scrapers.base import scrape_selenium
        from .content_cleaner import ContentCleaner
        import xml.etree.ElementTree as ET
        from urllib.parse import urljoin

        cleaner = ContentCleaner()
        urls = []

        # Common sitemap locations
        sitemap_urls = [
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/wp-sitemap.xml",  # WordPress
        ]

        # Try each sitemap URL
        for sitemap_url in sitemap_urls:
            try:
                logger.info(f"Trying sitemap: {sitemap_url}")

                # Use SeleniumBase to fetch the sitemap
                result = await scrape_selenium(
                    url=sitemap_url,
                    cleaner=cleaner,
                    css_selector=None
                )

                if not result.get("success"):
                    continue

                sitemap_content = result.get("content", "")
                if not sitemap_content:
                    continue

                # Parse XML sitemap
                root = ET.fromstring(sitemap_content.encode() if isinstance(sitemap_content, str) else sitemap_content)

                # Handle namespace
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                has_ns = root.tag.startswith("{http://www.sitemaps.org}")

                # Check if this is a sitemap index
                if has_ns:
                    sitemap_tags = root.findall(".//sm:sitemap/sm:loc", ns)
                    url_tags = root.findall(".//sm:url/sm:loc", ns)
                else:
                    sitemap_tags = root.findall(".//sitemap/loc")
                    url_tags = root.findall(".//url/loc")

                # If sitemap index, we'd need to fetch child sitemaps (skip for now)
                # Just return direct URLs
                for url_tag in url_tags:
                    url = url_tag.text
                    if url and self._url_matches_pattern(url, config.pattern):
                        urls.append({
                            "url": url,
                            "status": "valid",  # We got it from sitemap
                            "source": "sitemap_selenium"
                        })

                if urls:
                    logger.info(f"Selenium fallback found {len(urls)} URLs from {sitemap_url}")
                    break

            except Exception as e:
                logger.debug(f"Failed to fetch {sitemap_url}: {e}")
                continue

        # Filter to max_urls (None = unlimited)
        valid_urls = urls[:config.max_urls] if config.max_urls is not None else urls

        return MapResult(
            domain=domain,
            total_urls=len(urls),
            valid_urls=len(valid_urls),
            urls=valid_urls,
            source_used="sitemap_selenium_fallback",
        )

    def _url_matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a glob pattern"""
        import fnmatch
        if pattern == "*":
            return True
        return fnmatch.fnmatch(url, pattern)

    async def map_many_domains(
        self,
        domains: list[str],
        config: MapConfig,
    ) -> dict[str, MapResult]:
        """Map multiple domains in parallel

        Args:
            domains: List of domains to map
            config: Configuration for URL discovery

        Returns:
            Dictionary mapping domain to MapResult
        """
        from crawl4ai import SeedingConfig

        # Normalize domains
        normalized = []
        for d in domains:
            if not d.startswith(("http://", "https://")):
                normalized.append(f"https://{d}")
            else:
                normalized.append(d)

        seeder = await self._get_seeder()

        seeding_config = SeedingConfig(
            source=config.source,
            pattern=config.pattern if config.pattern != "*" else None,
            extract_head=config.extract_head,
            live_check=config.live_check,
            max_urls=config.max_urls,
            concurrency=config.concurrency,
            query=config.query,
            scoring_method=config.scoring_method,
            score_threshold=config.score_threshold,
            filter_nonsense_urls=config.filter_nonsense,
            verbose=True,
        )

        try:
            # Use many_urls for parallel discovery
            results = await seeder.many_urls(normalized, seeding_config)

            # Convert to MapResult format
            output = {}
            for domain_url, urls in results.items():
                from urllib.parse import urlparse
                parsed = urlparse(domain_url)
                domain = parsed.netloc

                # Accept URLs with status="valid" or status="unknown"
                valid_urls = [u for u in urls if u.get("status") in ("valid", "unknown")]

                # Apply max_urls limit if set
                limited_urls = valid_urls[:config.max_urls] if config.max_urls is not None else valid_urls

                output[domain] = MapResult(
                    domain=domain,
                    total_urls=len(urls),
                    valid_urls=len(valid_urls),
                    urls=limited_urls,
                    source_used=config.source,
                )

            return output

        except Exception as e:
            logger.error(f"Error mapping multiple domains: {e}")
            # Return empty results
            return {d: MapResult(
                domain=d,
                total_urls=0,
                valid_urls=0,
                urls=[],
                source_used=config.source,
            ) for d in domains}

    async def crawl_site(
        self,
        url: str,
        config: CrawlConfig,
    ) -> CrawlResult:
        """Deep crawl a site using BFS or Best-First strategy

        Args:
            url: Starting URL to crawl
            config: Configuration for deep crawling

        Returns:
            CrawlResult with crawled pages
        """
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
        from crawl4ai.cache_context import CacheMode
        from crawl4ai.deep_crawling import (
            BFSDeepCrawlStrategy,
            BestFirstCrawlingStrategy,
        )
        from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
        from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

        # Build strategy based on config
        # Build filter chain from all pattern sources
        filter_chain = None
        filters = []

        # Add simple pattern to filter chain
        if config.pattern:
            filters.append(URLPatternFilter(
                patterns=[config.pattern],
                use_glob=True,
                reverse=False
            ))

        # Add include_patterns
        if config.include_patterns:
            filters.append(URLPatternFilter(
                patterns=config.include_patterns,
                use_glob=True,
                reverse=False  # include these patterns
            ))

        # Add exclude_patterns
        if config.exclude_patterns:
            filters.append(URLPatternFilter(
                patterns=config.exclude_patterns,
                use_glob=True,
                reverse=True  # exclude these patterns
            ))

        # Create FilterChain - always create one (even if empty) to avoid Crawl4AI bug
        # Crawl4AI crashes with AttributeError when filter_chain=None
        filter_chain = FilterChain(filters) if filters else FilterChain([])
        if filters:
            logger.info(f"Using filter chain with {len(filters)} filters")

        if config.strategy == "best_first" and config.keywords:
            logger.info(f"Using Best-First strategy with keywords: {config.keywords}")
            url_scorer = KeywordRelevanceScorer(
                keywords=config.keywords,
                weight=1.0
            )
            deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                url_scorer=url_scorer,
                filter_chain=filter_chain,
            )
        else:
            # BFS strategy (default)
            logger.info(f"Using BFS strategy")

            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                filter_chain=filter_chain,
            )

        # Build crawl config with anti-detection settings
        crawl_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl_strategy,
            only_text=config.only_text,
            word_count_threshold=config.word_count_threshold,
            # Anti-detection: delays between requests
            mean_delay=0.5,  # Average delay between requests (seconds)
            delay_before_return_html=0.3,  # Delay before returning content
            # User agent rotation
            user_agent_mode="random",
            verbose=False,
        )

        # Build browser config with stealth mode
        browser_config = BrowserConfig(
            headless=True,
            enable_stealth=True,  # Anti-fingerprinting
            user_agent_mode="random",
            text_mode=config.only_text,
            verbose=False,
        )

        pages = []
        successful = 0
        failed = 0

        try:
            async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
                result = await crawler.arun(url, config=crawl_config)

                # When using deep_crawl_strategy, arun returns a LIST of CrawlResult
                if isinstance(result, list):
                    # Process list of crawl results from deep crawling
                    for page_result in result:
                        # Check for blocking
                        error_msg = None
                        if not page_result.success:
                            # Get HTML content for block detection
                            page_html = getattr(page_result, 'html', '') or ''
                            status_code = getattr(page_result, 'status_code', None)
                            error_msg = detect_blocking(page_html, status_code)
                            if not error_msg:
                                error_msg = getattr(page_result, 'error_message', 'Crawl failed')

                        page_data = {
                            "url": page_result.url,
                            "success": page_result.success,
                            "title": getattr(page_result, 'title', None) or (
                                page_result.metadata.get('title', '') if hasattr(page_result, 'metadata') and page_result.metadata else ''
                            ),
                            "content": page_result.markdown.raw_markdown if hasattr(page_result, 'markdown') and page_result.markdown else None,
                            "depth": getattr(page_result.metadata, 'get', lambda x: None)('depth', 0) if hasattr(page_result, 'metadata') else 0,
                            "error": error_msg if not page_result.success else None,
                        }
                        pages.append(page_data)
                        if page_result.success:
                            successful += 1
                        else:
                            failed += 1
                            if error_msg and ("blocked" in error_msg.lower() or "captcha" in error_msg.lower() or "rate" in error_msg.lower()):
                                logger.warning(f"Blocking detected at {page_result.url}: {error_msg}")
                elif result.success:
                    # Single page result (no deep crawling happened)
                    page_data = {
                        "url": result.url,
                        "success": result.success,
                        "title": result.metadata.get('title', '') if hasattr(result, 'metadata') and result.metadata else '',
                        "content": result.markdown.raw_markdown if hasattr(result, 'markdown') and result.markdown else None,
                        "depth": 0,
                    }
                    pages.append(page_data)
                    successful += 1
                else:
                    # Check for blocking on failed single result
                    page_html = getattr(result, 'html', '') or ''
                    status_code = getattr(result, 'status_code', None)
                    error_msg = detect_blocking(page_html, status_code) or getattr(result, 'error_message', 'Crawl failed')

                    if error_msg and ("blocked" in error_msg.lower() or "captcha" in error_msg.lower() or "rate" in error_msg.lower()):
                        logger.warning(f"Blocking detected at {url}: {error_msg}")

                    page_data = {
                        "url": result.url,
                        "success": False,
                        "title": "",
                        "content": None,
                        "depth": 0,
                        "error": error_msg,
                    }
                    pages.append(page_data)
                    failed += 1

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        return CrawlResult(
            domain=url,
            total_crawled=len(pages),
            successful=successful,
            failed=failed,
            pages=pages,
        )

    async def scrape_structured(
        self,
        url: str,
        config: StructuredScrapeConfig,
    ) -> StructuredResult:
        """Scrape a URL and extract structured data using schema-based extraction

        Uses JsonCssExtractionStrategy from Crawl4AI to extract structured
        data without LLM costs. Supports pre-built schemas for common page types.

        Args:
            url: URL to scrape
            config: Configuration for structured extraction

        Returns:
            StructuredResult with extracted items
        """
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, JsonCssExtractionStrategy
        from crawl4ai.cache_context import CacheMode
        from .extraction_schemas import get_schema

        logger.info(f"Structured scraping: {url} (schema={config.schema_type})")

        # Get the pre-built schema
        schema = get_schema(config.schema_type)
        if not schema:
            return StructuredResult(
                url=url,
                success=False,
                schema_type=config.schema_type,
                items=[],
                item_count=0,
                error=f"Unknown schema type: {config.schema_type}",
            )

        # Allow custom selector override
        if config.custom_selector:
            schema = schema.copy()
            schema["baseSelector"] = config.custom_selector

        # Build extraction strategy
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        # Build crawl config
        crawl_config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS if config.bypass_cache else None,
        )

        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=crawl_config)

                if result.success and result.extracted_content:
                    # Parse the extracted JSON content
                    try:
                        items = json.loads(result.extracted_content)
                    except json.JSONDecodeError:
                        # Sometimes Crawl4AI returns non-JSON content
                        items = []

                    # Ensure items is a list
                    if not isinstance(items, list):
                        items = [items] if items else []

                    # Filter out empty items
                    items = [item for item in items if item and isinstance(item, dict)]

                    title = result.metadata.get("title", "") if hasattr(result, "metadata") else ""

                    return StructuredResult(
                        url=url,
                        success=True,
                        schema_type=config.schema_type,
                        items=items,
                        item_count=len(items),
                        title=title,
                        raw_html=result.html[:1000] if hasattr(result, "html") else None,
                    )
                else:
                    error_msg = getattr(result, "error_message", "Extraction failed or no content found")
                    return StructuredResult(
                        url=url,
                        success=False,
                        schema_type=config.schema_type,
                        items=[],
                        item_count=0,
                        error=error_msg,
                    )

        except Exception as e:
            logger.error(f"Error scraping structured data from {url}: {e}")
            return StructuredResult(
                url=url,
                success=False,
                schema_type=config.schema_type,
                items=[],
                item_count=0,
                error=str(e),
            )

    async def scrape_many_structured(
        self,
        urls: list[str],
        config: StructuredScrapeConfig,
        max_concurrent: int = 5,
    ) -> list[StructuredResult]:
        """Scrape multiple URLs with structured extraction

        Args:
            urls: List of URLs to scrape
            config: Configuration for structured extraction
            max_concurrent: Maximum concurrent requests

        Returns:
            List of StructuredResult objects
        """
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, JsonCssExtractionStrategy
        from crawl4ai.cache_context import CacheMode
        from .extraction_schemas import get_schema

        logger.info(f"Structured scraping {len(urls)} URLs (schema={config.schema_type})")

        # Get the pre-built schema
        schema = get_schema(config.schema_type)
        if not schema:
            return [StructuredResult(
                url=url,
                success=False,
                schema_type=config.schema_type,
                items=[],
                item_count=0,
                error=f"Unknown schema type: {config.schema_type}",
            ) for url in urls]

        # Allow custom selector override
        if config.custom_selector:
            schema = schema.copy()
            schema["baseSelector"] = config.custom_selector

        # Build extraction strategy
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

        # Build crawl config
        crawl_config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS if config.bypass_cache else None,
        )

        results = []

        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                # Process in batches to avoid overwhelming the server
                for i in range(0, len(urls), max_concurrent):
                    batch = urls[i:i + max_concurrent]

                    # Use arun_many for batch processing
                    crawl_results = await crawler.arun_many(batch, config=crawl_config)

                    # Process results as they come back (generator)
                    async for result in crawl_results:
                        if result.success and result.extracted_content:
                            try:
                                items = json.loads(result.extracted_content)
                                if not isinstance(items, list):
                                    items = [items] if items else []
                                items = [item for item in items if item and isinstance(item, dict)]
                            except json.JSONDecodeError:
                                items = []

                            title = result.metadata.get("title", "") if hasattr(result, "metadata") else ""

                            results.append(StructuredResult(
                                url=result.url,
                                success=True,
                                schema_type=config.schema_type,
                                items=items,
                                item_count=len(items),
                                title=title,
                            ))
                        else:
                            error_msg = getattr(result, "error_message", "Extraction failed")
                            results.append(StructuredResult(
                                url=result.url,
                                success=False,
                                schema_type=config.schema_type,
                                items=[],
                                item_count=0,
                                error=error_msg,
                            ))

        except Exception as e:
            logger.error(f"Error in batch structured scraping: {e}")

        return results

    async def close(self):
        """Cleanup resources"""
        if self._seeder:
            try:
                await self._seeder.close()
            except Exception as e:
                logger.warning(f"Error closing seeder: {e}")


# Singleton factory
from ..utils import create_singleton_factory
get_map_crawl_service = create_singleton_factory(MapCrawlService, "get_map_crawl_service")
