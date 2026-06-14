"""Shared scraping implementations - used by both FastAPI and Celery"""

from typing import Callable, Any
from loguru import logger
from datetime import datetime
import re

from ..core.constants import (
    CRAWL4AI_WORD_COUNT_THRESHOLD,
    SELENIUM_PAGE_LOAD_WAIT_SECONDS,
    MIN_CONTENT_LENGTH,
    MAX_CONTENT_LENGTH,
    DEFAULT_HEADERS,
    CRAWL4AI_RETRY_COUNT,
    SELENIUM_RETRY_COUNT,
    CRAWL4AI_MAX_CONCURRENT,
)
from ..utils import extract_domain


# Semaphore to limit concurrent Crawl4AI browsers
import asyncio
_crawl4ai_semaphore = asyncio.Semaphore(CRAWL4AI_MAX_CONCURRENT)


# Block detection patterns
BLOCK_PATTERNS = {
    "captcha": re.compile(r"captcha|challenge|prove.?human|robot.?check", re.I),
    "blocked": re.compile(r"access.?denied|forbidden|blocked|unavailable", re.I),
    "rate_limit": re.compile(r"rate.?limit|too.?many.?requests|429", re.I),
    "checkpoint": re.compile(
        r"security.?checkpoint|verifying.?browser|browser.?verification|"
        r"wir.uberprüfen.ihren.browser|vercel.link/security-checkpoint|"
        r"click.here.to.fix.security",
        re.I
    ),
}


def _get_browser_proxy_url(target_url: str = None) -> str | None:
    """Get the proxy URL for browser-based scrapers (Crawl4AI, Selenium).

    Returns None if proxy is disabled or target is excluded (internal services).
    """
    from ..utils.proxy import get_proxy_manager
    manager = get_proxy_manager()
    return manager.get_proxy_url(target_url)


def _is_ip_block(error: str) -> bool:
    """Check if the error is likely an IP-based block (should rotate proxy)."""
    ip_block_patterns = (
        "403", "forbidden", "blocked", "captcha", "challenge",
        "checkpoint", "access denied", "rate limit", "429",
        "bot verification", "unavailable in your region",
    )
    error_lower = error.lower()
    return any(p in error_lower for p in ip_block_patterns)


def _rotate_proxy():
    """Rotate to next proxy in the pool after an IP-based block."""
    from ..utils.proxy import get_proxy_manager
    manager = get_proxy_manager()
    if manager.proxy_count > 1:
        new_proxy = manager.rotate()
        logger.info(f"Rotated proxy after IP block -> {new_proxy}")


def is_security_checkpoint(title: str, content: str, url: str = None) -> bool:
    """Detect if the page is a security checkpoint/challenge page

    Returns True if the page appears to be a bot protection checkpoint
    rather than actual content.
    """
    # Check title first (most reliable)
    title_lower = title.lower() if title else ""
    checkpoint_title_patterns = [
        "security checkpoint",
        "verifying your browser",
        "browser verification",
        "checkpoint",
        "access verification",
        "human verification",
        "wir überprüfen ihren browser",  # German
    ]
    for pattern in checkpoint_title_patterns:
        if pattern in title_lower:
            return True

    # Check content for specific indicators
    content_lower = content.lower() if content else ""
    checkpoint_content_indicators = [
        "vercel.link/security-checkpoint",
        "vercel.link/captcha",
        "cloudflare challenge",
        "checking your browser",
        "please wait while we verify",
        "enable javascript",
        "just a moment",
        "click here to verify",
    ]
    for indicator in checkpoint_content_indicators:
        if indicator in content_lower:
            return True

    # Check for suspiciously short content on documentation-type URLs
    # (docs pages should have substantial content)
    if url:
        url_lower = url.lower()
        # Documentation URLs should have longer content
        if any(x in url_lower for x in ["/docs/", "/documentation/", "/guide/", "/reference/"]):
            if len(content) < 300 and any(
                term in content_lower for term in ["verify", "check", "browser", "human", "robot", "javascript"]
            ):
                return True
        # Blog/article URLs should have reasonable content
        elif any(x in url_lower for x in ["/blog/", "/article/", "/posts/"]):
            if len(content) < 200 and "verify" in content_lower:
                return True

    return None  # Return None instead of False to indicate "not a checkpoint"


def is_low_quality_response(content: str, url: str = None) -> str | None:
    """Quick check if content is suspiciously low quality for the URL type

    Returns error message if low quality, None if content looks adequate.
    Fast check before expensive processing.
    """
    if not content or len(content) < 50:
        return "Blocked: Empty or near-empty response"

    url_lower = url.lower() if url else ""

    # Documentation should have substantial content
    if any(x in url_lower for x in ["/docs/", "/api/", "/reference/"]):
        if len(content) < 300:
            return f"Blocked: Documentation too short ({len(content)} chars < 300 minimum)"

    # Blog posts should have reasonable content
    if any(x in url_lower for x in ["/blog/", "/article/", "/posts/", "/news/"]):
        if len(content) < 200:
            return f"Blocked: Article too short ({len(content)} chars < 200 minimum)"

    return None


def postprocess_markdown(content: str) -> str:
    """Clean up markdown output from Crawl4AI / ContentCleaner.

    Removes common noise patterns that slip through extraction:
    - Video fallback text ("Your browser does not support the video tag")
    - Excessive whitespace from removed elements
    - Caps content to MAX_CONTENT_LENGTH to protect downstream consumers
    """
    if not content:
        return content

    # Remove video tag fallback text
    content = re.sub(
        r"\n*Your browser does not support the video tag\.?\n*",
        "\n",
        content,
        flags=re.I,
    )

    # Remove "Skip to content" / "Skip to main content" links
    content = re.sub(
        r"\[Skip to (?:main )?content\]\([^)]+\)\n?",
        "",
        content,
        flags=re.I,
    )

    # Collapse 3+ consecutive blank lines into 2
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Strip trailing whitespace per line
    content = "\n".join(line.rstrip() for line in content.split("\n"))

    # Cap content length
    if len(content) > MAX_CONTENT_LENGTH:
        logger.info(
            f"Content truncated: {len(content)} -> {MAX_CONTENT_LENGTH} chars"
        )
        # Truncate at last paragraph break within limit for clean cut
        truncated = content[:MAX_CONTENT_LENGTH]
        last_break = truncated.rfind("\n\n")
        if last_break > MAX_CONTENT_LENGTH * 0.7:
            content = truncated[:last_break]
        else:
            content = truncated
        content += f"\n\n... (content truncated at {MAX_CONTENT_LENGTH} chars, total was longer)"

    return content.strip()


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
            if block_type == "checkpoint":
                return "Blocked: Security checkpoint - bot verification required"

    return None


def build_scrape_response(
    success: bool,
    url: str,
    method: str,
    title: str = None,
    content: str = None,
    metadata: dict = None,
    error: str = None
) -> dict:
    """Build standard scrape response dict"""
    return {
        "success": success,
        "url": url,
        "domain": extract_domain(url),
        "method_used": method,
        "title": title,
        "content": content,
        "summary": None,
        "metadata": metadata or {},
        "error": error,
    }


def build_content_too_short_response(url: str, method: str, length: int) -> dict:
    """Build response for content that's too short"""
    return {
        "success": False,
        "url": url,
        "domain": extract_domain(url),
        "method_used": method,
        "error": f"Content too short ({length} chars < minimum)",
    }


def build_error_response(url: str, method: str, error) -> dict:
    """Build error response dict"""
    return {
        "success": False,
        "url": url,
        "domain": extract_domain(url),
        "method_used": method,
        "error": str(error),
    }


async def scrape_httpx(url: str, cleaner, css_selector: str = None) -> dict:
    """Fast HTTP scraper using httpx + trafilatura — no browser, no Chrome, no crashes.

    Handles static and server-rendered pages (~70% of the web). Falls back to
    ContentCleaner if trafilatura extracts nothing useful.
    """
    import httpx
    from ..utils.proxy import create_proxied_client

    html_headers = {
        "User-Agent": DEFAULT_HEADERS["User-Agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    try:
        async with create_proxied_client(timeout=15.0, target_url=url) as client:
            response = await client.get(url, headers=html_headers, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return build_error_response(url, "httpx", f"Not HTML: {content_type}")

            html = response.text

        # Try trafilatura (best for article/news extraction)
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html,
                url=url,
                include_tables=True,
                include_links=False,
                favor_recall=True,
                deduplicate=True,
            )
            if extracted and len(extracted) >= MIN_CONTENT_LENGTH:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    title_tag = soup.find("title")
                    title = title_tag.get_text(strip=True) if title_tag else ""
                except Exception:
                    title = ""

                if is_security_checkpoint(title, extracted, url):
                    return build_error_response(url, "httpx", "Blocked: Security checkpoint")

                clean = postprocess_markdown(extracted)
                return build_scrape_response(
                    success=True, url=url, method="httpx",
                    title=title, content=clean,
                    metadata=_build_metadata(len(clean.split())),
                )
        except Exception as te:
            logger.debug(f"trafilatura failed for {url}: {te}")

        # Fallback: ContentCleaner
        clean = cleaner.clean(html, url, css_selector)
        if len(clean) < MIN_CONTENT_LENGTH:
            return build_content_too_short_response(url, "httpx", len(clean))

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
        except Exception:
            title = ""

        if is_security_checkpoint(title, clean, url):
            return build_error_response(url, "httpx", "Blocked: Security checkpoint")

        clean = postprocess_markdown(clean)
        return build_scrape_response(
            success=True, url=url, method="httpx",
            title=title, content=clean,
            metadata=_build_metadata(len(clean.split())),
        )

    except httpx.HTTPStatusError as e:
        return build_error_response(url, "httpx", f"HTTP {e.response.status_code}")
    except httpx.TimeoutException:
        return build_error_response(url, "httpx", "Request timed out")
    except Exception as e:
        logger.debug(f"httpx scrape error for {url}: {e}")
        return build_error_response(url, "httpx", str(e))


async def scrape_crawl4ai(url: str, cleaner, css_selector: str = None, text_only: bool = False) -> dict:
    """
    Scrape using Crawl4AI (fast, JS-enabled) with stealth mode

    Args:
        url: URL to scrape
        cleaner: ContentCleaner instance
        css_selector: Optional CSS selector for targeted extraction
        text_only: If True, disable images for faster loading

    Returns dict with keys: success, url, domain, method_used, title, content, metadata, error
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        from crawl4ai.async_configs import CrawlerRunConfig
        from crawl4ai.cache_context import CacheMode
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        # Limit concurrent Crawl4AI browsers to prevent memory exhaustion
        async with _crawl4ai_semaphore:
            browser_kwargs = dict(
                headless=True,
                enable_stealth=True,
                user_agent_mode="random",
                text_mode=text_only,
                verbose=False,
                # --no-zygote disables the Chrome zygote process that requires
                # Linux namespaces — without it Chrome SIGTRAPs in containers
                # with allowPrivilegeEscalation=false.
                extra_args=["--disable-crash-reporter", "--no-zygote"],
            )
            # Route through VPN proxy (never hit the internet bare)
            proxy_url = _get_browser_proxy_url(url)
            if proxy_url:
                browser_kwargs["proxy"] = proxy_url

            browser_config = BrowserConfig(**browser_kwargs)

            run_config = CrawlerRunConfig(
                word_count_threshold=CRAWL4AI_WORD_COUNT_THRESHOLD,
                cache_mode=CacheMode.BYPASS,
                process_iframes=False,
                mean_delay=0.3,
                delay_before_return_html=0.2,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(),
                ),
            )

            async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                if result.success:
                    domain = extract_domain(url)
                    html_content = result.html

                    # FAST CHECK: Get title and check for checkpoint BEFORE expensive cleaning
                    title = result.metadata.get("title", "") if hasattr(result, "metadata") else ""

                    # Quick checkpoint detection on raw title (fail fast)
                    if title and "checkpoint" in title.lower():
                        logger.warning(f"Fast checkpoint detection for {url}: {title}")
                        return build_error_response(url, "crawl4ai", "Blocked: Security checkpoint - bot verification required")

                    # Quick status code check
                    status_code = getattr(result, 'status_code', None)
                    if status_code and status_code >= 400:
                        block_error = detect_blocking("", status_code)
                        if block_error:
                            logger.warning(f"Fast status code detection for {url}: {block_error}")
                            return build_error_response(url, "crawl4ai", block_error)

                    # Quick HTML checkpoint check (before cleaning)
                    if is_security_checkpoint(title, html_content[:1000], url) is True:
                        logger.warning(f"HTML checkpoint detected for {url}: {title}")
                        return build_error_response(url, "crawl4ai", "Blocked: Security checkpoint - bot verification required")

                    # Crawl4AI produces two markdown outputs:
                    # - fit_markdown: pruned by PruningContentFilter (removes cookie
                    #   banners, nav menus, sidebars, low-value nodes)
                    # - raw_markdown: full page after basic tag stripping
                    md_result = result.markdown

                    clean_markdown = ""
                    has_fit = hasattr(md_result, 'fit_markdown') and bool(md_result.fit_markdown)
                    has_raw = hasattr(md_result, 'raw_markdown') and bool(md_result.raw_markdown)
                    logger.debug(f"Crawl4AI markdown: fit={has_fit} ({len(md_result.fit_markdown) if has_fit else 0} chars), raw={has_raw} ({len(md_result.raw_markdown) if has_raw else 0} chars), type={type(md_result).__name__}")

                    if has_fit:
                        clean_markdown = md_result.fit_markdown
                    elif has_raw:
                        clean_markdown = md_result.raw_markdown
                    elif isinstance(md_result, str):
                        clean_markdown = md_result

                    # Fallback to ContentCleaner for css_selector or empty results
                    if len(clean_markdown) < MIN_CONTENT_LENGTH:
                        clean_markdown = cleaner.clean(html_content, url, css_selector)

                    if len(clean_markdown) < MIN_CONTENT_LENGTH:
                        return build_content_too_short_response(url, "crawl4ai", len(clean_markdown))

                    # Post-process: remove noise, cap length
                    clean_markdown = postprocess_markdown(clean_markdown)

                    # Final checkpoint check on cleaned content (catch anything missed)
                    if is_security_checkpoint(title, clean_markdown, url) is True:
                        logger.warning(f"Final checkpoint detected for {url}: {title}")
                        return build_error_response(url, "crawl4ai", "Blocked: Security checkpoint - bot verification required")

                    return build_scrape_response(
                        success=True,
                        url=url,
                        method="crawl4ai",
                        title=title,
                        content=clean_markdown,
                        metadata=_build_metadata(len(clean_markdown.split())),
                    )

                # If we get here, Crawl4AI didn't succeed - check for blocking
                page_html = getattr(result, 'html', '') or ''
                status_code = getattr(result, 'status_code', None)
                block_error = detect_blocking(page_html, status_code)

                if block_error:
                    logger.warning(f"Blocking detected for {url}: {block_error}")
                else:
                    # Log generic failure with more details if available
                    error_detail = getattr(result, 'error_message', 'No details')
                    logger.warning(f"Crawl4AI failed for {url}: {error_detail}")

                return build_error_response(url, "crawl4ai", block_error or "Scraping failed")

    except ImportError:
        logger.warning("Crawl4AI not installed")
        return build_error_response(url, "crawl4ai", "Crawl4AI not installed")
    except Exception as e:
        logger.warning(f"Crawl4AI error for {url}: {e}")
        return build_error_response(url, "crawl4ai", e)


async def scrape_selenium(url: str, cleaner, css_selector: str = None) -> dict:
    """
    Scrape using SeleniumBase with undetected Chrome mode

    Args:
        url: URL to scrape
        cleaner: ContentCleaner instance
        css_selector: Optional CSS selector for targeted extraction

    Returns dict with keys: success, url, domain, method_used, title, content, metadata, error
    """
    try:
        from seleniumbase import DriverContext
        from pathlib import Path
        import os

        # Find Playwright's Chromium binary
        chromium_paths = list(Path("/root/.cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome"))
        if chromium_paths:
            browser_path = str(chromium_paths[0])
            logger.info(f"Using Playwright Chromium: {browser_path}")
            # Set environment variable for SeleniumBase
            os.environ["SELENIUM_BROWSER_PATH"] = browser_path
        else:
            logger.warning("Playwright Chromium not found, SeleniumBase will use system browser")

        # Run sync Selenium in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()

        def _scrape_sync():
            # Route through VPN proxy (never hit the internet bare)
            driver_kwargs = dict(uc=True, headless=True)
            proxy_url = _get_browser_proxy_url(url)
            if proxy_url:
                driver_kwargs["chromium_arg"] = f"--proxy-server={proxy_url}"

            # Use undetected Chrome mode with Playwright's Chromium
            with DriverContext(**driver_kwargs) as driver:
                driver.open(url)
                driver.sleep(SELENIUM_PAGE_LOAD_WAIT_SECONDS)
                html_content = driver.get_page_source()
                title = driver.get_title()
                return html_content, title

        html_content, title = await loop.run_in_executor(None, _scrape_sync)

        # FAST CHECK: Detect checkpoints before expensive cleaning
        if title and "checkpoint" in title.lower():
            logger.warning(f"Fast checkpoint detection (Selenium) for {url}: {title}")
            return build_error_response(url, "selenium", "Blocked: Security checkpoint - bot verification required")

        # Quick HTML checkpoint check
        if is_security_checkpoint(title, html_content[:1000], url) is True:
            logger.warning(f"HTML checkpoint detected (Selenium) for {url}: {title}")
            return build_error_response(url, "selenium", "Blocked: Security checkpoint - bot verification required")

        clean_markdown = cleaner.clean(html_content, url, css_selector)

        # Check minimum content length
        if len(clean_markdown) < MIN_CONTENT_LENGTH:
            return build_content_too_short_response(url, "selenium", len(clean_markdown))

        # Post-process: remove noise, cap length
        clean_markdown = postprocess_markdown(clean_markdown)

        # Final checkpoint check on cleaned content
        if is_security_checkpoint(title, clean_markdown, url) is True:
            logger.warning(f"Final checkpoint detected (Selenium) for {url}: {title}")
            return build_error_response(url, "selenium", "Blocked: Security checkpoint - bot verification required")

        return build_scrape_response(
            success=True,
            url=url,
            method="selenium",
            title=title,
            content=clean_markdown,
            metadata=_build_metadata(len(clean_markdown.split())),
        )

    except Exception as e:
        logger.warning(f"Selenium error for {url}: {e}")
        return build_error_response(url, "selenium", e)


async def scrape_reddit(url: str, cleaner=None) -> dict:
    """
    Scrape Reddit using JSON API (bypasses HTML entirely)

    Returns dict with keys: success, url, domain, method_used, title, content, error
    """
    try:
        import httpx

        from ..utils.proxy import create_proxied_client

        json_url = normalize_reddit_url(url)
        logger.info(f"Fetching Reddit JSON: {json_url}")

        async with create_proxied_client(timeout=30.0, target_url=json_url) as client:
            response = await client.get(json_url, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            data = response.json()

        content, title = format_reddit_content(url, data)

        return build_scrape_response(
            success=True,
            url=url,
            method="reddit_api",
            title=title,
            content=content.strip(),
        )

    except Exception as e:
        logger.error(f"Reddit API error for {url}: {e}")
        return build_error_response(url, "reddit_api", e)


async def scrape_pdf(url: str, cleaner=None) -> dict:
    """
    Scrape PDF files by downloading and extracting text content

    Returns dict with keys: success, url, domain, method_used, title, content, metadata, error
    """
    import httpx
    import fitz  # PyMuPDF
    import io

    from ..utils.proxy import create_proxied_client

    try:
        logger.info(f"Downloading PDF: {url}")

        async with create_proxied_client(timeout=60.0, target_url=url) as client:
            response = await client.get(url, headers=DEFAULT_HEADERS)
            response.raise_for_status()

            # Verify it's actually a PDF
            content_type = response.headers.get("content-type", "")
            if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(f"URL doesn't appear to be a PDF: {content_type}")

            pdf_data = response.content
            pdf_file = io.BytesIO(pdf_data)

            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

            if doc.is_encrypted:
                return build_error_response(url, "pdf", "PDF is password protected")

            # Extract content from all pages
            markdown_content = []
            metadata = {
                "pages": len(doc),
                "fetched_at": datetime.now().isoformat()
            }

            # Get PDF metadata for title
            pdf_metadata = doc.metadata
            pdf_title = pdf_metadata.get("title") or pdf_metadata.get("subject", "")

            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if text.strip():
                    # Add page header
                    markdown_content.append(f"\n## Page {page_num + 1}\n\n")
                    markdown_content.append(text)

            doc.close()

            if not markdown_content:
                return build_error_response(url, "pdf", "No text content found in PDF")

            full_content = "".join(markdown_content)
            word_count = len(full_content.split())
            metadata["word_count"] = word_count

            # Use filename from URL as fallback title
            if not pdf_title:
                pdf_title = url.split("/")[-1].replace(".pdf", "").replace("_", " ").replace("-", " ").title()

            return build_scrape_response(
                success=True,
                url=url,
                method="pdf",
                title=pdf_title,
                content=full_content.strip(),
                metadata=metadata,
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading PDF {url}: {e}")
        return build_error_response(url, "pdf", f"HTTP error: {e.response.status_code}")
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading PDF {url}")
        return build_error_response(url, "pdf", "Download timeout")
    except Exception as e:
        logger.error(f"PDF processing error for {url}: {e}")
        return build_error_response(url, "pdf", str(e))


async def scrape_playwright(url: str, cleaner, css_selector: str = None,
                            text_only: bool = False) -> dict:
    """Raw Playwright scraper — bypasses Crawl4AI's launch wrapper.

    Crawl4AI's internal launch path SIGTRAPs in minimal containers (crashpad
    handler incompatibility with certain flag combinations). Raw
    ``playwright.chromium.launch()`` with a minimal flag set works reliably.
    This is the guaranteed-to-work tier between Crawl4AI and Selenium.
    """
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-crash-reporter",
                    "--disable-gpu",
                    "--no-zygote",
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            try:
                context = await browser.new_context(
                    user_agent=DEFAULT_HEADERS.get("User-Agent", ""),
                    ignore_https_errors=True,
                )
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                # Give JS frameworks a moment to hydrate/render
                await page.wait_for_timeout(2000)

                html = await page.content()
                title = await page.title()

                if not html or len(html) < MIN_CONTENT_LENGTH:
                    return build_error_response(
                        url, "playwright", "Page rendered empty or too short")

                clean_markdown = cleaner.clean(html, url, css_selector)
                if len(clean_markdown) < MIN_CONTENT_LENGTH:
                    return build_error_response(
                        url, "playwright",
                        f"Content too short: {len(clean_markdown)} chars")

                clean_markdown = postprocess_markdown(clean_markdown)

                if is_security_checkpoint(title, clean_markdown, url) is True:
                    return build_error_response(
                        url, "playwright",
                        "Blocked: Security checkpoint - bot verification required")

                return build_scrape_response(
                    success=True, url=url, method="playwright",
                    title=title, content=clean_markdown,
                    metadata=_build_metadata(len(clean_markdown.split())),
                )
            finally:
                await browser.close()
    except Exception as e:
        logger.warning(f"Raw Playwright scrape error for {url}: {e}")
        return build_error_response(url, "playwright", str(e))


async def scrape_with_fallback(
    url: str,
    cleaner,
    db: Any,
    force_method: str | None = None,
    css_selector: str | None = None,
    text_only: bool = False
) -> dict:
    """
    Main scraping routing logic with fallback chain

    Flow:
    1. Check if PDF → Use PDF scraper
    2. Check blacklist
    3. Reddit? → JSON API
    4. Force method? → Use it directly (no retries)
    5. DB prefers selenium? → Start with Selenium retries (3x)
    6. Try Crawl4AI with retries (3x)
    7. If Crawl4AI exhausted → Try Selenium with retries (3x)
    8. If all attempts fail: record failure (blacklist after 3 total failures)

    Retry logic prevents false "selenium-only" marking from temporary issues.

    Args:
        url: URL to scrape
        cleaner: ContentCleaner instance
        db: Database instance (PostgreSQL)
        force_method: Force specific scraping method
        css_selector: Optional CSS selector for targeted extraction
        text_only: If True, disable images for faster loading

    Returns dict response
    """
    import time
    start_time = time.time()
    domain = extract_domain(url)
    final_method = "unknown"

    # Helper to record metric
    async def record_metric(success: bool, method: str, error: str = None, content: str = None):
        try:
            duration_ms = (time.time() - start_time) * 1000
            await db.record_scrape_metric(
                url=url,
                domain=domain,
                method=method,
                success=success,
                duration_ms=duration_ms,
                content_length=len(content) if content else None,
                error=error
            )
        except Exception as e:
            # Don't fail the scrape if telemetry fails
            logger.debug(f"Failed to record metric: {e}")

    # Special handler: PDF files
    if url.lower().endswith(".pdf") or "pdf" in url.lower():
        logger.info(f"Using PDF scraper for {url}")
        result = await scrape_pdf(url)
        if result["success"]:
            await db.record_success(domain, "pdf")
            await record_metric(True, "pdf", content=result.get("content"))
        else:
            await record_metric(False, "pdf", error=result.get("error"))
        return result

    # Check blacklist
    if await db.is_blacklisted(domain):
        return build_scrape_response(
            success=False,
            url=url,
            method="blacklisted",
            error="Domain is blacklisted",
        )

    # Special handler: Reddit API
    if "reddit.com" in domain or "redd.it" in domain:
        logger.info(f"Using Reddit API for {url}")
        result = await scrape_reddit(url)
        if result["success"]:
            await db.record_success(domain, "reddit_api")
            await record_metric(True, "reddit_api", content=result.get("content"))
        else:
            await record_metric(False, "reddit_api", error=result.get("error"))
        return result

    # Force method? → Use it directly without retries
    if force_method:
        result = await _scrape_with_method(url, force_method, cleaner, css_selector, text_only)
        if result["success"]:
            await record_metric(True, force_method, content=result.get("content"))
        else:
            await record_metric(False, force_method, error=result.get("error"))
        return result

    # Check database for preferred method
    preferred = await db.get_domain_method(domain)

    # ========== RETRY LOGIC ==========
    # 0. Try httpx first (fast, no browser, works for ~70%+ of sites)
    #    Skip if DB knows this site needs a browser (selenium/crawl4ai preferred).
    # 1. If selenium-only preferred, try Selenium first (2x)
    # 2. Try Crawl4AI (2x) - JS-capable browser scraper
    # 3. If Crawl4AI fails → raw Playwright → Selenium (2x)

    import asyncio
    selenium_tried_first = False

    # Step 0: fast httpx scraper (skip only if site requires JS rendering)
    # Try httpx for all domains except those DB-marked as selenium-only.
    # "crawl4ai" in DB just means Chrome worked before — httpx may be faster.
    if preferred != "selenium":
        logger.info(f"Trying httpx (fast) for {url}")
        httpx_result = await scrape_httpx(url, cleaner, css_selector)
        if httpx_result["success"]:
            await db.record_success(domain, "httpx")
            await record_metric(True, "httpx", content=httpx_result.get("content"))
            return httpx_result
        logger.info(f"httpx failed for {domain}: {httpx_result.get('error', 'no content')}, falling back to browser")

    # If domain is already selenium-only, start with Selenium retries
    if preferred == "selenium":
        logger.info(f"Database prefers Selenium for {domain}, starting with Selenium retries")
        selenium_tried_first = True
        for attempt in range(1, SELENIUM_RETRY_COUNT + 1):
            logger.info(f"Selenium attempt {attempt}/{SELENIUM_RETRY_COUNT} for {url}")

            # Exponential backoff before retry
            if attempt > 1:
                delay = 2 ** (attempt - 1)
                logger.info(f"Waiting {delay} seconds before Selenium retry {attempt}...")
                await asyncio.sleep(delay)

            result = await scrape_selenium(url, cleaner, css_selector)
            if result["success"]:
                await db.record_success(domain, "selenium")
                await record_metric(True, "selenium", content=result.get("content"))
                return result
            # Rotate proxy on IP-based blocks before retrying
            if _is_ip_block(result.get("error", "")):
                _rotate_proxy()
            logger.warning(f"Selenium attempt {attempt} failed for {url}")
        # All Selenium attempts failed - continue to try Crawl4AI
        logger.warning(f"All Selenium attempts failed for {domain}, trying Crawl4AI as fallback")

    # Try Crawl4AI with retries (always try unless already succeeded)
    checkpoint_detected = False
    for attempt in range(1, CRAWL4AI_RETRY_COUNT + 1):
        logger.info(f"Crawl4AI attempt {attempt}/{CRAWL4AI_RETRY_COUNT} for {url}")

        # Exponential backoff before retry (2, 4, 8, 16... seconds)
        if attempt > 1:
            delay = 2 ** (attempt - 1)
            logger.info(f"Waiting {delay} seconds before retry {attempt}...")
            await asyncio.sleep(delay)

        result = await scrape_crawl4ai(url, cleaner, css_selector, text_only)
        if result["success"]:
            await db.record_success(domain, "crawl4ai")
            await record_metric(True, "crawl4ai", content=result.get("content"))
            return result

        # Check for security checkpoint - immediate fallback if detected
        error_msg = result.get("error", "")

        # Rotate proxy on IP-based blocks before retrying
        if _is_ip_block(error_msg):
            _rotate_proxy()

        if "Security checkpoint" in error_msg:
            logger.warning(f"Checkpoint detected on Crawl4AI attempt {attempt} - switching to Selenium immediately")
            checkpoint_detected = True
            break  # Skip remaining retries, go straight to Selenium

        # Check for browser crash - immediate fallback, don't retry Crawl4AI
        if "Target crashed" in error_msg or "crashed" in error_msg.lower():
            logger.warning(f"Browser crash detected on Crawl4AI attempt {attempt} - switching to Selenium immediately (no retry)")
            checkpoint_detected = True
            break  # Skip remaining retries, go straight to Selenium

        logger.warning(f"Crawl4AI attempt {attempt} failed for {url}")

    # Crawl4AI exhausted - try raw Playwright before Selenium (lighter, more reliable)
    logger.info(f"Trying raw Playwright for {url} (Crawl4AI exhausted)")
    pw_result = await scrape_playwright(url, cleaner, css_selector, text_only)
    if pw_result["success"]:
        await db.record_success(domain, "playwright")
        await record_metric(True, "playwright", content=pw_result.get("content"))
        return pw_result
    logger.warning(f"Raw Playwright failed for {domain}: {pw_result.get('error')}")

    # Playwright exhausted - mark domain for Selenium and try Selenium
    logger.warning(f"All Crawl4AI + Playwright attempts failed for {domain}, trying Selenium")
    await db.set_selenium_only(domain)

    # Try Selenium with retries (skip if we already tried it first and it failed)
    if not selenium_tried_first:
        for attempt in range(1, SELENIUM_RETRY_COUNT + 1):
            logger.info(f"Selenium attempt {attempt}/{SELENIUM_RETRY_COUNT} for {url}")

            # Exponential backoff before retry (2, 4, 8, 16... seconds)
            if attempt > 1:
                delay = 2 ** (attempt - 1)
                logger.info(f"Waiting {delay} seconds before Selenium retry {attempt}...")
                await asyncio.sleep(delay)

            result = await scrape_selenium(url, cleaner, css_selector)
            if result["success"]:
                await db.record_success(domain, "selenium")
                await record_metric(True, "selenium", content=result.get("content"))
                return result
            # Rotate proxy on IP-based blocks before retrying
            if _is_ip_block(result.get("error", "")):
                _rotate_proxy()
            logger.warning(f"Selenium attempt {attempt} failed for {url}")

    # All attempts failed - record failure and metric
    logger.error(f"All scraping attempts failed for {domain} (3x Crawl4AI + 1x Playwright + 3x Selenium)")
    await db.record_failure(domain, "all_methods_failed")
    await record_metric(False, "all_methods_failed", error="All scraping methods failed")

    return build_scrape_response(
        success=False,
        url=url,
        method="both_failed",
        error="All scraping methods failed. Try again later.",
    )


def normalize_reddit_url(url: str) -> str:
    """Normalize Reddit URL to JSON API format"""
    if ".json" not in url:
        return f"{url.replace('old.reddit.com', 'www.reddit.com').replace('new.reddit.com', 'www.reddit.com').rstrip('/')}.json"
    return url


def format_reddit_content(url: str, data: dict) -> tuple[str, str]:
    """
    Format Reddit JSON data into markdown

    Returns (content, title) tuple
    """
    content = f"# Reddit Thread\n\n**Original URL:** {url}\n\n"

    if "comments" in url:
        # Thread view - [post_data, comments_data]
        if isinstance(data, list) and len(data) >= 2:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
            comments_data = data[1].get("data", {})
        else:
            post_data = data.get("data", {})
            comments_data = {}

        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        author = post_data.get("author", "")
        score = post_data.get("score", 0)
        num_comments = post_data.get("num_comments", 0)
        permalink = post_data.get("permalink", "")

        content += f"## {title}\n\n"
        if selftext:
            content += f"{selftext}\n\n"
        content += f"**Score:** {score} | **Comments:** {num_comments}\n"
        content += f"**Posted by:** u/{author}\n"
        content += f"**Link:** https://www.reddit.com{permalink}\n\n"

        # Top comments
        from ..core.constants import REDDIT_MAX_COMMENTS
        comments = comments_data.get("children", [])
        if comments:
            content += "### Top Comments\n\n"
            for comment in comments[:REDDIT_MAX_COMMENTS]:
                comment_data = comment.get("data", {})
                if not comment_data:
                    continue

                comment_text = comment_data.get("body", "")
                comment_author = comment_data.get("author", "")
                comment_score = comment_data.get("score", 0)

                if comment_text:
                    comment_text = comment_text.replace("&gt;", ">")
                    content += f"**u/{comment_author}** ({comment_score} points):\n"
                    content += f"{comment_text}\n\n"
    else:
        # Subreddit or search results
        if isinstance(data, list):
            posts = data[0].get("data", {}).get("children", [])
        else:
            posts = data.get("data", {}).get("children", [])

        title = "Reddit Posts"
        content += "### Posts\n\n"

        from ..core.constants import REDDIT_MAX_POSTS
        for post in posts[:REDDIT_MAX_POSTS]:
            post_data = post.get("data", {})
            if not post_data:
                continue

            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            author = post_data.get("author", "")
            score = post_data.get("score", 0)
            permalink = post_data.get("permalink", "")
            is_self = post_data.get("is_self", False)

            content += f"#### {title}\n\n"
            if is_self and selftext:
                if len(selftext) > 500:
                    content += f"{selftext[:500]}...\n\n"
                else:
                    content += f"{selftext}\n\n"

            content += f"**{score} points** | [link](https://www.reddit.com{permalink}) by u/{author}\n\n"

    return content, title or data.get("data", {}).get("title", "Reddit Thread") if isinstance(data, dict) else "Reddit Thread"


async def _scrape_with_method(url: str, method: str, cleaner, css_selector: str = None, text_only: bool = False) -> dict:
    """Scrape using specific method"""
    if method == "httpx":
        return await scrape_httpx(url, cleaner, css_selector)
    elif method == "crawl4ai":
        return await scrape_crawl4ai(url, cleaner, css_selector, text_only)
    elif method == "selenium":
        return await scrape_selenium(url, cleaner, css_selector)
    elif method == "reddit_api":
        return await scrape_reddit(url, cleaner)
    elif method == "pdf":
        return await scrape_pdf(url, cleaner)
    else:
        return build_error_response(url, method, "Unknown method")


def _build_metadata(word_count: int) -> dict:
    """Build metadata dict for successful scrapes"""
    return {
        "word_count": word_count,
        "fetched_at": datetime.now().isoformat()
    }


def dict_to_scrape_response(data: dict, response_class):
    """Convert dict to ScrapeResponse (for FastAPI)"""
    return response_class(**data)
