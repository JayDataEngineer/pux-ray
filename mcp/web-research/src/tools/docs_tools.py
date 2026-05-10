"""Documentation Tools

Tools for fetching documentation from llms.txt sources.
- docs_list_sources: List available documentation libraries
- docs_fetch_docs: Fetch documentation from a URL

Note: Caching is now handled by FastMCP's ResponseCachingMiddleware
"""

from typing import Annotated

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field
import re
from urllib.parse import urlparse


# Get settings
from ..settings import get_settings
_settings = get_settings()
DOCS_CONFIG_PATH = _settings.docs_config_path
DOCS_LOCAL_DIR = _settings.docs_local_dir


# Session-level storage for dynamically discovered domains
_session_allowed_domains: set[str] = set()


def _reset_session_domains():
    """Reset session-allowed domains (for testing)"""
    global _session_allowed_domains
    _session_allowed_domains = set()


def _is_http_or_https(url: str) -> bool:
    """Check if the URL is an HTTP or HTTPS URL."""
    return url.startswith(("http://", "https://"))


def _normalize_path(path: str) -> str:
    """Accept paths in file:/// or relative format and map to absolute paths."""
    import os
    return (
        os.path.abspath(path[7:])
        if path.startswith("file://")
        else os.path.abspath(path)
    )


async def _add_domains_from_content(content: str, base_url: str, ctx: Context | None = None) -> None:
    """Extract domains from markdown links and add to session allowlist."""
    global _session_allowed_domains

    # Extract markdown links: [text](url) or [text](url "title")
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc.replace("www.", "")

    for match in re.finditer(link_pattern, content):
        url = match.group(2).split()[0]  # Remove trailing "title" if present
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                domain = parsed.netloc.replace("www.", "")
                # Only add external domains (not the same as base)
                if domain != base_domain and domain not in _session_allowed_domains:
                    _session_allowed_domains.add(domain)
                    if ctx:
                        await ctx.debug(f"Added discovered domain to allowlist: {domain}")
        except:
            pass


def _is_url_allowed(url: str, configured_domains: set[str]) -> bool:
    """Check if a URL is from an allowed domain.

    Allows:
    - Exact matches with configured domains
    - Subdomains of configured domains
    - Session-discovered domains (from llms.txt content)
    """
    global _session_allowed_domains

    parsed = urlparse(url)
    netloc = parsed.netloc

    # Remove port if present
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    # Remove www. prefix for checking
    check_netloc = netloc[4:] if netloc.startswith("www.") else netloc

    # Combine configured + session-discovered domains
    all_allowed = configured_domains | _session_allowed_domains

    # Check exact match or if requested domain is a subdomain of allowed
    for allowed in all_allowed:
        # Direct match
        if check_netloc == allowed:
            return True
        # Requested URL is a subdomain of allowed domain
        if check_netloc.endswith(f".{allowed}"):
            return True
        # Allowed domain is a subdomain of requested URL (for base domain matching)
        if allowed.endswith(f".{check_netloc}"):
            return True

    return False


async def _load_docs_sources(ctx: Context | None = None) -> tuple[dict, set, set]:
    """Load documentation sources from YAML config file.

    Returns:
        Tuple of (name -> url mapping, set of allowed local file paths, set of allowed domains)
    """
    import yaml
    import os
    local_sources = []
    remote_sources = []

    try:
        with open(DOCS_CONFIG_PATH, "r") as f:
            sources = yaml.safe_load(f) or []

        for s in sources:
            url_or_path = s.get("llms_txt", "")
            if _is_http_or_https(url_or_path):
                remote_sources.append(s)
            else:
                local_sources.append(s)

        # Build name -> url mapping
        mapping = {}
        for s in remote_sources:
            name = s.get("name", _extract_domain(s["llms_txt"]))
            mapping[name] = s["llms_txt"]

        # Build allowed local files set (for security)
        allowed_local = set()
        for s in local_sources:
            path = _normalize_path(s["llms_txt"])
            name = s.get("name", os.path.basename(path))
            if not os.path.exists(path):
                if ctx:
                    await ctx.warning(f"Local docs file not found: {path}")
                continue
            mapping[name] = f"file://{path}"
            allowed_local.add(path)

        # Build allowed domains set (for security - domain fencing)
        allowed_domains = set()
        for s in remote_sources:
            domain = _extract_domain(s["llms_txt"])
            allowed_domains.add(domain)

        return mapping, allowed_local, allowed_domains

    except FileNotFoundError:
        if ctx:
            await ctx.warning(f"Docs config not found: {DOCS_CONFIG_PATH}")
        return {}, set(), set()
    except Exception as e:
        if ctx:
            await ctx.warning(f"Failed to load docs config: {e}")
        return {}, set(), set()


def _extract_domain(url: str) -> str:
    """Extract full domain from URL for naming and domain fencing."""
    parsed = urlparse(url)
    # Return full domain without www prefix
    return parsed.netloc.replace("www.", "")


async def docs_list_sources(ctx: Context | None = None) -> str:
    """List all available documentation libraries and their llms.txt URLs.

    START HERE to discover which documentation libraries are available.
    This returns a list of llms.txt endpoints that act as indexes to
    documentation content.

    WORKFLOW:
    1. Call this tool first to get available libraries
    2. Call docs_fetch_docs() with a library's llms.txt URL
    3. Read the returned index to find specific documentation URLs
    4. Call docs_fetch_docs() again with those specific URLs to get actual content

    Returns:
        Formatted list of documentation sources with their URLs
    """
    if ctx:
        await ctx.debug("Loading documentation sources")

    sources, _, _ = await _load_docs_sources(ctx)
    if not sources:
        return "No documentation sources configured."

    lines = []
    for name, url_or_path in sources.items():
        lines.append(f"{name}")
        if url_or_path.startswith("file://"):
            lines.append(f"  Path: {url_or_path[7:]}")  # Strip file:// prefix
        else:
            lines.append(f"  URL: {url_or_path}")
    return "\n".join(lines)


async def docs_fetch_docs(
    url: Annotated[str, Field(description="The documentation URL to fetch. Use URLs from docs_list_sources or links found in llms.txt files.")],
    ctx: Context | None = None
) -> str:
    """Fetch documentation from a URL and convert to clean Markdown.

    CRITICAL WORKFLOW - This is a TWO-STEP process:

    1. FIRST CALL: Fetch the llms.txt URL (from docs_list_sources). This returns
       an INDEX of markdown links, not the actual documentation.

    2. READ THE INDEX: The returned markdown contains links like:
       - [Introduction](https://docs.example.com/intro)
       - [API Reference](https://docs.example.com/api)

    3. SECOND CALL: Call this tool AGAIN with the specific documentation URL
       (e.g., https://docs.example.com/intro) to get the actual content.

    If you only call this tool once with an llms.txt URL, you will NOT have
    the actual documentation - just a list of links. You MUST call it again
    with the specific page URLs.

    Args:
        url: The documentation URL to fetch. Can be:
            - llms.txt URL (returns index of links)
            - Specific documentation page URL (returns actual content)
            - Local file path (must be configured in docs_config.yaml)

    Returns:
        Clean Markdown content from the documentation source

    Security:
        Domain fencing is enabled - only URLs from configured documentation
        sources and their subdomains are allowed. This prevents fetching from
        internal services or arbitrary URLs.

    Note:
        Results are cached by FastMCP's ResponseCachingMiddleware with
        configurable TTL.
    """
    # Get allowed local files and domains for security
    _, allowed_local_files, allowed_domains = await _load_docs_sources(ctx)
    url_or_path = url.strip()

    # Handle local file paths
    if not _is_http_or_https(url_or_path):
        # Normalize the path (handles file:// and direct paths)
        abs_path = _normalize_path(url_or_path)

        # Security check: file must be in allowed list
        if abs_path not in allowed_local_files:
            raise ToolError(
                f"Local file not allowed: {abs_path}. "
                f"Allowed files are those listed in docs_config.yaml."
            )

        if ctx:
            await ctx.info(f"Reading local file: {abs_path}")

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            # If it's already markdown, return as-is
            if abs_path.endswith((".md", ".markdown", ".txt")):
                markdown = content
            # If it's HTML, clean it
            elif abs_path.endswith((".html", ".htm")):
                cleaner = ctx.lifespan_context.get("cleaner")
                if not cleaner:
                    raise ToolError("Content cleaner service not available")
                markdown = cleaner.clean(content, url=url_or_path)
            else:
                # Try to detect - if it looks like HTML, clean it
                if content.strip().startswith("<"):
                    cleaner = ctx.lifespan_context.get("cleaner")
                    if not cleaner:
                        raise ToolError("Content cleaner service not available")
                    markdown = cleaner.clean(content, url=url_or_path)
                else:
                    markdown = content

            if ctx:
                word_count = len(markdown.split())
                await ctx.info(f"Read {word_count} words from local file")

            return markdown

        except FileNotFoundError:
            raise ToolError(f"Local file not found: {abs_path}")
        except Exception as e:
            raise ToolError(f"Error reading local file: {str(e)}")

    # Handle HTTP/HTTPS URLs with domain fencing
    if ctx:
        await ctx.info(f"Fetching documentation: {url}")

    # Security: Domain fencing - check if URL is from allowed domain
    if not _is_url_allowed(url, allowed_domains):
        # Extract the domain from the requested URL for the error message
        requested_domain = urlparse(url).netloc
        raise ToolError(
            f"URL not allowed: {url} is from domain '{requested_domain}'. "
            f"Documentation fetches are restricted to configured sources only. "
            f"Allowed domains: {', '.join(sorted(allowed_domains))}"
        )

    # Get services from lifespan context with safe access
    cleaner = ctx.lifespan_context.get("cleaner")

    if not cleaner:
        raise ToolError("Content cleaner service not available")

    # Fetch the documentation
    import httpx

    from ..utils.proxy import create_proxied_client

    try:
        async with create_proxied_client(timeout=30.0, target_url=url) as client:
            if ctx:
                await ctx.debug(f"Sending HTTP GET to {url}")

            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        # Use ContentCleaner for better HTML->Markdown conversion
        markdown = cleaner.clean(html, url=url)

        if not markdown:
            raise ToolError(f"No content could be extracted from {url}")

        if ctx:
            word_count = len(markdown.split())
            await ctx.info(f"Fetched {word_count} words of documentation")

        # Extract domains from links in the content and add to session allowlist
        await _add_domains_from_content(markdown, url, ctx)

        return markdown

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 404:
            raise ToolError(f"Documentation not found (404) at {url}")
        elif status == 403:
            raise ToolError(f"Access denied (403) when fetching {url}")
        elif status >= 500:
            raise ToolError(f"Server error ({status}) when fetching {url}")
        else:
            raise ToolError(f"HTTP error {status} when fetching {url}")
    except httpx.TimeoutException:
        raise ToolError(f"Request timed out when fetching {url}")
    except httpx.ConnectError:
        raise ToolError(f"Could not connect to {url} - the server may be down")
    except httpx.RequestError as e:
        raise ToolError(f"Network error fetching {url}: {str(e)}")
    except ToolError:
        raise  # Re-raise ToolError as-is (user-facing message)
    except Exception as e:
        if ctx:
            await ctx.error(f"Unexpected error fetching {url}: {e}")
        raise ToolError(f"Unexpected error when processing {url}")
