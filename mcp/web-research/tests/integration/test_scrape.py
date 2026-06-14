"""Integration tests for scrape and extract tools."""

import pytest
from tests.integration.conftest import call_tool, call_tool_raw


@pytest.mark.asyncio
async def test_scrape_basic(mcp_client, session):
    """Basic scrape of a simple public URL succeeds."""
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://example.com"}
    )
    assert result["success"] is True
    assert result["url"].rstrip("/") == "https://example.com"
    assert len(result.get("content", "")) > 0
    # Clean response: no method_used or word_count
    assert "method_used" not in result
    assert "word_count" not in result


@pytest.mark.asyncio
async def test_scrape_httpbin(mcp_client, session):
    """Scraping httpbin returns content."""
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://httpbin.org/html"}
    )
    assert result["success"] is True
    assert len(result.get("content", "")) > 10


@pytest.mark.asyncio
async def test_scrape_method_override_httpx(mcp_client, session):
    """Forcing httpx method works and is fast."""
    import time
    start = time.monotonic()
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://example.com", "method": "httpx"}
    )
    elapsed = time.monotonic() - start
    assert result["success"] is True
    assert len(result.get("content", "")) > 0
    assert elapsed < 10.0, f"httpx scrape took {elapsed:.1f}s — should be fast"


@pytest.mark.asyncio
async def test_scrape_method_override_selenium(mcp_client, session):
    """Forcing selenium method works."""
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://example.com", "method": "selenium"}
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_scrape_method_override_crawl4ai(mcp_client, session):
    """Forcing crawl4ai method works."""
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://example.com", "method": "crawl4ai"}
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_scrape_invalid_url_scheme(mcp_client, session):
    """Non-HTTP URL schemes are rejected."""
    data = await call_tool_raw(
        mcp_client, session, "scrape",
        {"url": "ftp://example.com"}
    )
    result = data["result"]
    assert result.get("isError") is True
    assert "URL must start with" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_scrape_private_ip_blocked(mcp_client, session):
    """Private IPs are blocked for security."""
    data = await call_tool_raw(
        mcp_client, session, "scrape",
        {"url": "http://192.168.1.1"}
    )
    result = data["result"]
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_scrape_localhost_blocked(mcp_client, session):
    """localhost is blocked for security."""
    data = await call_tool_raw(
        mcp_client, session, "scrape",
        {"url": "http://localhost:8000/health"}
    )
    result = data["result"]
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_scrape_nonexistent_domain(mcp_client, session):
    """Scraping a nonexistent domain returns a failure, not a crash."""
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://this-domain-definitely-does-not-exist-xyz123.com"}
    )
    assert "success" in result
    assert "url" in result


@pytest.mark.asyncio
async def test_extract_structured(mcp_client, session):
    """extract returns structured data."""
    result = await call_tool(
        mcp_client, session, "extract",
        {"url": "https://news.ycombinator.com/", "schema_type": "blog"}
    )
    assert "success" in result
    assert "schema_type" in result
    assert result["schema_type"] == "blog"


@pytest.mark.asyncio
async def test_scrape_concurrent(mcp_client):
    """Multiple concurrent scrapes work."""
    import asyncio
    from tests.integration.conftest import _init_session

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]

    async def do_scrape(url: str):
        sid = await _init_session(mcp_client)
        return await call_tool(mcp_client, sid, "scrape", {"url": url})

    results = await asyncio.gather(*[do_scrape(u) for u in urls])
    successes = sum(1 for r in results if r.get("success") is True)
    assert successes >= 1, "At least one concurrent scrape should succeed"
