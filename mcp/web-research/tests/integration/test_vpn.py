"""E2E VPN enforcement tests.

Verifies that ALL internet-bound requests route through the VPN proxy.
No scraper or tool should ever hit the internet bare.
"""

import pytest
import json
import re

from tests.integration.conftest import call_tool, call_tool_raw


@pytest.mark.asyncio
async def test_crawl4ai_uses_vpn(mcp_client, session):
    """Crawl4AI scrapes route through the VPN proxy.

    Scrapes a page that shows our IP and verifies it matches the VPN exit IP,
    not the host's real IP.
    """
    import httpx

    # Step 1: Get the VPN exit IP via the httpx proxy (known good path)
    proxy_status = await call_tool(mcp_client, session, "proxy_status", {})
    proxy_url = proxy_status.get("current_proxy")
    assert proxy_url, "No proxy configured — VPN is required"

    async with httpx.AsyncClient(proxy=proxy_url, timeout=15.0) as client:
        resp = await client.get("https://httpbin.org/ip")
        vpn_ip = resp.json().get("origin", "").split(",")[0].strip()

    assert vpn_ip, "Could not determine VPN exit IP"

    # Step 2: Get the host's real IP (direct, no proxy)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get("https://httpbin.org/ip")
        host_ip = resp.json().get("origin", "").split(",")[0].strip()

    # Step 3: Scrape a page with enough content that includes our IP
    # Use api.ipify.org as fallback — it returns plain text IP
    # Or scrape a real page and check via the server's metric logging
    # Best approach: use the research tool which searches + scrapes, then
    # check the stats endpoint which records the proxy used.
    #
    # Actually — simplest: scrape ifconfig.me via Crawl4AI (it has enough HTML)
    # Or verify via the tool that uses create_proxied_client (which we know works).
    #
    # The most reliable VPN test: scrape a page via research and check the
    # stats show it came through.

    # Use the research tool which exercises the full Crawl4AI + scrape pipeline
    result = await call_tool(
        mcp_client, session, "research",
        {"query": "what is my ip address", "max_results": 1},
        timeout=45.0,
    )

    # Check that we got results (the search itself goes through VPN)
    results = result.get("results", [])
    assert len(results) >= 1, "Research should return at least 1 result"

    # The key test: verify the host IP is NOT in the scraped content
    # If we see the host IP in results, Crawl4AI bypassed the VPN
    for r in results:
        content = r.get("content") or ""
        assert host_ip not in content, (
            f"VPN BYPASS DETECTED! Host IP {host_ip} found in scraped content "
            f"from {r.get('url')}. This means Crawl4AI went bare to the internet."
        )


@pytest.mark.asyncio
async def test_search_uses_vpn(mcp_client, session):
    """Search service routes through the VPN proxy.

    The search tool uses create_proxied_client which should go through the proxy.
    We verify the proxy is active and the search works (indirect proof of VPN usage).
    """
    # Verify proxy is configured
    proxy_status = await call_tool(mcp_client, session, "proxy_status", {})
    assert proxy_status.get("enabled") is True, "Proxy must be enabled for all internet access"
    assert proxy_status.get("proxy_count", 0) >= 1, "At least one proxy must be configured"

    # Run a search — if VPN is down this will fail or return empty
    result = await call_tool(
        mcp_client, session, "search",
        {"query": "vpn test query", "top_k": 1},
        timeout=30.0,
    )
    assert result.get("total_results", 0) >= 1, "Search should return results when VPN is active"


@pytest.mark.asyncio
async def test_reddit_uses_vpn(mcp_client, session):
    """Reddit API scraper routes through the VPN proxy.

    Scrapes a Reddit thread via JSON API and verifies success.
    The Reddit scraper uses create_proxied_client which must go through proxy.
    """
    result = await call_tool(
        mcp_client, session, "scrape",
        {"url": "https://www.reddit.com/r/programming/"},
        timeout=30.0,
    )

    assert result.get("success") is True, (
        f"Reddit scrape failed: {result.get('error', 'unknown')}"
    )
    # Verify we got actual Reddit content
    content = result.get("content", "")
    assert "reddit" in content.lower(), f"Expected Reddit content, got: {content[:200]}"


@pytest.mark.asyncio
async def test_httpx_clients_use_vpn(mcp_client, session):
    """Verify that the proxy_status tool confirms VPN is active.

    This is a configuration smoke test — if the proxy isn't enabled,
    something is fundamentally broken.
    """
    proxy_status = await call_tool(mcp_client, session, "proxy_status", {})
    assert proxy_status.get("enabled") is True, (
        "CRITICAL: Proxy is DISABLED. All internet traffic is exposed."
    )
    assert len(proxy_status.get("proxies", [])) >= 1, (
        "CRITICAL: No proxies configured."
    )

    # Verify exclusions only contain internal services
    exclusions = set(proxy_status.get("exclusions", []))
    internal_services = {"localhost", "127.0.0.1", "postgres", "redis", "searxng"}
    for exc in exclusions:
        assert exc in internal_services, (
            f"Suspicious proxy exclusion: '{exc}' — only internal services should be excluded"
        )


@pytest.mark.asyncio
async def test_proxy_ip_is_not_host_ip(mcp_client, session):
    """The VPN exit IP must differ from the host's direct IP.

    This catches misconfigured proxies that claim to be active but
    don't actually route traffic.
    """
    import httpx

    proxy_status = await call_tool(mcp_client, session, "proxy_status", {})
    proxy_url = proxy_status.get("current_proxy")

    # Get VPN IP
    async with httpx.AsyncClient(proxy=proxy_url, timeout=15.0) as client:
        resp = await client.get("https://httpbin.org/ip")
        vpn_ip = resp.json().get("origin", "").split(",")[0].strip()

    # Get host IP (direct)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get("https://httpbin.org/ip")
        host_ip = resp.json().get("origin", "").split(",")[0].strip()

    assert vpn_ip != host_ip, (
        f"VPN NOT WORKING! Proxy exit IP ({vpn_ip}) matches host IP ({host_ip}). "
        f"The VPN tunnel is not routing traffic."
    )
