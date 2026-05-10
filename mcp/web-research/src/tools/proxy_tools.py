"""Proxy Management Tools

Administrative tools for monitoring and controlling proxy configuration.
- proxy_status: View current proxy configuration and stats
- proxy_test: Test proxy connectivity and report exit IP
- proxy_rotate: Manually rotate to the next proxy in the list
"""

import httpx
from loguru import logger
from fastmcp import Context
from fastmcp.exceptions import ToolError

from ..utils.proxy import get_proxy_manager


async def proxy_status(ctx: Context | None = None) -> dict:
    """Show current proxy configuration and rotation stats

    Returns the proxy list, current proxy, rotation strategy,
    exclusion list, and whether proxy is enabled.

    Returns:
        Dictionary with proxy configuration details
    """
    manager = get_proxy_manager()
    stats = manager.get_stats()

    if ctx:
        if stats["enabled"]:
            await ctx.info(
                f"Proxy enabled: {stats['proxy_count']} proxy(ies), "
                f"current: {stats['current_proxy']}, "
                f"rotation: {stats['rotation']}"
            )
        else:
            await ctx.info("Proxy disabled (no MCP_PROXY_URL or MCP_PROXY_URLS configured)")

    return stats


async def proxy_test(ctx: Context | None = None) -> dict:
    """Test proxy connectivity by checking the exit IP

    Makes requests through the configured proxy to detect the external IP.
    Compares against a direct (no-proxy) request to verify the proxy is working.

    Returns:
        Dictionary with proxy IP, direct IP, and whether proxy is working
    """
    manager = get_proxy_manager()

    if not manager.enabled:
        raise ToolError("Proxy is not configured. Set MCP_PROXY_URL or MCP_PROXY_URLS to enable.")

    proxy_url = manager.get_current()
    if not proxy_url:
        raise ToolError("No proxy available")

    if ctx:
        await ctx.info(f"Testing proxy: {proxy_url}")

    # IP detection services (try multiple for reliability)
    ip_services = [
        "https://api.ipify.org?format=json",
        "https://ifconfig.me/ip",
    ]

    results = {
        "proxy_url": proxy_url,
        "proxy_ip": None,
        "direct_ip": None,
        "proxy_working": False,
        "ip_different": None,
    }

    # Test through proxy
    try:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            timeout=15.0,
        ) as client:
            response = await client.get(ip_services[0])
            data = response.json()
            results["proxy_ip"] = data.get("ip", response.text.strip())
    except Exception as e:
        results["proxy_error"] = str(e)
        if ctx:
            await ctx.error(f"Proxy request failed: {e}")

    # Test direct (no proxy) for comparison
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(ip_services[0])
            data = response.json()
            results["direct_ip"] = data.get("ip", response.text.strip())
    except Exception as e:
        results["direct_error"] = str(e)

    # Determine if proxy is working
    results["proxy_working"] = results["proxy_ip"] is not None
    if results["proxy_ip"] and results["direct_ip"]:
        results["ip_different"] = results["proxy_ip"] != results["direct_ip"]

    if ctx:
        if results["proxy_working"]:
            status = "WORKING" if results.get("ip_different") else "WORKING (same IP - may not be routing traffic)"
            await ctx.info(f"Proxy test: {status}")
        else:
            await ctx.error("Proxy test: FAILED")

    return results


async def proxy_rotate(ctx: Context | None = None) -> dict:
    """Manually rotate to the next proxy in the rotation list

    Advances to the next proxy in the list (round-robin) regardless
    of the configured rotation strategy.

    Returns:
        Dictionary with the new current proxy and rotation stats
    """
    manager = get_proxy_manager()

    if not manager.enabled:
        raise ToolError("Proxy is not configured. Set MCP_PROXY_URL or MCP_PROXY_URLS to enable.")

    if len(manager._proxies) < 2:
        raise ToolError("Only one proxy configured — rotation requires 2 or more proxies.")

    new_proxy = manager.rotate()
    stats = manager.get_stats()

    if ctx:
        await ctx.info(f"Rotated to proxy: {new_proxy} (index {stats['current_index']})")

    return {
        "status": "rotated",
        "current_proxy": new_proxy,
        "index": stats["current_index"],
        "total_proxies": stats["proxy_count"],
    }
