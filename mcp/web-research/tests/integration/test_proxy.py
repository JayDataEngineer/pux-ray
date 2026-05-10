"""Integration tests for proxy management tools."""

import pytest
from tests.integration.conftest import call_tool, call_tool_raw


@pytest.mark.asyncio
async def test_proxy_status(mcp_client, session):
    """proxy_status returns configuration info."""
    result = await call_tool(mcp_client, session, "proxy_status", {})
    assert "enabled" in result
    assert "rotation" in result
    # If proxy is configured, check structure
    if result["enabled"]:
        assert result["proxy_count"] >= 1
        assert result["current_proxy"] is not None


@pytest.mark.asyncio
async def test_proxy_test(mcp_client, session):
    """proxy_test runs connectivity check (may fail if proxy not configured)."""
    data = await call_tool_raw(mcp_client, session, "proxy_test", {})
    result = data["result"]

    # If proxy not configured, we get an error - that's acceptable
    if result.get("isError"):
        assert "not configured" in result["content"][0]["text"].lower()
        pytest.skip("Proxy not configured in this environment")
        return

    # If proxy IS configured, check the response
    inner = result.get("content", [{}])[0]
    if isinstance(inner, dict) and inner.get("type") == "text":
        import json
        parsed = json.loads(inner["text"])
        assert "proxy_working" in parsed


@pytest.mark.asyncio
async def test_proxy_rotate_with_single_proxy(mcp_client, session):
    """proxy_rotate with single proxy returns error."""
    data = await call_tool_raw(mcp_client, session, "proxy_rotate", {})
    result = data["result"]

    if result.get("isError"):
        # Expected: either not configured or only one proxy
        text = result["content"][0]["text"].lower()
        assert "not configured" in text or "only one proxy" in text
    else:
        # Multiple proxies configured - rotation should work
        import json
        inner = result.get("content", [{}])[0]
        parsed = json.loads(inner["text"])
        assert parsed["status"] == "rotated"
