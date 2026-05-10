"""Integration tests for the health endpoint and MCP protocol basics."""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(mcp_client):
    """Health endpoint returns healthy status."""
    response = await mcp_client.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["server"] == "mcp-research-server"


@pytest.mark.asyncio
async def test_mcp_initialize(mcp_client):
    """MCP initialize handshake works."""
    from tests.integration.conftest import _init_session

    session_id = await _init_session(mcp_client)
    assert session_id is not None
    assert len(session_id) > 0


@pytest.mark.asyncio
async def test_list_tools(mcp_client, session):
    """tools/list returns expected tool names."""
    from tests.integration.conftest import _mcp_request

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    data, _ = await _mcp_request(mcp_client, request, session)
    tools = data["result"]["tools"]
    tool_names = {t["name"] for t in tools}

    expected = {
        "research",
        "search",
        "scrape",
        "extract",
        "list_schemas",
        "map",
        "crawl",
        "domains",
        "stats",
        "reset",
        "clear_blacklist",
        "proxy_status",
        "proxy_test",
        "proxy_rotate",
        "docs_list_sources",
        "docs_fetch_docs",
    }
    missing = expected - tool_names
    assert not missing, f"Missing tools: {missing}"


@pytest.mark.asyncio
async def test_session_persistence(mcp_client, session):
    """Session ID persists across multiple requests."""
    from tests.integration.conftest import _mcp_request

    for i in range(3):
        request = {
            "jsonrpc": "2.0",
            "id": i + 2,
            "method": "tools/call",
            "params": {"name": "get_domains", "arguments": {}},
        }
        data, new_session = await _mcp_request(mcp_client, request, session)
        assert session == new_session, "Session ID should persist across requests"


@pytest.mark.asyncio
async def test_invalid_tool_name(mcp_client, session):
    """Calling a nonexistent tool returns isError."""
    from tests.integration.conftest import call_tool_raw

    data = await call_tool_raw(
        mcp_client, session, "nonexistent_tool_xyz", {}
    )
    result = data["result"]
    assert result.get("isError") is True
    assert "Unknown tool" in result["content"][0]["text"]
