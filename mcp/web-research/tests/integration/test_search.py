"""Integration tests for search and research tools."""

import pytest
from tests.integration.conftest import call_tool


@pytest.mark.asyncio
async def test_search_basic(mcp_client, session):
    """Basic search returns results with expected structure."""
    result = await call_tool(
        mcp_client, session, "search",
        {"query": "python async await", "top_k": 5}
    )
    assert result["query"] == "python async await"
    assert result["total_results"] >= 0
    assert "results" in result
    assert len(result["results"]) <= 5

    for r in result["results"]:
        assert "title" in r
        assert "url" in r
        assert "snippet" in r


@pytest.mark.asyncio
async def test_search_top_k_limit(mcp_client, session):
    """top_k parameter caps returned results."""
    result = await call_tool(
        mcp_client, session, "search",
        {"query": "docker tutorial", "top_k": 3}
    )
    assert len(result["results"]) <= 3


@pytest.mark.asyncio
async def test_search_empty_query_rejected(mcp_client, session):
    """Empty query is rejected (min_length=1)."""
    from tests.integration.conftest import call_tool_raw

    data = await call_tool_raw(
        mcp_client, session, "search",
        {"query": ""}
    )
    result = data["result"]
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_research_basic(mcp_client, session):
    """Research tool searches and scrapes top results."""
    result = await call_tool(
        mcp_client, session, "research",
        {"query": "python asyncio", "max_results": 2}
    )
    assert result["query"] == "python asyncio"
    assert "results" in result
    assert len(result["results"]) <= 2

    for r in result["results"]:
        assert "title" in r
        assert "url" in r
        assert "snippet" in r
        # Content may be None if scrape failed, but key must exist
        assert "content" in r


@pytest.mark.asyncio
async def test_search_concurrent(mcp_client):
    """Multiple searches can run concurrently."""
    import asyncio
    from tests.integration.conftest import _init_session

    async def do_search(query: str):
        sid = await _init_session(mcp_client)
        return await call_tool(
            mcp_client, sid, "search",
            {"query": query, "top_k": 3}
        )

    results = await asyncio.gather(
        do_search("golang tutorial"),
        do_search("rust programming"),
    )
    for r in results:
        assert "results" in r
        assert r["total_results"] >= 0
