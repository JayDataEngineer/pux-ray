"""Integration tests for admin tools: stats, domains, reset."""

import pytest
from tests.integration.conftest import call_tool


@pytest.mark.asyncio
async def test_stats(mcp_client, session):
    """stats returns stats with expected structure."""
    result = await call_tool(
        mcp_client, session, "stats",
        {"hours": 24}
    )
    assert "total_scrapes" in result
    assert "success_rate" in result
    assert "avg_duration_ms" in result
    assert "by_method" in result
    assert "period_hours" in result
    assert result["period_hours"] == 24


@pytest.mark.asyncio
async def test_stats_custom_hours(mcp_client, session):
    """stats respects hours parameter."""
    result = await call_tool(
        mcp_client, session, "stats",
        {"hours": 1}
    )
    assert result["period_hours"] == 1


@pytest.mark.asyncio
async def test_domains(mcp_client, session):
    """domains returns domain list."""
    result = await call_tool(mcp_client, session, "domains", {})
    assert "total" in result
    assert "domains" in result
    assert isinstance(result["domains"], list)


@pytest.mark.asyncio
async def test_list_schemas(mcp_client, session):
    """list_schemas returns available extraction schemas."""
    result = await call_tool(mcp_client, session, "list_schemas", {})
    assert "total" in result
    assert "schemas" in result
    assert result["total"] >= 0


@pytest.mark.asyncio
async def test_reset(mcp_client, session):
    """reset clears all domain records."""
    from src.db.database import Database

    # Insert test data first
    db_obj = Database()
    await db_obj.init()
    try:
        await db_obj.record_success("test-cleanup.example.com", "crawl4ai")

        # Clean via tool
        result = await call_tool(mcp_client, session, "reset", {})
        assert result["status"] == "success"
        assert result["records_removed"] >= 1

        # Verify cleaned
        domain_list = await db_obj.get_all_domains()
        test_domains = [d for d in domain_list if d["domain"] == "test-cleanup.example.com"]
        assert len(test_domains) == 0
    finally:
        await db_obj.close()
