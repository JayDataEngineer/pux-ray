"""Integration tests for the blacklist system.

Tests the 24-hour failure window, auto-blacklist threshold,
domain recovery, and clear_blacklist tool.
"""

import pytest
from datetime import datetime, timezone, timedelta

from tests.integration.conftest import call_tool, call_tool_raw


# ---------- Direct Database Tests ----------


@pytest.mark.asyncio
async def test_record_failure_increments(db):
    """record_failure increments failure count."""
    from src.db.database import Database

    db_obj = Database()
    await db_obj.init()

    try:
        # Record first failure
        result = await db_obj.record_failure("test-failure.example.com")
        assert result["blacklisted"] is False
        assert result["failure_count"] == 1

        # Record second failure
        result = await db_obj.record_failure("test-failure.example.com")
        assert result["blacklisted"] is False
        assert result["failure_count"] == 2

        # Third failure triggers blacklist (threshold=3)
        result = await db_obj.record_failure("test-failure.example.com")
        assert result["blacklisted"] is True
        assert result["failure_count"] == 3

    finally:
        # Cleanup
        from sqlalchemy import delete
        from src.db.models import Domain
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-failure.example.com")
            )
            await sess.commit()
        await db_obj.close()


@pytest.mark.asyncio
async def test_record_success_clears_blacklist(db):
    """record_success clears blacklist and resets failures."""
    from src.db.database import Database

    db_obj = Database()
    await db_obj.init()

    try:
        # Blacklist the domain first
        await db_obj.blacklist("test-recovery.example.com")
        assert await db_obj.is_blacklisted("test-recovery.example.com") is True

        # Record success - should clear blacklist
        await db_obj.record_success("test-recovery.example.com", "crawl4ai")
        assert await db_obj.is_blacklisted("test-recovery.example.com") is False

    finally:
        from sqlalchemy import delete
        from src.db.models import Domain
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-recovery.example.com")
            )
            await sess.commit()
        await db_obj.close()


@pytest.mark.asyncio
async def test_24h_failure_window_resets_stale_count(db):
    """Failures older than 24 hours are reset before counting."""
    from src.db.database import Database
    from src.db.models import Domain

    db_obj = Database()
    await db_obj.init()

    try:
        # Insert a domain with stale failures (48h ago)
        async with db_obj._get_session() as sess:
            domain = Domain(
                domain="test-stale.example.com",
                failure_count=5,
                last_failure=datetime.now(timezone.utc) - timedelta(hours=48),
                is_blacklisted=False,
            )
            sess.add(domain)
            await sess.commit()

        # Record a new failure - stale count should be reset
        result = await db_obj.record_failure("test-stale.example.com")
        assert result["blacklisted"] is False
        assert result["failure_count"] == 1  # Reset from 5 to 1

    finally:
        from sqlalchemy import delete
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-stale.example.com")
            )
            await sess.commit()
        await db_obj.close()


@pytest.mark.asyncio
async def test_recent_failures_not_reset(db):
    """Failures within 24 hours are NOT reset."""
    from src.db.database import Database
    from src.db.models import Domain

    db_obj = Database()
    await db_obj.init()

    try:
        # Insert a domain with recent failures (1h ago)
        async with db_obj._get_session() as sess:
            domain = Domain(
                domain="test-recent.example.com",
                failure_count=2,
                last_failure=datetime.now(timezone.utc) - timedelta(hours=1),
                is_blacklisted=False,
            )
            sess.add(domain)
            await sess.commit()

        # Record a new failure - count should increment, not reset
        result = await db_obj.record_failure("test-recent.example.com")
        assert result["failure_count"] == 3  # 2 + 1
        assert result["blacklisted"] is True  # Hit threshold

    finally:
        from sqlalchemy import delete
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-recent.example.com")
            )
            await sess.commit()
        await db_obj.close()


@pytest.mark.asyncio
async def test_cleanup_old_blacklisted(db):
    """cleanup_old_blacklisted removes only old blacklisted domains."""
    from src.db.database import Database
    from src.db.models import Domain

    db_obj = Database()
    await db_obj.init()

    try:
        now = datetime.now(timezone.utc)

        # Insert: old blacklisted (should be cleaned)
        async with db_obj._get_session() as sess:
            sess.add(Domain(
                domain="old-blacklisted.example.com",
                is_blacklisted=True,
                updated_at=now - timedelta(days=5),
            ))
            # Insert: recent blacklisted (should NOT be cleaned)
            sess.add(Domain(
                domain="recent-blacklisted.example.com",
                is_blacklisted=True,
                updated_at=now - timedelta(hours=1),
            ))
            await sess.commit()

        count = await db_obj.cleanup_old_blacklisted(days_old=2)
        assert count == 1

        # Verify the recent one survived
        assert await db_obj.is_blacklisted("recent-blacklisted.example.com") is True

    finally:
        from sqlalchemy import delete
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(
                    Domain.domain.in_([
                        "old-blacklisted.example.com",
                        "recent-blacklisted.example.com",
                    ])
                )
            )
            await sess.commit()
        await db_obj.close()


# ---------- MCP Tool Tests ----------


@pytest.mark.asyncio
async def test_clear_blacklist_tool(mcp_client, session):
    """clear_blacklist tool unblacklists all domains."""
    from src.db.database import Database

    db_obj = Database()
    await db_obj.init()

    try:
        # Blacklist a domain
        await db_obj.blacklist("test-clear.example.com")
        assert await db_obj.is_blacklisted("test-clear.example.com") is True

        # Call the tool
        result = await call_tool(
            mcp_client, session, "clear_blacklist", {}
        )
        assert result["status"] == "success"
        assert result["domains_unblacklisted"] >= 1

        # Verify cleared
        assert await db_obj.is_blacklisted("test-clear.example.com") is False

    finally:
        from sqlalchemy import delete
        from src.db.models import Domain
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-clear.example.com")
            )
            await sess.commit()
        await db_obj.close()


@pytest.mark.asyncio
async def test_get_domains_shows_blacklist_status(mcp_client, session):
    """domains tool returns blacklist status for each domain."""
    from src.db.database import Database

    db_obj = Database()
    await db_obj.init()

    try:
        await db_obj.record_success("test-getdom.example.com", "crawl4ai")

        # Flush Redis cache so the domains tool reads fresh from DB
        import redis as redis_lib
        r = redis_lib.Redis(host="redis", port=6379)
        for key in r.scan_iter(match="mcp-server__tools/call::*"):
            r.delete(key)
        r.close()

        result = await call_tool(mcp_client, session, "domains", {})
        assert result["total"] >= 1

        # Find our test domain
        domains = result["domains"]
        test_domain = next(
            (d for d in domains if d["domain"] == "test-getdom.example.com"), None
        )
        assert test_domain is not None
        assert test_domain["preferred_method"] == "crawl4ai"
        assert test_domain["is_blacklisted"] is False

    finally:
        from sqlalchemy import delete
        from src.db.models import Domain
        async with db_obj._get_session() as sess:
            await sess.execute(
                delete(Domain).where(Domain.domain == "test-getdom.example.com")
            )
            await sess.commit()
        await db_obj.close()
