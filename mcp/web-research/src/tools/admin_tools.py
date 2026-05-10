"""Admin Tools

Administrative tools for managing the MCP server.
- domains: List tracked domains with preferred methods
- stats: View scrape statistics and metrics
- reset: Clear all domain tracking data
- clear_blacklist: Clear all blacklisted domains (unblock them)

Note: Domain tracking uses PostgreSQL database shared with Celery workers.
"""

from typing import Annotated, Literal

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field


async def domains(ctx: Context | None = None) -> dict:
    """List all tracked domains with their preferred scraping methods

    Returns:
        Dictionary with total count and list of domain records

    Note:
        This data is managed in PostgreSQL database and is not cached.
    """
    if ctx:
        await ctx.debug("Fetching all tracked domains")

    # Get db from lifespan context with safe access
    db = ctx.lifespan_context.get("db")
    if not db:
        raise ToolError("Database service not available")

    domain_list = await db.get_all_domains()

    if ctx:
        await ctx.debug(f"Retrieved {len(domain_list)} domains")

    return {
        "total": len(domain_list),
        "domains": domain_list
    }


async def stats(
    hours: Annotated[int, Field(
        description="Time period in hours (default: 24)",
        ge=1,
        le=168
    )] = 24,
    ctx: Context | None = None
) -> dict:
    """Get scrape statistics and metrics for monitoring

    Shows performance metrics including:
    - Total scrapes, success rate
    - Average duration (p50, p95, p99 percentiles)
    - Breakdown by scraping method
    - Top failing domains

    Args:
        hours: Time period to analyze (1-168 hours, default 24)

    Returns:
        Dictionary with comprehensive scrape statistics

    Note:
        Metrics are stored in PostgreSQL database.
    """
    if ctx:
        await ctx.info(f"Fetching scrape stats for past {hours} hours")

    # Get db from lifespan context with safe access
    db = ctx.lifespan_context.get("db")
    if not db:
        raise ToolError("Database service not available")

    scrape_stats = await db.get_scrape_stats(hours=hours)

    if ctx:
        await ctx.info(
            f"Stats: {scrape_stats['total_scrapes']} scrapes, "
            f"{scrape_stats['success_rate']}% success rate, "
            f"avg {scrape_stats['avg_duration_ms']}ms"
        )

    return scrape_stats


async def reset(ctx: Context | None = None) -> dict:
    """Clear all domain tracking data

    This resets all learned scraping methods and blacklist entries.
    Use this to start fresh.

    Returns:
        Dictionary with status and count of removed records

    Warning:
        This operation cannot be undone. All learned domain preferences
        will be lost and must be re-learned through scraping.
    """
    # Get db from lifespan context with safe access
    db = ctx.lifespan_context.get("db")
    if not db:
        raise ToolError("Database service not available")

    count = await db.clean()

    if ctx:
        await ctx.info(f"Cleaned {count} domain records")

    return {
        "status": "success",
        "records_removed": count
    }


async def clear_blacklist(ctx: Context | None = None) -> dict:
    """Clear all blacklisted domains - unblock them immediately

    This resets the blacklist for ALL domains, allowing them to be
    scraped again. Useful if domains were incorrectly blacklisted
    or if you want to retry after fixing connection issues.

    The changes take effect immediately - no server restart required.
    Also clears the FastMCP response cache so unblocked domains work immediately.

    Returns:
        Dictionary with status and count of unblacklisted domains

    Note:
        This is a safer alternative to reset which removes
        all learned data. clear_blacklist only resets the blacklist
        while keeping learned scraping methods.
    """
    if ctx:
        await ctx.info("Clearing all blacklisted domains...")

    # Get db from lifespan context with safe access
    db = ctx.lifespan_context.get("db")
    if not db:
        raise ToolError("Database service not available")

    # Get Redis client for cache clearing
    redis = None
    try:
        from key_value.aio.stores.redis import RedisStore
        from ..settings import get_settings
        settings = get_settings()
        redis = RedisStore(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=0,
        )
        redis = await redis.get_client()
    except Exception as e:
        from loguru import logger
        logger.debug(f"Redis client not available for cache clearing: {e}")

    count = await db.clear_blacklist(redis=redis)

    if ctx:
        await ctx.info(f"Cleared blacklist for {count} domains")

    return {
        "status": "success",
        "domains_unblacklisted": count
    }
