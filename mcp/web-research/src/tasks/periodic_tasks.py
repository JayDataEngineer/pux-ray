"""Periodic Celery tasks for maintenance operations"""

import httpx
from loguru import logger

from ..celery_app import app
from ..tasks.base import BaseTask


def _check_redis() -> dict:
    """Check Redis connectivity"""
    try:
        import redis.asyncio as aioredis
        import os

        redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
        import asyncio

        async def _ping():
            client = await aioredis.from_url(redis_url)
            await client.ping()
            await client.close()

        asyncio.run(_ping())
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": f"error: {e}"}


def _check_searxng() -> dict:
    """Check SearXNG service availability"""
    try:
        import os

        searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")
        response = httpx.get(f"{searxng_url}/search", params={"q": "test", "format": "json"}, timeout=5.0)
        response.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"SearXNG health check failed: {e}")
        return {"status": f"error: {e}"}


@app.task(bind=True, base=BaseTask, name="tasks.periodic.cleanup_blacklist")
def cleanup_blacklist(self, days_old: int = 2) -> dict:
    """
    Periodic task to clean up old blacklisted domains

    Removes blacklisted domains that haven't been updated in X days.
    This prevents the blacklist from growing indefinitely.

    Args:
        days_old: Remove domains older than this many days

    Returns:
        Dict with cleanup results
    """
    count = self.run_async(lambda: self.db.cleanup_old_blacklisted(days_old))
    result = {
        "status": "success",
        "days_old": days_old,
        "domains_removed": count
    }
    logger.info(f"Periodic cleanup completed: {result}")
    return result


@app.task(bind=True, base=BaseTask, name="tasks.periodic.health_check")
def health_check(self) -> dict:
    """
    Periodic health check task

    Verifies that core services are responsive:
    - Database (PostgreSQL)
    - Redis (Celery broker/cache)
    - SearXNG (search engine)
    """
    # Check Database
    try:
        domains_count = self.run_async(lambda: self.db.get_all_domains())
        db_status = "ok"
        count = len(domains_count)
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"error: {e}"
        count = 0

    # Check Redis
    redis_status = _check_redis()

    # Check SearXNG
    searxng_status = _check_searxng()

    result = {
        "status": "success",
        "database": db_status,
        "domains_count": count,
        "redis": redis_status.get("status", "unknown"),
        "searxng": searxng_status.get("status", "unknown"),
    }

    overall_health = "ok" if all(
        s == "ok" for s in [db_status, redis_status.get("status", "unknown"), searxng_status.get("status", "unknown")]
    ) else "degraded"

    result["overall"] = overall_health
    logger.info(f"Health check completed: {overall_health} - DB: {db_status}, Redis: {redis_status.get('status')}, SearXNG: {searxng_status.get('status')}")

    return result


@app.task(bind=True, base=BaseTask, name="tasks.periodic.cleanup_old_metrics")
def cleanup_old_metrics(self, days: int = 7) -> dict:
    """
    Periodic task to clean up old scrape metrics

    Removes scrape metrics older than specified days to keep
    the database size manageable.

    Args:
        days: Keep metrics newer than this many days (default: 7)

    Returns:
        Dict with cleanup results
    """
    count = self.run_async(lambda: self.db.cleanup_old_metrics(days))
    result = {
        "status": "success",
        "days": days,
        "metrics_removed": count
    }
    logger.info(f"Periodic metrics cleanup completed: {result}")
    return result
