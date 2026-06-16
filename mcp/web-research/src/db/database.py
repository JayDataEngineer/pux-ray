"""PostgreSQL database for domain tracking and blacklist

Uses SQLAlchemy 2.0 async ORM instead of raw SQL.
"""

from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger
import os

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update, delete, func, case
from sqlalchemy.orm import selectinload

from .models import Domain, ScrapeMetric
from ..core.constants import BLACKLIST_FAILURE_THRESHOLD


class Database:
    """PostgreSQL database for tracking scraping methods and blacklist"""

    def __init__(
        self,
        db_url: str = None,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize PostgreSQL database connection

        Args:
            db_url: Full database URL (overrides other params)
            host: Database host (default from env or postgres)
            port: Database port (default from env or 5432)
            database: Database name (default from env or mcp_server)
            user: Database user (default from env or postgres)
            password: Database password (default from env or postgres)
        """
        if db_url:
            # Convert postgresql:// to postgresql+asyncpg:// for SQLAlchemy
            if not db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            self.db_url = db_url
        else:
            host = host or os.getenv("POSTGRES_HOST", "postgres")
            port = port or os.getenv("POSTGRES_PORT", "5432")
            database = database or os.getenv("POSTGRES_DB", "mcp_server")
            user = user or os.getenv("POSTGRES_USER", "postgres")
            # Require password from env or parameter - no default
            password = password or os.getenv("POSTGRES_PASSWORD")
            if not password:
                raise ValueError("POSTGRES_PASSWORD environment variable must be set")

            self.db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

        self._engine = None
        self._sessionmaker = None

    async def init(self):
        """Initialize database schema and connection pool"""
        from .models import Base

        self._engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )

        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=True,  # Force fresh queries after commit (fixes stale blacklist cache)
        )

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info(f"PostgreSQL initialized (SQLAlchemy 2.0): {self.db_url}")

    async def close(self):
        """Close the connection pool"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            logger.info("PostgreSQL connection pool closed")

    def _get_session(self) -> AsyncSession:
        """Get a new database session"""
        if self._sessionmaker is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._sessionmaker()

    async def get_domain_method(self, domain: str) -> Optional[str]:
        """Get preferred scraping method for domain"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain.preferred_method)
                .where(Domain.domain == domain)
                .where(Domain.is_blacklisted == False)
            )
            return result.scalar_one_or_none()

    async def record_success(self, domain: str, method: str):
        """Record successful scrape — upsert to handle concurrent inserts safely."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from .models import Domain as DomainModel

        async with self._get_session() as session:
            stmt = pg_insert(DomainModel).values(
                domain=domain,
                preferred_method=method,
                last_success=datetime.now(timezone.utc),
                failure_count=0,
                is_blacklisted=False,
            ).on_conflict_do_update(
                index_elements=["domain"],
                set_={
                    "preferred_method": method,
                    "last_success": datetime.now(timezone.utc),
                    "failure_count": 0,
                    "is_blacklisted": False,
                }
            )
            await session.execute(stmt)
            await session.commit()
            logger.debug(f"Success recorded: {domain} -> {method}")

    async def set_selenium_only(self, domain: str):
        """Mark domain as selenium-only — upsert to handle concurrent inserts."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from .models import Domain as DomainModel

        async with self._get_session() as session:
            stmt = pg_insert(DomainModel).values(
                domain=domain,
                preferred_method="selenium",
            ).on_conflict_do_update(
                index_elements=["domain"],
                set_={"preferred_method": "selenium"}
            )
            await session.execute(stmt)
            await session.commit()

    async def blacklist(self, domain: str):
        """Blacklist a domain - all scraping failed"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain).where(Domain.domain == domain)
            )
            db_domain = result.scalar_one_or_none()

            if db_domain:
                db_domain.is_blacklisted = True
            else:
                new_domain = Domain(domain=domain, is_blacklisted=True)
                session.add(new_domain)

            await session.commit()
            logger.warning(f"Blacklisted: {domain}")

    async def record_failure(self, domain: str, method: str = "unknown") -> dict:
        """Record a scrape failure and blacklist if threshold exceeded

        Failures older than 24 hours are reset before counting — only failures
        within a 24-hour window count toward the blacklist threshold. This
        prevents VPN blips and temporary outages from permanently blacklisting
        a domain.

        Args:
            domain: The domain that failed
            method: The method that was attempted

        Returns:
            Dict with 'blacklisted' bool and 'failure_count' int
        """
        from ..core.constants import BLACKLIST_FAILURE_THRESHOLD

        async with self._get_session() as session:
            result = await session.execute(
                select(Domain).where(Domain.domain == domain)
            )
            db_domain = result.scalar_one_or_none()

            if db_domain:
                # Reset failure count if last failure was >24h ago (stale failures don't count)
                now = datetime.now(timezone.utc)
                if db_domain.last_failure:
                    hours_since = (now - db_domain.last_failure).total_seconds() / 3600
                    if hours_since > 24:
                        if db_domain.failure_count > 0:
                            logger.info(
                                f"Resetting stale failure count for {domain} "
                                f"({db_domain.failure_count} failures, last was {hours_since:.0f}h ago)"
                            )
                        db_domain.failure_count = 0

                # Increment failure count
                new_count = db_domain.failure_count + 1
                db_domain.failure_count = new_count
                db_domain.last_failure = now

                # Only blacklist if threshold exceeded
                if new_count >= BLACKLIST_FAILURE_THRESHOLD:
                    db_domain.is_blacklisted = True
                    await session.commit()
                    logger.warning(
                        f"Blacklisted {domain} after {new_count} failures within 24h "
                        f"(threshold: {BLACKLIST_FAILURE_THRESHOLD})"
                    )
                    return {"blacklisted": True, "failure_count": new_count}

                await session.commit()
                logger.debug(f"Failure {new_count}/{BLACKLIST_FAILURE_THRESHOLD} for {domain}")
                return {"blacklisted": False, "failure_count": new_count}
            else:
                # New domain with failure - don't blacklist on first failure
                new_domain = Domain(
                    domain=domain,
                    preferred_method="crawl4ai",
                    failure_count=1,
                    last_failure=datetime.now(timezone.utc),
                    is_blacklisted=False,
                )
                session.add(new_domain)
                await session.commit()
                logger.debug(f"First failure recorded for {domain}")
                return {"blacklisted": False, "failure_count": 1}

    async def is_blacklisted(self, domain: str) -> bool:
        """Check if domain is blacklisted"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain.is_blacklisted)
                .where(Domain.domain == domain)
            )
            return result.scalar_one_or_none() or False

    async def get_blacklisted_domains(self) -> set:
        """Get all blacklisted domains as a set"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain.domain)
                .where(Domain.is_blacklisted == True)
            )
            return set(result.scalars().all())

    async def get_all_domains(self) -> List[dict]:
        """Get all domain records"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain)
                .order_by(Domain.created_at.desc())
            )
            domains = result.scalars().all()
            return [d.to_dict() for d in domains]

    async def cleanup_old_blacklisted(self, days_old: int = 2) -> int:
        """Remove blacklisted domains older than specified days (default 2)"""
        from datetime import timedelta

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)

        async with self._get_session() as session:
            # Find blacklisted domains with updated_at older than cutoff
            result = await session.execute(
                select(Domain.domain)
                .where(Domain.is_blacklisted == True)
                .where(Domain.updated_at < cutoff_date)
            )
            old_domains = result.scalars().all()

            if not old_domains:
                return 0

            # Delete old blacklisted domains
            delete_result = await session.execute(
                delete(Domain)
                .where(Domain.domain.in_(old_domains))
            )
            await session.commit()

            count = delete_result.rowcount
            logger.info(f"Cleaned up {count} blacklisted domains older than {days_old} days")
            return count

    async def clear_blacklist(self, redis=None) -> int:
        """Clear ALL blacklisted domains - unblacklist everything immediately

        This is useful when you want to reset the blacklist and allow
        retrying all domains that were previously blocked.

        Args:
            redis: Optional Redis client to clear FastMCP cache. If provided,
                   will invalidate cached responses for unblacklisted domains.

        Returns:
            Number of domains that were unblacklisted
        """
        async with self._get_session() as session:
            # Find all blacklisted domains
            result = await session.execute(
                select(Domain).where(Domain.is_blacklisted == True)
            )
            blacklisted = result.scalars().all()

            if not blacklisted:
                return 0

            # Collect domain names for cache clearing
            domain_names = [d.domain for d in blacklisted]

            # Update all to unblacklisted
            count = 0
            for domain in blacklisted:
                domain.is_blacklisted = False
                domain.failure_count = 0
                count += 1

            await session.commit()

            # Clear FastMCP response cache for affected domains
            if redis and domain_names:
                try:
                    # Delete all cache keys matching any of the domains
                    for domain in domain_names:
                        # FastMCP cache keys are hashed, so we can't easily match by domain.
                        # Just flush the entire tools/call cache namespace.
                        break
                    # Flush all tool call caches to ensure fresh results
                    async for key in redis.scan_iter(match="mcp-server__tools/call::*", count=100):
                        await redis.delete(key)
                    logger.info(f"Cleared FastMCP tool cache for {len(domain_names)} domains")
                except Exception as e:
                    logger.warning(f"Failed to clear FastMCP cache: {e}")

            logger.info(f"Cleared blacklist for {count} domains")
            return count

    async def clean(self) -> int:
        """Clean all records from database"""
        async with self._get_session() as session:
            result = await session.execute(
                delete(Domain)
            )
            await session.commit()
            count = result.rowcount
            logger.info(f"Database cleaned: {count} records removed")
            return count

    async def check_urls(
        self,
        max_urls: Optional[int] = None,
        threshold: int = None
    ) -> dict:
        """
        Check URLs in database to verify scraping still works

        For each domain:
        - Try to scrape
        - If success: update timestamp
        - If failure: increment count
        - If failures >= threshold: blacklist
        """
        from ..services.scrape_service import get_scrape_service
        from ..models.unified import ScrapeRequest

        if threshold is None:
            threshold = BLACKLIST_FAILURE_THRESHOLD

        scrape_svc = get_scrape_service()
        domains = await self.get_all_domains()

        stats = {
            "total_checked": 0,
            "still_valid": 0,
            "moved_to_selenium": 0,
            "blacklisted": 0,
            "details": []
        }

        for record in domains[:max_urls]:
            domain = record["domain"]

            # Skip already blacklisted
            if record["is_blacklisted"]:
                continue

            stats["total_checked"] += 1

            # Test URL (use domain root or construct one)
            test_url = f"https://{domain}/"

            try:
                result = await scrape_svc.scrape(
                    ScrapeRequest(url=test_url)
                )

                if result.success:
                    await self.record_success(domain, result.method_used.value)
                    stats["still_valid"] += 1
                    stats["details"].append({
                        "domain": domain,
                        "status": "valid",
                        "method": result.method_used.value
                    })
                else:
                    # Increment failure count
                    new_count = record["failure_count"] + 1
                    await self._increment_failure(domain, new_count)

                    if new_count >= threshold:
                        await self.blacklist(domain)
                        stats["blacklisted"] += 1
                        stats["details"].append({
                            "domain": domain,
                            "status": "blacklisted",
                            "failures": new_count
                        })
                    else:
                        stats["details"].append({
                            "domain": domain,
                            "status": "failed",
                            "failures": new_count
                        })

            except Exception as e:
                logger.error(f"Check failed for {domain}: {e}")
                stats["details"].append({
                    "domain": domain,
                    "status": "error",
                    "error": str(e)
                })

        return stats

    async def _increment_failure(self, domain: str, count: int):
        """Increment failure counter"""
        async with self._get_session() as session:
            result = await session.execute(
                select(Domain).where(Domain.domain == domain)
            )
            db_domain = result.scalar_one_or_none()

            if db_domain:
                db_domain.failure_count = count
                db_domain.last_failure = datetime.now(timezone.utc)
            else:
                new_domain = Domain(
                    domain=domain,
                    failure_count=count,
                    last_failure=datetime.now(timezone.utc)
                )
                session.add(new_domain)

            await session.commit()

    # ========== Telemetry Methods ==========

    async def record_scrape_metric(
        self,
        url: str,
        domain: str,
        method: str,
        success: bool,
        duration_ms: float,
        content_length: int = None,
        error: str = None
    ):
        """Record a scrape metric for telemetry

        Args:
            url: The URL that was scraped
            domain: The domain of the URL
            method: The scraping method used (crawl4ai, selenium, pdf, reddit_api)
            success: Whether the scrape succeeded
            duration_ms: Duration in milliseconds
            content_length: Length of content returned (if successful)
            error: Error message (if failed)
        """
        from ..utils import extract_domain

        # Ensure domain is extracted properly
        if not domain or "." not in domain:
            domain = extract_domain(url)

        async with self._get_session() as session:
            metric = ScrapeMetric(
                url=url[:2048],  # Truncate if too long
                domain=domain[:255],
                method=method,
                success=success,
                duration_ms=duration_ms,
                content_length=content_length,
                error=error[:500] if error else None
            )
            session.add(metric)
            await session.commit()

    async def get_scrape_stats(self, hours: int = 24) -> dict:
        """Get scrape statistics for the past N hours

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            Dict with statistics including totals, averages, success rate
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        async with self._get_session() as session:
            # Total scrapes
            total_result = await session.execute(
                select(func.count())
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
            )
            total = total_result.scalar() or 0

            # Successful scrapes
            success_result = await session.execute(
                select(func.count())
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
                .where(ScrapeMetric.success == True)
            )
            successful = success_result.scalar() or 0

            # Failed scrapes
            failed = total - successful

            # Average duration
            avg_duration_result = await session.execute(
                select(func.avg(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
            )
            avg_duration = avg_duration_result.scalar() or 0

            # Average duration by success
            avg_success_duration_result = await session.execute(
                select(func.avg(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
                .where(ScrapeMetric.success == True)
            )
            avg_success_duration = avg_success_duration_result.scalar() or 0

            avg_fail_duration_result = await session.execute(
                select(func.avg(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
                .where(ScrapeMetric.success == False)
            )
            avg_fail_duration = avg_fail_duration_result.scalar() or 0

            # Count by method
            by_method_result = await session.execute(
                select(
                    ScrapeMetric.method,
                    func.count().label('count'),
                    func.avg(ScrapeMetric.duration_ms).label('avg_duration'),
                    func.sum(case((ScrapeMetric.success == True, 1), else_=0)).label('successes')
                )
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
                .group_by(ScrapeMetric.method)
            )
            by_method = []
            for row in by_method_result:
                by_method.append({
                    "method": row.method,
                    "count": row.count,
                    "avg_duration_ms": round(row.avg_duration, 2) if row.avg_duration else 0,
                    "successes": row.successes or 0,
                    "failures": row.count - (row.successes or 0)
                })

            # Top failing domains
            top_fails_result = await session.execute(
                select(
                    ScrapeMetric.domain,
                    func.count().label('fail_count')
                )
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
                .where(ScrapeMetric.success == False)
                .group_by(ScrapeMetric.domain)
                .order_by(func.count().desc())
                .limit(10)
            )
            top_failing = [
                {"domain": row.domain, "failures": row.fail_count}
                for row in top_fails_result
            ]

            # Percentiles (p50, p95, p99)
            p50_result = await session.execute(
                select(func.percentile_cont(0.5).within_group(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
            )
            p50 = p50_result.scalar() or 0

            p95_result = await session.execute(
                select(func.percentile_cont(0.95).within_group(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
            )
            p95 = p95_result.scalar() or 0

            p99_result = await session.execute(
                select(func.percentile_cont(0.99).within_group(ScrapeMetric.duration_ms))
                .select_from(ScrapeMetric)
                .where(ScrapeMetric.created_at >= cutoff_time)
            )
            p99 = p99_result.scalar() or 0

            return {
                "period_hours": hours,
                "total_scrapes": total,
                "successful": successful,
                "failed": failed,
                "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
                "avg_duration_ms": round(avg_duration, 2),
                "avg_success_duration_ms": round(avg_success_duration, 2),
                "avg_fail_duration_ms": round(avg_fail_duration, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "by_method": by_method,
                "top_failing_domains": top_failing,
            }

    async def cleanup_old_metrics(self, days: int = 7) -> int:
        """Remove scrape metrics older than specified days

        Args:
            days: Keep metrics newer than this many days

        Returns:
            Number of metrics removed
        """
        from datetime import timedelta

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        async with self._get_session() as session:
            result = await session.execute(
                delete(ScrapeMetric)
                .where(ScrapeMetric.created_at < cutoff_date)
            )
            await session.commit()
            count = result.rowcount
            logger.info(f"Cleaned up {count} scrape metrics older than {days} days")
            return count


# Singleton
_db: Database = None

# Sentinel: a singleton "null" Database instance returned when Postgres
# is unavailable.  Every method is a no-op so the rest of the codebase
# (scrape_with_fallback, admin tools, etc.) never needs to check for None.
class _NullDatabase:
    """No-op database — used when Postgres is unreachable.

    All methods accept the same signature as the real Database class but
    silently return empty/default values so callers never need None-checks.
    """
    async def init(self): pass
    async def close(self): pass
    async def get_domain_method(self, domain): return None
    async def record_success(self, domain, method): pass
    async def set_selenium_only(self, domain): pass
    async def blacklist(self, domain): pass
    async def record_failure(self, domain, method="unknown"): return {"blacklisted": False, "failure_count": 0}
    async def is_blacklisted(self, domain): return False
    async def get_blacklisted_domains(self): return set()
    async def get_all_domains(self): return []
    async def cleanup_old_blacklisted(self, days_old=2): return 0
    async def clear_blacklist(self, redis=None): return 0
    async def clean(self): return 0
    async def check_urls(self, max_urls=None, threshold=None): return {"total_checked": 0, "still_valid": 0, "moved_to_selenium": 0, "blacklisted": 0, "details": []}
    async def record_scrape_metric(self, **kwargs): pass
    async def get_scrape_stats(self, hours=24): return {"period_hours": hours, "total_scrapes": 0, "successful": 0, "failed": 0, "success_rate": 0, "avg_duration_ms": 0, "avg_success_duration_ms": 0, "avg_fail_duration_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "by_method": [], "top_failing_domains": []}
    async def cleanup_old_metrics(self, days=7): return 0

_NULL_DB = _NullDatabase()


async def get_db() -> Database:
    global _db
    if _db is None:
        try:
            _db = Database()
            await _db.init()
        except Exception:
            from loguru import logger
            logger.warning("PostgreSQL unavailable — returning null database (domain tracking disabled)")
            return _NULL_DB
    return _db
