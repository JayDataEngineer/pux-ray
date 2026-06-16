"""Unified scraping service with method routing and consistent output

Flow:
1. Check URL-level Redis cache -> return cached if hit
2. Rate limiting (max 3 concurrent per domain)
3. Check blacklist -> reject if blacklisted
4. Reddit -> special JSON API handler
5. Check database -> use learned preference
6. Try Crawl4AI (fast)
7. Fallback to Selenium (stealth)
8. Blacklist if both fail
9. Store successful scrape in Redis cache (24h TTL)

Note: Rate limiting uses in-memory semaphores (no Redis required).
Note: Domain tracking uses PostgreSQL (shared with Celery workers).
"""

import hashlib
import json

from loguru import logger

from ..models.unified import ScrapeRequest, ScrapeResponse, ScrapingMethod
from ..services.content_cleaner import get_content_cleaner
from ..utils.rate_limiter import get_rate_limiter
from ..scrapers.base import scrape_with_fallback
from ..utils import extract_domain, create_singleton_factory

# Cache successful scrapes for 24 hours
SCRAPE_CACHE_TTL = 86400


def _cache_key(url: str, text_only: bool) -> str:
    """Generate Redis cache key from URL + options."""
    h = hashlib.sha256(f"{url}:{text_only}".encode()).hexdigest()
    return f"scrape_cache:{h}"


class UnifiedScrapeService:
    """Unified scraping with consistent output format

    URL-level Redis cache prevents repeated scrapes of the same URL.
    Rate limiting uses in-memory semaphores (no Redis required).
    Domain tracking uses PostgreSQL (shared with Celery workers).
    """

    def __init__(self, db=None, cleaner=None):
        self._db = db
        self._cleaner = cleaner
        self._db_instance = None
        self._redis = None

    async def _get_db(self):
        if self._db is not None:
            return self._db
        if self._db_instance is None:
            from ..db.database import get_db
            try:
                self._db_instance = await get_db()
            except Exception:
                from loguru import logger
                logger.warning("Scrape service: database unavailable, domain learning disabled")
                return None
        return self._db_instance

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                from ..settings import get_settings
                s = get_settings()
                self._redis = aioredis.Redis(
                    host=s.redis_host,
                    port=s.redis_port,
                    password=s.redis_password or None,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning(f"Redis unavailable for scrape cache: {e}")
        return self._redis

    @property
    def cleaner(self):
        if self._cleaner is None:
            self._cleaner = get_content_cleaner()
        return self._cleaner

    async def scrape(self, request: ScrapeRequest) -> ScrapeResponse:
        """Main scrape entry point with URL-level cache and rate limiting"""
        cache_k = _cache_key(request.url, request.text_only)

        # Check URL cache first
        redis = await self._get_redis()
        if redis:
            try:
                cached = await redis.get(cache_k)
                if cached:
                    logger.debug(f"Scrape cache HIT: {request.url}")
                    return self._dict_to_response(json.loads(cached))
            except Exception:
                pass

        db = await self._get_db()
        domain = extract_domain(request.url)
        rate_limiter = get_rate_limiter()

        acquired = await rate_limiter.acquire(domain)
        if not acquired:
            return ScrapeResponse(
                success=False,
                url=request.url,
                domain=domain,
                method_used=ScrapingMethod.CRAWL4AI,
                error="Rate limit: Too many concurrent requests to this domain.",
            )

        try:
            result_dict = await scrape_with_fallback(
                url=request.url,
                cleaner=self.cleaner,
                db=db,
                force_method=request.force_method.value if request.force_method else None,
                css_selector=request.css_selector,
                text_only=request.text_only
            )

            response = self._dict_to_response(result_dict)

            # Cache successful scrapes
            if response.success and response.content and redis:
                try:
                    await redis.setex(cache_k, SCRAPE_CACHE_TTL, json.dumps(result_dict))
                    logger.debug(f"Scrape cached (24h): {request.url}")
                except Exception:
                    pass

            return response

        except Exception as e:
            # Catch any unexpected exceptions to prevent TaskGroup errors
            from loguru import logger

            logger.error(f"Unexpected error in scrape_service for {request.url}: {e}")

            return ScrapeResponse(
                success=False,
                url=request.url,
                domain=domain,
                method_used=ScrapingMethod.CRAWL4AI,
                error=f"Internal error: {str(e)[:200]}" if len(str(e)) < 200 else "Internal error during scraping",
            )

        finally:
            await rate_limiter.release(domain)

    def _dict_to_response(self, data: dict) -> ScrapeResponse:
        """Convert dict result to ScrapeResponse"""
        method_str = data.get("method_used", "crawl4ai")
        try:
            method = ScrapingMethod(method_str)
        except ValueError:
            method = ScrapingMethod.CRAWL4AI

        return ScrapeResponse(
            success=data.get("success", False),
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            method_used=method,
            title=data.get("title"),
            content=data.get("content"),
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )


# Singleton factory
get_scrape_service = create_singleton_factory(UnifiedScrapeService, "get_scrape_service")
