"""Redis-backed domain-based rate limiting for scrape requests

Prevents overwhelming single domains with concurrent requests.
Uses semaphores stored in Redis to track active requests per domain.
"""

import asyncio
from typing import Optional
from loguru import logger

from ..utils.redis_mixin import RedisMixin


class RateLimitService(RedisMixin):
    """
    Domain-based rate limiting using Redis

    Tracks concurrent requests per domain using Redis semaphores.
    When a domain has too many concurrent requests, new requests wait.
    """

    def __init__(
        self,
        redis_url: str = None,
        max_concurrent: int = 3,
        acquire_timeout: float = 30.0,
        ttl: int = 300
    ):
        """
        Initialize rate limit service

        Args:
            redis_url: Redis URL (default from env or redis://localhost:6379/0)
            max_concurrent: Max concurrent requests per domain (default: 3)
            acquire_timeout: Max seconds to wait for permit (default: 30)
            ttl: Lock TTL in seconds - auto-releases if process crashes (default: 300)
        """
        super().__init__(redis_url)
        self.max_concurrent = max_concurrent
        self.acquire_timeout = acquire_timeout
        self.ttl = ttl

    async def acquire(
        self,
        domain: str,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire a permit for scraping this domain

        Args:
            domain: Domain to scrape
            timeout: Max seconds to wait (uses default if None)

        Returns:
            True if permit acquired, False if timeout exceeded
        """
        if timeout is None:
            timeout = self.acquire_timeout

        redis = await self._get_redis()
        key = f"ratelimit:domain:{domain}"

        start_time = asyncio.get_event_loop().time()
        deadline = start_time + timeout

        while True:
            now = asyncio.get_event_loop().time()

            # Check timeout
            if now >= deadline:
                logger.warning(f"Rate limit timeout for {domain}")
                return False

            # Try to acquire a slot
            current = await redis.incr(key)

            if current == 1:
                # First request - set expiry
                await redis.expire(key, self.ttl)

            if current <= self.max_concurrent:
                # Acquired!
                logger.debug(f"Rate limit acquired for {domain} ({current}/{self.max_concurrent} active)")
                return True

            # Too many requests - decrement and wait
            await redis.decr(key)
            wait_time = min(0.5, deadline - now)
            if wait_time <= 0:
                logger.warning(f"Rate limit timeout for {domain}")
                return False
            await asyncio.sleep(wait_time)

    async def release(self, domain: str):
        """
        Release permit for this domain

        Args:
            domain: Domain being scraped
        """
        try:
            redis = await self._get_redis()
            key = f"ratelimit:domain:{domain}"

            current = await redis.decr(key)
            logger.debug(f"Rate limit released for {domain} ({max(0, current)}/{self.max_concurrent} active)")

            # Clean up if no active requests
            if current <= 0:
                await redis.delete(key)

        except Exception as e:
            logger.error(f"Error releasing rate limit for {domain}: {e}")

    async def get_active_count(self, domain: str) -> int:
        """Get current active request count for domain"""
        try:
            redis = await self._get_redis()
            key = f"ratelimit:domain:{domain}"
            value = await redis.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Error getting active count for {domain}: {e}")
            return 0

    async def get_all_active(self) -> dict:
        """Get all domains with active requests"""
        try:
            redis = await self._get_redis()
            keys = await redis.keys("ratelimit:domain:*")

            result = {}
            for key in keys:
                domain = key.replace("ratelimit:domain:", "")
                count = await redis.get(key)
                if count:
                    result[domain] = int(count)
            return result

        except Exception as e:
            logger.error(f"Error getting all active domains: {e}")
            return {}

    async def clear_domain(self, domain: str):
        """Clear all permits for a domain (emergency use)"""
        try:
            redis = await self._get_redis()
            key = f"ratelimit:domain:{domain}"
            await redis.delete(key)
            logger.info(f"Cleared rate limit for {domain}")
        except Exception as e:
            logger.error(f"Error clearing rate limit for {domain}: {e}")


# Singleton factory
from ..utils import create_singleton_factory
get_rate_limit_service = create_singleton_factory(RateLimitService, "get_rate_limit_service")
