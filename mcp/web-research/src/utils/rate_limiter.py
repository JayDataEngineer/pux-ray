"""Simple in-memory rate limiting for scrape requests

Prevents overwhelming single domains with concurrent requests.
Uses asyncio.Semaphore for in-memory rate limiting (no Redis required).
"""

import asyncio
from collections import defaultdict
from typing import Optional
from loguru import logger


class SimpleRateLimiter:
    """
    Domain-based rate limiting using in-memory semaphores

    Tracks concurrent requests per domain using asyncio.Semaphore.
    When a domain has too many concurrent requests, new requests wait.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        acquire_timeout: float = 30.0,
    ):
        """
        Initialize rate limiter

        Args:
            max_concurrent: Max concurrent requests per domain (default: 3)
            acquire_timeout: Max seconds to wait for permit (default: 30)
        """
        self.max_concurrent = max_concurrent
        self.acquire_timeout = acquire_timeout
        self.semaphores = defaultdict(lambda: asyncio.Semaphore(max_concurrent))

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

        semaphore = self.semaphores[domain]

        try:
            # Wait with timeout
            await asyncio.wait_for(semaphore.acquire(), timeout)
            logger.debug(f"Rate limit acquired for {domain}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Rate limit timeout for {domain}")
            return False

    async def release(self, domain: str):
        """
        Release permit for this domain

        Args:
            domain: Domain being scraped
        """
        try:
            semaphore = self.semaphores[domain]
            semaphore.release()
            logger.debug(f"Rate limit released for {domain}")
        except Exception as e:
            logger.error(f"Error releasing rate limit for {domain}: {e}")

    async def get_active_count(self, domain: str) -> int:
        """Get current active request count for domain

        Note: This is an approximation based on semaphore state
        """
        semaphore = self.semaphores[domain]
        # Get the internal counter: max_concurrent - available permits
        # This is a best-effort approximation
        return self.max_concurrent - semaphore._value

    async def clear_domain(self, domain: str):
        """Clear all permits for a domain (emergency use)"""
        if domain in self.semaphores:
            del self.semaphores[domain]
            logger.info(f"Cleared rate limit for {domain}")


# Singleton
_rate_limiter: SimpleRateLimiter | None = None


def get_rate_limiter() -> SimpleRateLimiter:
    """Get the global rate limiter instance (singleton)"""
    global _rate_limiter
    if _rate_limiter is None:
        from ..settings import get_settings
        settings = get_settings()
        _rate_limiter = SimpleRateLimiter(
            max_concurrent=settings.rate_limit_max_concurrent,
            acquire_timeout=settings.rate_limit_acquire_timeout,
        )
    return _rate_limiter


def reset_rate_limiter():
    """Reset rate limiter (useful for testing)"""
    global _rate_limiter
    _rate_limiter = None
