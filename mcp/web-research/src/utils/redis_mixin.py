"""Redis connection mixin for shared lazy initialization pattern"""

import os
from typing import Optional


class RedisMixin:
    """Shared Redis connection logic for cache and rate limiting services"""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize with Redis URL (connection is lazy)

        Args:
            redis_url: Redis URL (defaults to CELERY_BROKER_URL env var)
        """
        if redis_url is None:
            redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        """Get or create Redis connection (lazy initialization)"""
        import redis.asyncio as aioredis

        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
