"""Redis caching service for scrape and search results"""

import json
from typing import Optional
from loguru import logger

from ..utils.redis_mixin import RedisMixin


class CacheService(RedisMixin):
    """Redis-backed caching for search and scrape results"""

    async def _get_redis(self):
        """Get Redis connection (lazy initialization) with logging"""
        redis = await super()._get_redis()
        if self._redis is None or redis == self._redis:
            logger.info(f"Cache connected to Redis: {self.redis_url}")
        return redis

    async def get_scrape(self, url: str) -> Optional[dict]:
        """Get cached scrape result"""
        redis = await self._get_redis()
        key = f"scrape:{url}"
        try:
            data = await redis.get(key)
            if data:
                logger.debug(f"Cache HIT for scrape: {url}")
                return json.loads(data)
            logger.debug(f"Cache MISS for scrape: {url}")
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set_scrape(self, url: str, data: dict, ttl: int = 86400) -> bool:
        """Cache scrape result"""
        redis = await self._get_redis()
        key = f"scrape:{url}"
        try:
            await redis.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def get_search(self, query: str, pages: int, exclude_blacklist: bool) -> Optional[dict]:
        """Get cached search result"""
        redis = await self._get_redis()
        key = f"search:{query}:{pages}:{exclude_blacklist}"
        try:
            data = await redis.get(key)
            if data:
                logger.info(f"Cache HIT for search: {query}")
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set_search(self, query: str, pages: int, exclude_blacklist: bool, data: dict, ttl: int = 3600) -> bool:
        """Cache search result"""
        redis = await self._get_redis()
        key = f"search:{query}:{pages}:{exclude_blacklist}"
        try:
            await redis.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False


# Singleton factory
from ..utils import create_async_singleton_factory
get_cache_service = create_async_singleton_factory(CacheService, "get_cache_service")
