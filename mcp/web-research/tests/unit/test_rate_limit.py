"""Tests for rate limiting service"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.services.rate_limit_service import RateLimitService


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    return redis


@pytest.fixture
def rate_limiter(mock_redis):
    """Create rate limiter with mock Redis"""
    limiter = RateLimitService(
        redis_url="redis://localhost:6379/0",
        max_concurrent=3,
        ttl=300
    )
    limiter._redis = mock_redis
    return limiter


class TestRateLimitService:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_acquire_first_request(self, rate_limiter, mock_redis):
        """First request should acquire immediately"""
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        acquired = await rate_limiter.acquire("example.com")
        assert acquired is True
        mock_redis.incr.assert_called_once_with("ratelimit:domain:example.com")
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self, rate_limiter, mock_redis):
        """Request under limit should succeed"""
        mock_redis.incr.return_value = 2  # Second request, under limit of 3

        acquired = await rate_limiter.acquire("example.com")
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self, rate_limiter, mock_redis):
        """Request at limit should succeed"""
        mock_redis.incr.return_value = 3  # Exactly at limit

        acquired = await rate_limiter.acquire("example.com")
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_over_limit(self, rate_limiter, mock_redis):
        """Request over limit should decrement and wait"""
        # First call returns over limit
        mock_redis.incr.return_value = 4

        acquired = await rate_limiter.acquire("example.com", timeout=0.1)
        assert acquired is False
        # Should have decremented after seeing it was over limit
        mock_redis.decr.assert_called()

    @pytest.mark.asyncio
    async def test_release(self, rate_limiter, mock_redis):
        """Test releasing a permit"""
        mock_redis.decr.return_value = 2  # Still 2 active after release

        await rate_limiter.release("example.com")
        mock_redis.decr.assert_called_once_with("ratelimit:domain:example.com")

    @pytest.mark.asyncio
    async def test_release_clears_key_at_zero(self, rate_limiter, mock_redis):
        """Releasing to zero should delete the key"""
        mock_redis.decr.return_value = 0  # No active requests

        await rate_limiter.release("example.com")
        mock_redis.delete.assert_called_once_with("ratelimit:domain:example.com")

    @pytest.mark.asyncio
    async def test_get_active_count(self, rate_limiter, mock_redis):
        """Test getting active request count"""
        mock_redis.get.return_value = "3"

        count = await rate_limiter.get_active_count("example.com")
        assert count == 3
        mock_redis.get.assert_called_once_with("ratelimit:domain:example.com")

    @pytest.mark.asyncio
    async def test_get_active_count_no_active(self, rate_limiter, mock_redis):
        """Test getting count when no active requests"""
        mock_redis.get.return_value = None

        count = await rate_limiter.get_active_count("example.com")
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_domain(self, rate_limiter, mock_redis):
        """Test clearing all permits for a domain"""
        await rate_limiter.clear_domain("example.com")
        mock_redis.delete.assert_called_once_with("ratelimit:domain:example.com")

    @pytest.mark.asyncio
    async def test_concurrent_different_domains(self, rate_limiter, mock_redis):
        """Concurrent requests to different domains should be independent"""
        mock_redis.incr.return_value = 1

        # Different domains should have different keys
        await rate_limiter.acquire("example.com")
        await rate_limiter.acquire("another.com")

        assert mock_redis.incr.call_count == 2
        calls = mock_redis.incr.call_args_list
        keys = [call[0][0] for call in calls]
        assert "ratelimit:domain:example.com" in keys
        assert "ratelimit:domain:another.com" in keys
