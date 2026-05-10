"""Base Celery task class with shared async/sync bridge patterns

Uses pattern matching (pro+pattern) and TRY for error handling.
"""

import asyncio
from typing import Any, Callable
from functools import wraps
from loguru import logger

from celery import Task
from ..db.database import Database
from ..services.content_cleaner import get_content_cleaner


def run_sync(coro: Callable[[], Any]) -> Any:
    """Run async coroutine in sync context with cached event loop"""
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro())


def try_async(coro: Callable[[], Any], default: Any = None, reraise: bool = False) -> Any:
    """TRY pattern for async: execute coroutine, return default on error

    Args:
        coro: Coroutine to execute
        default: Value to return on error
        reraise: If True, re-raise the exception

    Returns:
        Coroutine result or default on error
    """
    try:
        return run_sync(coro)
    except Exception as e:
        logger.warning(f"Async operation failed: {e}")
        if reraise:
            raise
        return default


class AsyncMixin:
    """Mixin for running async code in sync Celery context"""

    _loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for this thread"""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def run_async(self, coro: Callable[[], Any]) -> Any:
        """Run async coroutine in this task's event loop"""
        return self.loop.run_until_complete(coro())


class DatabaseMixin:
    """Mixin providing lazy database initialization"""

    _db: Database | None = None
    _db_initialized: bool = False

    @property
    def db(self) -> Database:
        """Lazy init database connection"""
        if self._db is None:
            self._db = Database()
            # Initialize async connection
            if not self._db_initialized:
                self.run_async(self._db.init())
                self._db_initialized = True
        return self._db

    def run_async(self, coro: Callable[[], Any]) -> Any:
        """Override in subclass or mix with AsyncMixin"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro())
        finally:
            loop.close()


class CleanerMixin:
    """Mixin providing lazy content cleaner initialization"""

    _cleaner: Any | None = None

    @property
    def cleaner(self) -> Any:
        """Lazy init content cleaner"""
        if self._cleaner is None:
            self._cleaner = get_content_cleaner()
        return self._cleaner


class CacheMixin:
    """Mixin providing lazy cache service initialization"""

    _cache: Any | None = None

    @property
    def cache(self) -> Any | None:
        """Lazy init cache service (returns None if unavailable)"""
        if self._cache is None:
            try:
                from ..services.cache_service import get_cache_service
                self._cache = self.run_async(get_cache_service())
            except Exception as e:
                logger.warning(f"Cache service unavailable: {e}")
                self._cache = None
        return self._cache

    def run_async(self, coro: Callable[[], Any]) -> Any:
        """Override in subclass or mix with AsyncMixin"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro())
        finally:
            loop.close()


class BaseTask(Task, AsyncMixin, DatabaseMixin, CleanerMixin, CacheMixin):
    """Base Celery task with all common mixins combined

    Combines async execution, database, cleaner, and cache access.
    Subclasses can use self.db, self.cleaner, self.cache, self.run_async().
    """

    def after_return(self, *args, **kwargs):
        """Cleanup after task completes - keep resources alive for reuse"""
        pass
