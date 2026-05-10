"""Singleton pattern factories for services"""

import asyncio
from typing import Callable


def create_singleton_factory(cls, name: str, init_method: str = None):
    """Create a singleton factory function"""
    instance = None

    def factory(**kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(**kwargs)
        return instance

    factory.__name__ = name
    return factory


def create_async_singleton_factory(cls, name: str, init_method: str = None):
    """Create an async singleton factory function"""
    instance = None
    initialized = False

    async def factory(**kwargs):
        nonlocal instance, initialized
        if instance is None:
            instance = cls(**kwargs)
        if not initialized and init_method:
            await getattr(instance, init_method)()
            initialized = True
        return instance

    factory.__name__ = name
    return factory
