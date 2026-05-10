"""Utility modules for reducing code duplication"""

from .url_utils import extract_domain
from .singleton import create_singleton_factory, create_async_singleton_factory
from .rate_limiter import get_rate_limiter

__all__ = [
    "extract_domain",
    "create_singleton_factory",
    "create_async_singleton_factory",
    "get_rate_limiter",
]

# Note: RedisMixin is kept for legacy services but no longer used by core MCP tools
