"""HTTP client utilities"""

import httpx

from .proxy import get_proxy_manager


def create_async_client(timeout: float = 30.0, target_url: str | None = None) -> httpx.AsyncClient:
    """Create a configured async HTTP client with optional proxy support.

    Args:
        timeout: Request timeout in seconds.
        target_url: The URL being requested (used for proxy exclusion checking).
    """
    manager = get_proxy_manager()
    proxy_url = manager.get_proxy_url(target_url)

    kwargs: dict = {
        "timeout": timeout,
        "follow_redirects": True,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    }

    if proxy_url:
        kwargs["proxy"] = proxy_url

    return httpx.AsyncClient(**kwargs)
