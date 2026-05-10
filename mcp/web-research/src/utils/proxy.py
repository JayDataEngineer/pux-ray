"""Proxy manager with rotation support.

Provides a centralized proxy configuration layer for all HTTP clients.
Supports single proxy, multiple proxies with rotation (round-robin/random),
and hostname exclusions for internal services.

Configuration via environment variables (loaded through Settings):
- MCP_PROXY_URL: Single proxy URL (e.g. http://gluetun:8888 or socks5://host:1080)
- MCP_PROXY_URLS: Comma-separated list for rotation (takes precedence)
- MCP_PROXY_ROTATION: "round-robin" (default) or "random"
- MCP_PROXY_EXCLUDE: Comma-separated hostnames to bypass proxy
"""

import random
import threading
from urllib.parse import urlparse

import httpx
from loguru import logger


class ProxyManager:
    """Thread-safe proxy manager with rotation and exclusion support."""

    def __init__(
        self,
        proxy_url: str | None = None,
        proxy_urls: str | None = None,
        rotation: str = "round-robin",
        exclude: str = "searxng,postgres,redis,localhost,127.0.0.1",
    ):
        # Build proxy list: MCP_PROXY_URLS takes precedence over MCP_PROXY_URL
        if proxy_urls:
            self._proxies = [u.strip() for u in proxy_urls.split(",") if u.strip()]
        elif proxy_url:
            self._proxies = [proxy_url.strip()]
        else:
            self._proxies = []

        self._rotation = rotation if rotation in ("round-robin", "random") else "round-robin"
        self._exclude = {h.strip().lower() for h in exclude.split(",") if h.strip()}
        self._index = 0
        self._lock = threading.Lock()

        if self._proxies:
            logger.info(f"Proxy enabled: {len(self._proxies)} proxy(ies), rotation={self._rotation}")
            logger.info(f"Proxy exclusions: {self._exclude}")
        else:
            logger.debug("Proxy disabled (no MCP_PROXY_URL or MCP_PROXY_URLS configured)")

    @property
    def enabled(self) -> bool:
        return len(self._proxies) > 0

    @property
    def proxy_count(self) -> int:
        return len(self._proxies)

    def get_proxy_url(self, target_url: str | None = None) -> str | None:
        """Return the next proxy URL, or None if disabled or target is excluded.

        Args:
            target_url: The URL being requested. If its hostname is in the
                        exclusion list, returns None (direct connection).
        """
        if not self._proxies:
            return None

        # Check exclusions
        if target_url:
            try:
                hostname = urlparse(target_url).hostname or ""
                if hostname.lower() in self._exclude:
                    return None
            except Exception:
                pass

        return self._rotate()

    def get_proxy_dict(self, target_url: str | None = None) -> dict | None:
        """Return httpx-compatible proxy dict {"http://": ..., "https://": ...} or None."""
        proxy = self.get_proxy_url(target_url)
        if proxy is None:
            return None
        return {"http://": proxy, "https://": proxy}

    def get_current(self) -> str | None:
        """Return the current proxy URL without advancing rotation."""
        if not self._proxies:
            return None
        with self._lock:
            return self._proxies[self._index]

    def rotate(self) -> str | None:
        """Manually advance to the next proxy and return it."""
        if not self._proxies:
            return None
        with self._lock:
            self._index = (self._index + 1) % len(self._proxies)
            current = self._proxies[self._index]
        logger.info(f"Manually rotated to proxy: {current}")
        return current

    def get_stats(self) -> dict:
        """Return current proxy configuration stats."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "proxy_count": len(self._proxies),
                "proxies": self._proxies,
                "rotation": self._rotation,
                "current_index": self._index,
                "current_proxy": self._proxies[self._index] if self._proxies else None,
                "exclusions": sorted(self._exclude),
            }

    def _rotate(self) -> str:
        """Return next proxy URL based on rotation strategy."""
        with self._lock:
            if self._rotation == "random":
                return random.choice(self._proxies)
            # round-robin
            proxy = self._proxies[self._index]
            self._index = (self._index + 1) % len(self._proxies)
            return proxy


def create_proxied_client(
    timeout: float = 30.0,
    follow_redirects: bool = True,
    target_url: str | None = None,
    **kwargs,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with proxy settings applied.

    If proxy is disabled or target is excluded, creates a regular client.

    Args:
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow redirects.
        target_url: The URL being requested (for exclusion checking).
        **kwargs: Additional arguments passed to httpx.AsyncClient.
    """
    manager = get_proxy_manager()
    proxy_url = manager.get_proxy_url(target_url)

    client_kwargs: dict = {
        "timeout": timeout,
        "follow_redirects": follow_redirects,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        **kwargs,
    }

    if proxy_url:
        client_kwargs["proxy"] = proxy_url
        logger.debug(f"Creating proxied client for {target_url or 'unknown'} -> {proxy_url}")

    return httpx.AsyncClient(**client_kwargs)


# ========== Singleton ==========

_proxy_manager: ProxyManager | None = None


def get_proxy_manager() -> ProxyManager:
    """Get or create the global ProxyManager instance."""
    global _proxy_manager
    if _proxy_manager is None:
        from ..settings import get_settings
        settings = get_settings()
        _proxy_manager = ProxyManager(
            proxy_url=settings.proxy_url,
            proxy_urls=settings.proxy_urls,
            rotation=settings.proxy_rotation,
            exclude=settings.proxy_exclude,
        )
    return _proxy_manager


def reset_proxy_manager():
    """Reset the proxy manager (useful for testing)."""
    global _proxy_manager
    _proxy_manager = None
