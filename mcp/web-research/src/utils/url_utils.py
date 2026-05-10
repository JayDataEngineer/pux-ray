"""URL utilities"""

from urllib.parse import urlparse
from typing import Optional


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path
    except Exception:
        return url


__all__ = ["extract_domain"]
