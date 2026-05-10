"""Color palette extraction service.

Extracts dominant colors from images using ColorThief. No model loading needed.
"""

import asyncio
import io
import tempfile
from typing import Optional

import httpx
from loguru import logger

from ..settings import get_settings


class ColorService:
    """Color palette extraction via ColorThief."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def extract_colors(
        self,
        image_url: str,
        color_count: int = 5,
        quality: int = 10,
    ) -> dict:
        """Extract dominant colors and palette from an image."""
        settings = get_settings()
        if not settings.is_enabled("color"):
            return {"success": False, "error": "Color extraction is disabled"}

        try:
            # Download image to temp file (ColorThief needs a file path or file-like)
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
                response.raise_for_status()

            image_data = response.content

            async with self._lock:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._extract_sync, image_data, color_count, quality,
                )
                return result

        except Exception as e:
            logger.error(f"Color extraction error: {e}")
            return {"success": False, "error": f"Color extraction error: {str(e)[:200]}"}

    def _extract_sync(
        self,
        image_data: bytes,
        color_count: int,
        quality: int,
    ) -> dict:
        from colorthief import ColorThief

        # ColorThief accepts file-like objects
        img_file = io.BytesIO(image_data)
        ct = ColorThief(img_file)

        # Get dominant color
        dominant = ct.get_color(quality=quality)

        # Get palette (includes dominant)
        palette = ct.get_palette(color_count=color_count, quality=quality)

        return {
            "success": True,
            "dominant_color": list(dominant),
            "palette": [list(c) for c in palette],
        }

    async def close(self) -> None:
        pass


_color_service: ColorService | None = None


def get_color_service() -> ColorService:
    global _color_service
    if _color_service is None:
        _color_service = ColorService()
    return _color_service
