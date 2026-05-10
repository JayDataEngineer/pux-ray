"""EXIF metadata extraction service.

Extracts EXIF data from images using Pillow. No model loading needed.
"""

import asyncio
import io
from typing import Optional

import httpx
from loguru import logger
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from ..settings import get_settings


def _decode_exif(raw_exif: dict) -> dict:
    """Decode EXIF tag IDs into human-readable names."""
    decoded = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")

        # Handle nested GPS IFD
        if tag_name == "GPSInfo":
            gps = {}
            for gps_tag_id, gps_value in value.items():
                gps_name = GPSTAGS.get(gps_tag_id, f"GPS_{gps_tag_id}")
                gps[gps_name] = str(gps_value)
            decoded["GPSInfo"] = gps
        else:
            decoded[tag_name] = str(value)

    return decoded


class ExifService:
    """EXIF metadata extraction via Pillow."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def extract_exif(self, image_url: str) -> dict:
        """Extract EXIF metadata from an image."""
        settings = get_settings()
        if not settings.is_enabled("exif"):
            return {"success": False, "error": "EXIF extraction is disabled"}

        try:
            async with self._lock:
                image = await self._load_image(image_url)
                raw_exif = image._getexif()

                if raw_exif is None:
                    return {
                        "success": True,
                        "exif": {},
                        "has_exif": False,
                    }

                decoded = _decode_exif(raw_exif)
                return {
                    "success": True,
                    "exif": decoded,
                    "has_exif": True,
                    "field_count": len(decoded),
                }

        except Exception as e:
            logger.error(f"EXIF extraction error: {e}")
            return {"success": False, "error": f"EXIF extraction error: {str(e)[:200]}"}

    async def _load_image(self, image_url: str) -> Image.Image:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    async def close(self) -> None:
        pass


_exif_service: ExifService | None = None


def get_exif_service() -> ExifService:
    global _exif_service
    if _exif_service is None:
        _exif_service = ExifService()
    return _exif_service
