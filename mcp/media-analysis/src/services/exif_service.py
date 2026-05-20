"""EXIF metadata extraction service.

Extracts EXIF data from images using Pillow. No model loading needed.
"""

import asyncio
from typing import Optional

from loguru import logger
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from ..settings import get_settings
from .media_utils import load_image


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
                image = await load_image(image_url)
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

    async def close(self) -> None:
        pass


_exif_service: ExifService | None = None


def get_exif_service() -> ExifService:
    global _exif_service
    if _exif_service is None:
        _exif_service = ExifService()
    return _exif_service
