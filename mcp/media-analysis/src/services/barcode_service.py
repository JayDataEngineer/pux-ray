"""QR code and barcode detection service.

Uses pyzbar + OpenCV for detection. No model loading needed.
"""

import asyncio
import io
from typing import Optional

import httpx
from loguru import logger
from PIL import Image

from ..settings import get_settings


class BarcodeService:
    """QR code and barcode detection via pyzbar + OpenCV."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def read_barcodes(self, image_url: str) -> dict:
        """Detect and read QR codes and barcodes from an image."""
        settings = get_settings()
        if not settings.is_enabled("barcode"):
            return {"success": False, "error": "Barcode detection is disabled"}

        try:
            image = await self._load_image(image_url)

            async with self._lock:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._detect_sync, image,
                )
                return result

        except Exception as e:
            logger.error(f"Barcode detection error: {e}")
            return {"success": False, "error": f"Barcode detection error: {str(e)[:200]}"}

    def _detect_sync(self, image: Image.Image) -> dict:
        import numpy as np
        from pyzbar.pyzbar import decode

        # Convert PIL to numpy array for pyzbar
        img_array = np.array(image.convert("RGB"))
        decoded = decode(img_array)

        barcodes = []
        for obj in decoded:
            barcode = {
                "type": obj.type,
                "data": obj.data.decode("utf-8", errors="replace"),
            }
            # Add bounding rect if available
            if obj.rect:
                barcode["rect"] = {
                    "left": obj.rect.left,
                    "top": obj.rect.top,
                    "width": obj.rect.width,
                    "height": obj.rect.height,
                }
            barcodes.append(barcode)

        return {
            "success": True,
            "barcodes": barcodes,
            "count": len(barcodes),
        }

    async def _load_image(self, image_url: str) -> Image.Image:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    async def close(self) -> None:
        pass


_barcode_service: BarcodeService | None = None


def get_barcode_service() -> BarcodeService:
    global _barcode_service
    if _barcode_service is None:
        _barcode_service = BarcodeService()
    return _barcode_service
