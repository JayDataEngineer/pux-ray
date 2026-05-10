"""YOLOv8-nano object detection service.

Uses ultralytics with ONNX backend for fast CPU inference.
Lazy-loads on first request.
"""

import asyncio
import base64
import io
import time
from typing import Optional

import httpx
from loguru import logger
from PIL import Image

from ..settings import get_settings


class YoloService:
    """YOLOv8-nano object detection."""

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"YOLO model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"YOLO model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("yolo"):
                raise RuntimeError("YOLO model is disabled")

            try:
                logger.info(f"Loading YOLO model: {settings.yolo_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"YOLO model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load YOLO model: {e}")
                raise

    def _load_model_sync(self) -> None:
        from ultralytics import YOLO

        settings = get_settings()
        self._model = YOLO(settings.yolo_model)

    async def detect(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        confidence: float = 0.25,
    ) -> dict:
        """Detect objects in an image."""
        await self._ensure_loaded()

        try:
            image = await self._load_image(image_url, image_base64)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._detect_sync, image, confidence,
                )
                return {"success": True, "detections": result}

            except Exception as e:
                logger.error(f"YOLO detection error: {e}")
                return {"success": False, "error": f"Detection error: {str(e)[:200]}"}

    def _detect_sync(self, image: Image.Image, confidence: float) -> list[dict]:
        import numpy as np

        img_array = np.array(image)
        results = self._model(img_array, conf=confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detections.append({
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    },
                })
        return detections

    async def _load_image(self, image_url: str | None, image_base64: str | None) -> Image.Image:
        if image_url:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
                response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.load()
            return image
        elif image_base64:
            data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(data))
            image.load()
            return image
        else:
            raise ValueError("Either image_url or image_base64 must be provided")

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("YOLO model unloaded")


_yolo_service: YoloService | None = None


def get_yolo_service() -> YoloService:
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YoloService()
    return _yolo_service
