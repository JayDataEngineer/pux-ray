"""Image segmentation service.

Uses SAM 2 (Segment Anything Model 2) for image segmentation.
Supports auto, point, and box prompting modes.
Lazy-loads on first request.
"""

import asyncio
import io
import time
from typing import Optional

import httpx
import numpy as np
from loguru import logger
from PIL import Image

from ..settings import get_settings, get_device


class SegmentService:
    """SAM 2 image segmentation."""

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Segmentation model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Segmentation model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("segment"):
                raise RuntimeError("Segmentation is disabled")

            try:
                logger.info(f"Loading segmentation model: {settings.segment_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"Segmentation model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load segmentation model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import torch
        from sam2 import build_sam2

        settings = get_settings()
        device = get_device()

        # Build SAM 2 with small model
        self._model = build_sam2(
            settings.segment_model,
            device=device,
        )

    async def segment(
        self,
        image_url: str,
        mode: str = "auto",
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        box: list[float] | None = None,
    ) -> dict:
        """Segment an image using SAM 2."""
        await self._ensure_loaded()

        try:
            image = await self._load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._segment_sync, image, mode, points, point_labels, box,
                )
                return result

            except Exception as e:
                logger.error(f"Segmentation error: {e}")
                return {"success": False, "error": f"Segmentation error: {str(e)[:200]}"}

    def _segment_sync(
        self,
        image: Image.Image,
        mode: str,
        points: list[list[float]] | None,
        point_labels: list[int] | None,
        box: list[float] | None,
    ) -> dict:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = get_device()
        predictor = SAM2ImagePredictor(self._model)

        img_array = np.array(image.convert("RGB"))

        with torch.inference_mode():
            predictor.set_image(img_array)

            if mode == "box" and box:
                # Box prompt: [x1, y1, x2, y2]
                box_np = np.array(box, dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    box=box_np,
                    multimask_output=True,
                )

            elif mode == "points" and points:
                # Point prompt
                points_np = np.array(points, dtype=np.float32)
                labels_np = np.array(
                    point_labels or [1] * len(points),
                    dtype=np.int32,
                )
                masks, scores, _ = predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    multimask_output=True,
                )

            else:
                # Auto mode: generate grid of points
                h, w = img_array.shape[:2]
                grid_size = 16
                x_points = np.linspace(0, w - 1, grid_size)
                y_points = np.linspace(0, h - 1, grid_size)
                xx, yy = np.meshgrid(x_points, y_points)
                grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
                grid_labels = np.ones(len(grid_points), dtype=np.int32)

                masks, scores, _ = predictor.predict(
                    point_coords=grid_points,
                    point_labels=grid_labels,
                    multimask_output=True,
                )

        # Convert masks to results
        result_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            result_masks.append({
                "score": float(scores[i]),
                "area": int(mask.sum()),
                "percentage": round(float(mask.sum()) / (h * w) * 100, 2),
            })

        # Sort by score descending
        result_masks.sort(key=lambda x: x["score"], reverse=True)

        return {
            "success": True,
            "masks": result_masks,
            "count": len(result_masks),
            "mode": mode,
        }

    async def _load_image(self, image_url: str) -> Image.Image:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("Segmentation model unloaded")


_segment_service: SegmentService | None = None


def get_segment_service() -> SegmentService:
    global _segment_service
    if _segment_service is None:
        _segment_service = SegmentService()
    return _segment_service
