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

SAM2_MODELS = {
    "sam2_hiera_s": "facebook/sam2-hiera-small",
    "sam2_hiera_t": "facebook/sam2-hiera-tiny",
    "sam2_hiera_b+": "facebook/sam2-hiera-base-plus",
    "sam2_hiera_l": "facebook/sam2-hiera-large",
}


class SegmentService:
    """SAM 2 image segmentation."""

    def __init__(self):
        self._predictor = None
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

                from .idle_watcher import get_idle_watcher
                get_idle_watcher().watch("segment", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load segmentation model: {e}")
                raise

    def _load_model_sync(self) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        settings = get_settings()
        device = get_device()
        model_name = settings.segment_model
        repo_id = SAM2_MODELS.get(model_name, "facebook/sam2-hiera-small")

        self._predictor = SAM2ImagePredictor.from_pretrained(repo_id, device=device)

    async def segment(
        self,
        image_url: str,
        mode: str = "auto",
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        box: list[float] | None = None,
    ) -> dict:
        await self._ensure_loaded()

        try:
            image = await self._load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                from .idle_watcher import get_idle_watcher
                get_idle_watcher().touch("segment")

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

        img_array = np.array(image.convert("RGB"))

        with torch.inference_mode():
            self._predictor.set_image(img_array)

            if mode == "box" and box:
                box_np = np.array(box, dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box_np,
                    multimask_output=True,
                )

            elif mode == "points" and points:
                points_np = np.array(points, dtype=np.float32)
                labels_np = np.array(
                    point_labels or [1] * len(points),
                    dtype=np.int32,
                )
                masks, scores, _ = self._predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    multimask_output=True,
                )

            else:
                h, w = img_array.shape[:2]
                grid_size = 16
                x_points = np.linspace(0, w - 1, grid_size)
                y_points = np.linspace(0, h - 1, grid_size)
                xx, yy = np.meshgrid(x_points, y_points)
                grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
                grid_labels = np.ones(len(grid_points), dtype=np.int32)

                masks, scores, _ = self._predictor.predict(
                    point_coords=grid_points,
                    point_labels=grid_labels,
                    multimask_output=True,
                )

        result_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            result_masks.append({
                "score": float(scores[i]),
                "area": int(mask.sum()),
                "percentage": round(float(mask.sum()) / (h * w) * 100, 2),
            })

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
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            self._loaded = False
            logger.info("Segmentation model unloaded")


_segment_service: SegmentService | None = None


def get_segment_service() -> SegmentService:
    global _segment_service
    if _segment_service is None:
        _segment_service = SegmentService()
    return _segment_service
