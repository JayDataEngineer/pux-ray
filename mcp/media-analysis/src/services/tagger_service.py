"""WD14 image tagger service.

Uses onnxruntime for CPU inference. Tags images with content labels.
Lazy-loads on first request.
"""

import asyncio
import time
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from ..settings import get_settings, get_device
from .media_utils import load_image


class TaggerService:
    """WD14 image tagger via onnxruntime."""

    def __init__(self):
        self._session = None
        self._labels: list[str] = []
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Tagger model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Tagger model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("tagger"):
                raise RuntimeError("Tagger model is disabled")

            try:
                logger.info(f"Loading tagger model: {settings.tagger_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"Tagger model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load tagger model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import json
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        settings = get_settings()
        model_path = hf_hub_download(
            settings.tagger_model, filename="model.onnx",
        )
        labels_path = hf_hub_download(
            settings.tagger_model, filename="selected_tags.csv",
        )

        self._session = ort.InferenceSession(
            model_path,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if get_device() == "cuda"
                else ["CPUExecutionProvider"]
            ),
        )

        # Parse labels from CSV
        self._labels = []
        with open(labels_path) as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    self._labels.append(parts[1])

    async def tag(
        self,
        image_url: str,
        threshold: float | None = None,
    ) -> dict:
        """Tag an image with content labels."""
        await self._ensure_loaded()

        if threshold is None:
            threshold = get_settings().tagger_threshold

        try:
            image = await load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                tags = await loop.run_in_executor(
                    None, self._tag_sync, image, threshold,
                )
                return {"success": True, "tags": tags}

            except Exception as e:
                logger.error(f"Tagger error: {e}")
                return {"success": False, "error": f"Tagger error: {str(e)[:200]}"}

    def _tag_sync(self, image: Image.Image, threshold: float) -> list[dict]:
        # Preprocess: resize to 448x448, normalize
        image = image.convert("RGB").resize((448, 448))
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Model expects NHWC format: [1, 448, 448, 3]
        img_array = np.expand_dims(img_array, axis=0)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: img_array})

        # Apply sigmoid to get probabilities
        probs = 1.0 / (1.0 + np.exp(-outputs[0][0]))

        tags = []
        for i, prob in enumerate(probs):
            if i < len(self._labels) and prob >= threshold:
                tags.append({"tag": self._labels[i], "confidence": float(prob)})

        tags.sort(key=lambda x: x["confidence"], reverse=True)
        return tags

    async def close(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
            self._loaded = False
            logger.info("Tagger model unloaded")


_tagger_service: TaggerService | None = None


def get_tagger_service() -> TaggerService:
    global _tagger_service
    if _tagger_service is None:
        _tagger_service = TaggerService()
    return _tagger_service
