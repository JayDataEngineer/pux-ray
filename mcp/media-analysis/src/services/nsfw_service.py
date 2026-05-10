"""NSFW content detection service.

Uses NudeNet ONNX classifier for detecting NSFW content in images.
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


class NsfwService:
    """NSFW content detection via NudeNet ONNX."""

    def __init__(self):
        self._session = None
        self._labels: list[str] = []
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"NSFW model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"NSFW model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("nsfw"):
                raise RuntimeError("NSFW detection is disabled")

            try:
                logger.info("Loading NSFW detection model (NudeNet)")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"NSFW model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load NSFW model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        device = get_device()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        # NudeNet classifier model from HuggingFace
        model_path = hf_hub_download(
            "notpro/nudenet",
            filename="nudenet_classifier.onnx",
        )

        self._session = ort.InferenceSession(model_path, providers=providers)

        # NudeNet classifier labels
        self._labels = [
            "safe",
            "questionable",
            "unsafe",
        ]

    async def classify_nsfw(
        self,
        image_url: str,
        threshold: float = 0.5,
    ) -> dict:
        """Classify an image for NSFW content."""
        await self._ensure_loaded()

        try:
            image = await self._load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._classify_sync, image, threshold,
                )
                return result

            except Exception as e:
                logger.error(f"NSFW classification error: {e}")
                return {"success": False, "error": f"Classification error: {str(e)[:200]}"}

    def _classify_sync(self, image: Image.Image, threshold: float) -> dict:
        # NudeNet expects 224x224 input
        image = image.convert("RGB").resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # [1, 224, 224, 3]

        # Some NudeNet models expect NCHW
        input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape

        # Check if model expects NCHW [1, 3, H, W] or NHWC [1, H, W, 3]
        if len(input_shape) == 4 and input_shape[1] == 3:
            img_array = np.transpose(img_array, (0, 3, 1, 2))

        outputs = self._session.run(None, {input_name: img_array})
        probs = outputs[0][0]

        # Build label scores
        if len(probs) == len(self._labels):
            scores = {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}
        else:
            scores = {f"class_{i}": float(p) for i, p in enumerate(probs)}

        # Determine NSFW status
        unsafe_score = scores.get("unsafe", scores.get("class_2", 0.0))
        is_nsfw = unsafe_score >= threshold

        return {
            "success": True,
            "is_nsfw": is_nsfw,
            "nsfw_score": unsafe_score,
            "scores": scores,
        }

    async def _load_image(self, image_url: str) -> Image.Image:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    async def close(self) -> None:
        if self._session is not None:
            del self._session
            self._session = None
            self._loaded = False
            logger.info("NSFW model unloaded")


_nsfw_service: NsfwService | None = None


def get_nsfw_service() -> NsfwService:
    global _nsfw_service
    if _nsfw_service is None:
        _nsfw_service = NsfwService()
    return _nsfw_service
