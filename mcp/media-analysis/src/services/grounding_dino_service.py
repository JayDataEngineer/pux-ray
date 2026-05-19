"""Grounding DINO service — open-vocabulary object detection.

Detect any object described in natural language text.
Uses IDEA-Research/grounding-dino-tiny (~180M params, ~1.3GB VRAM).
"""

import asyncio
import time
from typing import Optional

import httpx
import io
import base64
from loguru import logger
from PIL import Image

from ..settings import get_settings, get_device


class GroundingDinoService:

    _instance = None

    def __init__(self):
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("grounding_dino"):
                raise RuntimeError("Grounding DINO is disabled")

            try:
                logger.info(f"Loading Grounding DINO: {settings.grounding_dino_model}")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)
                elapsed = time.time() - start
                logger.info(f"Grounding DINO loaded in {elapsed:.1f}s (device={get_device()})")
                self._loaded = True

                from .idle_watcher import get_idle_watcher
                get_idle_watcher().watch("grounding_dino", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load Grounding DINO: {e}")
                raise

    def _load_model_sync(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        settings = get_settings()
        device = get_device()
        self._processor = AutoProcessor.from_pretrained(settings.grounding_dino_model)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            settings.grounding_dino_model,
        ).to(device)
        self._model.eval()

    async def detect(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        text_prompt: str = "",
        threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict:
        await self._ensure_loaded()

        from .idle_watcher import get_idle_watcher
        get_idle_watcher().touch("grounding_dino")

        if not text_prompt:
            return {"success": False, "error": "text_prompt is required (e.g. 'a cat . a dog .')"}

        try:
            image = await self._load_image(image_url, image_base64)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._infer_sync, image, text_prompt, threshold, text_threshold
                    ),
                    timeout=60.0,
                )
                return {"success": True, "text_prompt": text_prompt, "detections": result}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Inference timed out after 60s"}
            except Exception as e:
                logger.error(f"Grounding DINO inference error: {e}")
                return {"success": False, "error": f"Inference error: {str(e)[:200]}"}

    def _infer_sync(
        self, image: Image.Image, text_prompt: str, threshold: float, text_threshold: float
    ) -> list[dict]:
        import torch

        device = get_device()
        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        detections = []
        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            detections.append({
                "label": label,
                "confidence": round(score.item(), 3),
                "box": [round(x, 2) for x in box.tolist()],
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
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            logger.info("Grounding DINO unloaded")


_grounding_dino_service: GroundingDinoService | None = None


def get_grounding_dino_service() -> GroundingDinoService:
    global _grounding_dino_service
    if _grounding_dino_service is None:
        _grounding_dino_service = GroundingDinoService()
    return _grounding_dino_service
