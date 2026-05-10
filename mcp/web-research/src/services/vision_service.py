"""Vision service using Florence-2 for image analysis.

Lazy-loads the model on first request. Uses asyncio.Lock for
serialized CPU inference.
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
from ..utils.proxy import get_proxy_manager


# Valid Florence-2 task types
VALID_TASKS = frozenset({
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OCR>",
    "<OCR_WITH_REGION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
})


class VisionService:
    """Florence-2 vision model service with lazy loading."""

    _instance = None

    def __init__(self):
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        """Lazy-load model on first call. Thread-safe via lock."""
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Model failed to load: {self._load_error}")
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.vision_enabled:
                raise RuntimeError("Vision model is disabled via settings")

            try:
                logger.info(f"Loading vision model: {settings.vision_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(
                    f"Vision model loaded in {elapsed:.1f}s "
                    f"(device={settings.vision_device})"
                )
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True  # Mark as "attempted" to avoid retry loop
                logger.error(f"Failed to load vision model: {e}")
                raise

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        settings = get_settings()

        self._processor = AutoProcessor.from_pretrained(
            settings.vision_model,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.vision_model,
            trust_remote_code=True,
        ).to(settings.vision_device)
        self._model.eval()

    async def analyze(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        task: str = "<MORE_DETAILED_CAPTION>",
        text_input: str | None = None,
    ) -> dict:
        """Analyze an image using Florence-2.

        Args:
            image_url: URL to download image from.
            image_base64: Base64-encoded image data.
            task: Florence-2 task prompt (e.g., '<DETAILED_CAPTION>').
            text_input: Optional additional text for tasks like
                        CAPTION_TO_PHRASE_GROUNDING.

        Returns:
            Dict with success, task, results, and optional error.
        """
        await self._ensure_loaded()

        # Validate task
        if task not in VALID_TASKS:
            return {
                "success": False,
                "error": f"Invalid task '{task}'. Valid: {sorted(VALID_TASKS)}",
            }

        # Load image
        try:
            image = await self._load_image(image_url, image_base64)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {str(e)[:200]}",
            }

        # Run inference (serialized via lock)
        async with self._lock:
            try:
                settings = get_settings()

                # Construct prompt: most Florence-2 tasks take ONLY the task token.
                # Only CAPTION_TO_PHRASE_GROUNDING accepts additional text after the task token.
                prompt = task
                if text_input and task == "<CAPTION_TO_PHRASE_GROUNDING>":
                    prompt = f"{task} {text_input}"

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self._infer_sync,
                        image,
                        prompt,
                        task,
                    ),
                    timeout=settings.vision_inference_timeout,
                )

                return {
                    "success": True,
                    "task": task,
                    "results": result,
                }

            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Inference timed out after {settings.vision_inference_timeout}s",
                }
            except Exception as e:
                logger.error(f"Vision inference error: {e}")
                return {
                    "success": False,
                    "error": f"Inference error: {str(e)[:200]}",
                }

    def _infer_sync(self, image: Image.Image, prompt: str, task: str) -> dict:
        """Synchronous inference (runs in thread pool, under lock)."""
        import torch

        settings = get_settings()

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(settings.vision_device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=settings.vision_max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        result = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size,
        )

        return result

    async def _load_image(
        self,
        image_url: str | None,
        image_base64: str | None,
    ) -> Image.Image:
        """Load PIL Image from URL or base64."""
        if image_url:
            proxy_mgr = get_proxy_manager()
            proxy_url = proxy_mgr.get_proxy_url(image_url)
            kwargs = {
                "timeout": 30.0,
                "follow_redirects": True,
                "headers": {"User-Agent": "MCP-Vision/1.0"},
            }
            if proxy_url:
                kwargs["proxy"] = proxy_url

            async with httpx.AsyncClient(**kwargs) as client:
                response = await client.get(image_url)
                response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image.load()  # Force load to catch corrupt images
            return image

        elif image_base64:
            data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(data))
            image.load()
            return image

        else:
            raise ValueError("Either image_url or image_base64 must be provided")

    async def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            logger.info("Vision model unloaded")


# Singleton factory (matches existing pattern from content_cleaner.py)
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    """Get or create the global VisionService instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
