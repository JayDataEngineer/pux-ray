"""Kosmos-2.5 service — document OCR and markdown generation.

Uses microsoft/kosmos-2.5 (1.3B params, ~3-4GB VRAM).
Converts document images to structured markdown or OCR with bounding boxes.
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


class KosmosService:

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
            if not settings.is_enabled("kosmos"):
                raise RuntimeError("Kosmos-2.5 is disabled")

            try:
                logger.info(f"Loading Kosmos-2.5: {settings.kosmos_model}")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)
                elapsed = time.time() - start
                logger.info(f"Kosmos-2.5 loaded in {elapsed:.1f}s (device={get_device()})")
                self._loaded = True

                from .idle_watcher import get_idle_watcher
                get_idle_watcher().watch("kosmos", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load Kosmos-2.5: {e}")
                raise

    def _load_model_sync(self) -> None:
        import torch
        from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

        settings = get_settings()
        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(settings.kosmos_model)
        self._model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            settings.kosmos_model,
            device_map=device if device == "cuda" else None,
            dtype=dtype,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    async def convert(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        mode: str = "markdown",
        max_new_tokens: int = 4096,
    ) -> dict:
        await self._ensure_loaded()

        from .idle_watcher import get_idle_watcher
        get_idle_watcher().touch("kosmos")

        if mode not in ("markdown", "ocr"):
            return {"success": False, "error": "mode must be 'markdown' or 'ocr'"}

        try:
            image = await self._load_image(image_url, image_base64)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self._infer_sync, image, mode, max_new_tokens),
                    timeout=120.0,
                )
                return {"success": True, "mode": mode, "result": result}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Inference timed out after 120s"}
            except Exception as e:
                logger.error(f"Kosmos-2.5 inference error: {e}")
                return {"success": False, "error": f"Inference error: {str(e)[:200]}"}

    def _infer_sync(self, image: Image.Image, mode: str, max_new_tokens: int) -> str:
        import torch

        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        prompt = "<md>" if mode == "markdown" else "<ocr>"

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        height, width = inputs.pop("height"), inputs.pop("width")
        inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
            inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text[0]

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
            logger.info("Kosmos-2.5 unloaded")


_kosmos_service: KosmosService | None = None


def get_kosmos_service() -> KosmosService:
    global _kosmos_service
    if _kosmos_service is None:
        _kosmos_service = KosmosService()
    return _kosmos_service
