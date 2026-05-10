"""Phi-4-multimodal service — visual reasoning via multimodal LLM.

Uses microsoft/Phi-4-multimodal-instruct (5.6B params).
Chat about images: describe, reason, answer questions about visual content.
Falls back to eager attention on CPU (no flash-attn required).
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


class Phi4VisionService:

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
            if not settings.is_enabled("phi4_vision"):
                raise RuntimeError("Phi-4 vision is disabled")

            try:
                logger.info(f"Loading Phi-4 multimodal: {settings.phi4_vision_model}")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)
                elapsed = time.time() - start
                logger.info(f"Phi-4 multimodal loaded in {elapsed:.1f}s (device={get_device()})")
                self._loaded = True

                from .idle_watcher import get_idle_watcher
                get_idle_watcher().watch("phi4_vision", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load Phi-4 multimodal: {e}")
                raise

    def _load_model_sync(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        settings = get_settings()
        device = get_device()
        attn = "flash_attention_2" if device == "cuda" else "eager"

        self._processor = AutoProcessor.from_pretrained(
            settings.phi4_vision_model,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.phi4_vision_model,
            device_map=device if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            _attn_implementation=attn,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        self._generation_config = GenerationConfig.from_pretrained(settings.phi4_vision_model)

    async def chat(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 2048,
    ) -> dict:
        await self._ensure_loaded()

        from .idle_watcher import get_idle_watcher
        get_idle_watcher().touch("phi4_vision")

        try:
            image = await self._load_image(image_url, image_base64)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._infer_sync, image, prompt, max_new_tokens
                    ),
                    timeout=120.0,
                )
                return {"success": True, "response": result}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Inference timed out after 120s"}
            except Exception as e:
                logger.error(f"Phi-4 vision inference error: {e}")
                return {"success": False, "error": f"Inference error: {str(e)[:200]}"}

    def _infer_sync(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|)"
        prompt_suffix = "<|end|>"

        full_prompt = f"{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}"

        inputs = self._processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self._model.device)

        generate_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self._generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

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
            logger.info("Phi-4 multimodal unloaded")


_phi4_vision_service: Phi4VisionService | None = None


def get_phi4_vision_service() -> Phi4VisionService:
    global _phi4_vision_service
    if _phi4_vision_service is None:
        _phi4_vision_service = Phi4VisionService()
    return _phi4_vision_service
