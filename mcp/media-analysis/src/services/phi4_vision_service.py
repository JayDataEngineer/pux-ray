"""Gemma 4 E4B vision service — visual reasoning via llama-cpp-python GGUF.

Uses unsloth/gemma-4-E4B-it-GGUF (IQ4_NL, 4.84GB) with mmproj vision encoder.
Runs on CPU/DRAM only (n_gpu_layers=0). ~10-20 tok/s.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from ..settings import get_settings


class Phi4VisionService:

    _instance = None

    def __init__(self):
        self._llm = None
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
                logger.info(f"Loading Gemma 4 vision: {settings.phi4_vision_model}")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)
                elapsed = time.time() - start
                logger.info(f"Gemma 4 vision loaded in {elapsed:.1f}s")
                self._loaded = True

                from .idle_watcher import get_idle_watcher
                get_idle_watcher().watch("phi4_vision", self)

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load Gemma 4 vision: {e}")
                raise

    def _load_model_sync(self) -> None:
        from llama_cpp import Llama

        settings = get_settings()
        n_threads = settings.phi4_vision_n_threads

        self._llm = Llama.from_pretrained(
            repo_id=settings.phi4_vision_model,
            filename=settings.phi4_vision_filename,
            mmproj=settings.phi4_vision_mmproj,
            n_gpu_layers=0,
            n_ctx=4096,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            verbose=False,
        )

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

        if not image_url and not image_base64:
            return {"success": False, "error": "Either image_url or image_base64 must be provided"}

        image_input = image_url
        if not image_input and image_base64:
            image_input = f"data:image/png;base64,{image_base64}"

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._infer_sync, image_input, prompt, max_new_tokens
                    ),
                    timeout=300.0,
                )
                return {"success": True, "response": result}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Inference timed out after 300s"}
            except Exception as e:
                logger.error(f"Gemma 4 vision inference error: {e}")
                return {"success": False, "error": f"Inference error: {str(e)[:200]}"}

    def _infer_sync(self, image_input: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_input},
                    },
                ],
            }
        ]

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
        )

        return response["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded = False
            logger.info("Gemma 4 vision unloaded")


_phi4_vision_service: Phi4VisionService | None = None


def get_phi4_vision_service() -> Phi4VisionService:
    global _phi4_vision_service
    if _phi4_vision_service is None:
        _phi4_vision_service = Phi4VisionService()
    return _phi4_vision_service
