"""ASR service using Parakeet TDT v3 via onnx-asr.

Lazy-loads on first request. CPU-optimized, ~36x realtime.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from ..settings import get_settings


class AsrService:
    """Parakeet TDT v3 speech-to-text."""

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"ASR model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"ASR model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("asr"):
                raise RuntimeError("ASR model is disabled")

            try:
                logger.info(f"Loading ASR model: {settings.asr_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"ASR model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load ASR model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import onnx_asr

        settings = get_settings()
        self._model = onnx_asr.load_model(settings.asr_model)

    async def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to text."""
        await self._ensure_loaded()

        async with self._lock:
            try:
                settings = get_settings()
                loop = asyncio.get_event_loop()
                text = await asyncio.wait_for(
                    loop.run_in_executor(None, self._transcribe_sync, audio_path),
                    timeout=120.0,
                )
                return {"success": True, "text": text}

            except asyncio.TimeoutError:
                return {"success": False, "error": "Transcription timed out"}
            except Exception as e:
                logger.error(f"ASR error: {e}")
                return {"success": False, "error": f"ASR error: {str(e)[:200]}"}

    def _transcribe_sync(self, audio_path: str) -> str:
        return self._model.recognize(audio_path)

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("ASR model unloaded")


_asr_service: AsrService | None = None


def get_asr_service() -> AsrService:
    global _asr_service
    if _asr_service is None:
        _asr_service = AsrService()
    return _asr_service
