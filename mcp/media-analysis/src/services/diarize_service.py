"""Speaker diarization service.

Uses Pyannote 3.1 for speaker diarization.
Disabled by default — requires HuggingFace token with license accepted.
Lazy-loads on first request.
"""

import asyncio
import os
import subprocess
import tempfile
import time
from typing import Optional

import httpx
from loguru import logger

from ..settings import get_settings, get_device


class DiarizeService:
    """Speaker diarization via Pyannote 3.1."""

    def __init__(self):
        self._pipeline = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Diarization model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Diarization model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("pyannote"):
                raise RuntimeError(
                    "Speaker diarization is disabled. "
                    "Set MEDIA_PYANNOTE_ENABLED=true and "
                    "MEDIA_PYANNOTE_TOKEN=<your_hf_token> to enable. "
                    "Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1"
                )

            if not settings.pyannote_token:
                raise RuntimeError(
                    "Pyannote requires a HuggingFace token. "
                    "Set MEDIA_PYANNOTE_TOKEN and accept the license at "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                )

            try:
                logger.info("Loading speaker diarization model (Pyannote 3.1)")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"Diarization model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load diarization model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        settings = get_settings()
        device = get_device()

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.pyannote_token,
        )

        if device == "cuda":
            self._pipeline.to(torch.device("cuda"))

    async def diarize(
        self,
        audio_url: str,
        num_speakers: int | None = None,
    ) -> dict:
        """Perform speaker diarization on audio."""
        await self._ensure_loaded()

        try:
            # Download and convert audio to WAV
            audio_path = await self._download_and_convert(audio_url)

            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self._diarize_sync, audio_path, num_speakers,
                    )
                    return result
            finally:
                os.unlink(audio_path)

        except RuntimeError as e:
            return {"success": False, "error": str(e)[:300]}
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return {"success": False, "error": f"Diarization error: {str(e)[:200]}"}

    def _diarize_sync(self, audio_path: str, num_speakers: int | None) -> dict:
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = self._pipeline(audio_path, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3),
                "speaker": speaker,
            })

        # Get unique speakers
        speakers = sorted(set(s["speaker"] for s in segments))

        return {
            "success": True,
            "segments": segments,
            "total_segments": len(segments),
            "speakers": speakers,
            "num_speakers": len(speakers),
        }

    async def _download_and_convert(self, audio_url: str) -> str:
        """Download audio and convert to WAV."""
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(audio_url, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()

        suffix = "." + audio_url.rsplit(".", 1)[-1] if "." in audio_url else ".wav"
        raw = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        raw.write(response.content)
        raw.close()

        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()

        subprocess.run(
            ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
            capture_output=True, check=True,
        )
        os.unlink(raw.name)

        return wav.name

    async def close(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            logger.info("Diarization model unloaded")


_diarize_service: DiarizeService | None = None


def get_diarize_service() -> DiarizeService:
    global _diarize_service
    if _diarize_service is None:
        _diarize_service = DiarizeService()
    return _diarize_service
