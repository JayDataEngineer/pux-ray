"""Kokoro TTS - lightweight CPU text-to-speech.

Uses the kokoro library (82M params) for fast CPU inference.
Wraps espeak-ng as phonemizer backend.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from typing import Optional

from ray import serve

logger = logging.getLogger(__name__)


@serve.deployment(
    name="kokoro_tts",
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
    max_ongoing_requests=4,
)
class KokoroTTS:
    """CPU-based Kokoro TTS. No GPU needed."""

    def __init__(self):
        # Lazy import - kokoro may not be installed yet
        pass

    def _ensure_model(self):
        if not hasattr(self, "_model") or self._model is None:
            from kokoro import KokoroModel
            from registry.models import ModelRegistry

            registry = ModelRegistry()
            model_path = registry.get_path("tts", "kokoro")
            if model_path.exists():
                self._model = KokoroModel.load(str(model_path))
            else:
                # Try loading from HuggingFace
                self._model = KokoroModel.load()
            logger.info("Kokoro TTS loaded (CPU)")

    async def synthesize(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> bytes:
        """Synthesize speech. Returns audio bytes."""
        self._ensure_model()
        audio = self._model.synthesize(text, voice=voice, speed=speed)

        if output_format == "wav":
            return self._to_wav(audio)
        elif output_format == "mp3":
            return self._to_mp3(audio)
        else:
            return self._to_wav(audio)

    def _to_wav(self, audio, sample_rate: int = 24000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype("int16").tobytes())
        return buf.getvalue()

    def _to_mp3(self, audio, sample_rate: int = 24000) -> bytes:
        # Requires ffmpeg or pydub
        wav_data = self._to_wav(audio, sample_rate)
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
            input=wav_data, capture_output=True,
        )
        return result.stdout

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "af_bella"),
            speed=body.get("speed", 1.0),
            output_format=body.get("response_format", "wav"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
