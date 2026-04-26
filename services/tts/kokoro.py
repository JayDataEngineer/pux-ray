"""Kokoro TTS - lightweight CPU text-to-speech.

Uses the kokoro library (82M params) for fast CPU inference.
"""

from __future__ import annotations

import io
import logging
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
        self._model = None
        self._pipeline = None

    def _ensure_model(self):
        if self._pipeline is not None:
            return

        from kokoro import KModel, KPipeline

        self._model = KModel()
        self._pipeline = KPipeline(lang_code="a", model=self._model)
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

        voice_pack = self._pipeline.load_voice(voice)
        # pipeline() yields (graphemes, phonemes, audio_tensor) segments
        audio_chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice_pack, speed=speed):
            audio_chunks.append(audio.cpu().numpy())

        import numpy as np
        audio_data = np.concatenate(audio_chunks) if audio_chunks else np.array([])

        if output_format == "wav":
            return self._to_wav(audio_data)
        return self._to_wav(audio_data)

    def _to_wav(self, audio, sample_rate: int = 24000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype("int16").tobytes())
        return buf.getvalue()

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
