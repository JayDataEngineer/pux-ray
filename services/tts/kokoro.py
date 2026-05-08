"""Kokoro TTS - lightweight CPU text-to-speech.

Uses the kokoro library (82M params) for fast CPU inference.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="kokoro_tts",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0},
    max_ongoing_requests=4,
)
class KokoroTTS(BaseGPUDeployment):
    """CPU-based Kokoro TTS. No GPU needed."""

    def __init__(self):
        super().__init__()
        self._pipeline = None

    def _load(self, model_name: str = "kokoro") -> None:
        from kokoro import KModel, KPipeline

        self._model = KModel()
        self._pipeline = KPipeline(lang_code="a", model=self._model)
        self.model = True
        self.model_name = model_name
        logger.info("Kokoro TTS loaded (CPU)")

    def _unload(self) -> None:
        self._model = None
        self._pipeline = None
        self.model = None
        self.model_name = None
        super()._unload()

    def synthesize(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
    ) -> bytes:
        """Synthesize speech. Returns audio bytes (blocking CPU work)."""
        if not self.is_loaded():
            self.load_model("kokoro")

        voice_pack = self._pipeline.load_voice(voice)
        audio_chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice_pack, speed=speed):
            audio_chunks.append(audio.cpu().numpy())

        import numpy as np
        audio_data = np.concatenate(audio_chunks) if audio_chunks else np.array([])
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
        """TNAP endpoint: {action, input: {text, voice}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            audio = await asyncio.to_thread(
                self.synthesize,
                text=extracted.get("text", ""),
                voice=extracted.get("voice", "af_bella"),
                speed=extracted.get("speed", 1.0),
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("kokoro_tts error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)
