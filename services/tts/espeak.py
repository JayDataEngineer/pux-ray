"""eSpeak-NG TTS - ultralight CPU-only phoneme synthesis.

No GPU needed. Uses espeak-ng binary via subprocess.
Conforms to TNAP: all requests/responses use the unified protocol schema.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, TNAPInput

logger = logging.getLogger(__name__)


@serve.deployment(
    name="espeak_tts",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0},
    max_ongoing_requests=4,
)
class EspeakTTS(BaseGPUDeployment):
    """eSpeak-NG TTS. Zero GPU, instant startup."""

    def __init__(self):
        super().__init__()
        from registry.config import Config
        self._espeak_bin = Config().get("binaries.espeak_ng", "espeak-ng")

    def _load(self, model_name: str = "espeak") -> None:
        result = subprocess.run(
            ["which", self._espeak_bin], capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"espeak-ng binary not found: {self._espeak_bin}. "
                f"Install with: apt install espeak-ng"
            )
        self.model = True
        self.model_name = model_name
        logger.info("eSpeak TTS ready (bin=%s)", self._espeak_bin)

    def _unload(self) -> None:
        self.model = None
        self.model_name = None
        super()._unload()

    def synthesize(
        self,
        text: str,
        voice: str = "en",
        speed: int = 175,
        pitch: int = 50,
    ) -> bytes:
        """Synthesize speech via espeak-ng. Returns WAV bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            cmd = [
                self._espeak_bin,
                "-v", voice,
                "-s", str(speed),
                "-p", str(pitch),
                "-w", tmp.name,
                text,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return Path(tmp.name).read_bytes()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, voice}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            audio = await asyncio.to_thread(
                lambda: self.synthesize(
                    text=extracted.get("text", ""),
                    voice=extracted.get("voice", "en"),
                    speed=extracted.get("speed", 175),
                    pitch=extracted.get("pitch", 50),
                ),
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("espeak_tts error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)