"""eSpeak-NG TTS - ultralight CPU-only phoneme synthesis.

No GPU needed. Uses espeak-ng binary via subprocess.
"""

from __future__ import annotations

import io
import logging
import subprocess
import tempfile
from pathlib import Path

from ray import serve

from registry.config import Config

logger = logging.getLogger(__name__)


@serve.deployment(
    name="espeak_tts",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    max_ongoing_requests=4,
)
class EspeakTTS:
    """eSpeak-NG TTS. Zero GPU, instant startup."""

    _ESPEAK_BIN = Config().get("binaries.espeak_ng", "espeak-ng")

    def __init__(self):
        self._check_binary()

    def _check_binary(self):
        result = subprocess.run(
            ["which", self._ESPEAK_BIN], capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("espeak-ng not found. Install: apt install espeak-ng")

    async def synthesize(
        self,
        text: str,
        voice: str = "en",
        speed: int = 175,
        pitch: int = 50,
        output_format: str = "wav",
    ) -> bytes:
        """Synthesize speech via espeak-ng."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            cmd = [
                self._ESPEAK_BIN,
                "-v", voice,
                "-s", str(speed),
                "-p", str(pitch),
                "-w", tmp.name,
                text,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return Path(tmp.name).read_bytes()

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "en"),
            output_format=body.get("response_format", "wav"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
