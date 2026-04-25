"""IndexTTS - GPU text-to-speech using IndexTTS-2 model.

High-quality multi-speaker TTS. Requires ~13GB VRAM.
"""

from __future__ import annotations

import logging
from typing import Optional

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="index_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0.01,
    },
)
class IndexTTSDeployment(BaseGPUDeployment):
    """GPU-based IndexTTS-2. High quality multi-speaker TTS."""

    def _load(self, model_name: str = "index-tts") -> None:
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        model_path = registry.get_path("tts", model_name)

        # IndexTTS loads from directory with gpt.pth, s2mel.pth, etc.
        # The actual import depends on the IndexTTS library structure
        import sys
        sys.path.insert(0, str(model_path))

        from index_tts import IndexTTSModel
        self.model = IndexTTSModel(str(model_path))
        self.model.to("cuda")
        self.model_name = model_name

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_format: str = "wav",
    ) -> bytes:
        """Synthesize speech. Returns audio bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        audio = self.model.synthesize(text, voice=voice)
        import soundfile as sf
        import io as _io
        buf = _io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        return buf.getvalue()

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "default"),
            output_format=body.get("response_format", "wav"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
