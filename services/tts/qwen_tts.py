"""Qwen3-TTS - GPU text-to-speech using Qwen3 TTS model.

Multi-speaker TTS with voice cloning support.
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="qwen_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0.01,
        "num_cpus": 0.5,
    },
)
class QwenTTSDeployment(BaseGPUDeployment):
    """GPU-based Qwen3-TTS with multiple voice modes."""

    def _load(self, model_name: str = "qwen3-tts") -> None:
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        try:
            model_path = registry.get_path("tts", model_name)
        except (KeyError, ValueError):
            model_path = registry.get_path("tts", "qwen3-tts")
            model_name = "qwen3-tts"

        # Qwen3-TTS uses a specific pipeline
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(str(model_path))
        self.model.to("cuda")
        self.model_name = model_name
        logger.info("Qwen3-TTS loaded from %s", model_path)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def synthesize(
        self,
        text: str,
        voice: str = "Chelsie",
        mode: str = "customvoice",
        output_format: str = "wav",
    ) -> bytes:
        """Synthesize speech.

        Modes: customvoice, voicedesign, clone, fast
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        audio = self.model.generate_custom_voice(text, voice=voice)
        import soundfile as sf
        import io
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        return buf.getvalue()

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "Chelsie"),
            mode=body.get("mode", "customvoice"),
            output_format=body.get("response_format", "wav"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
