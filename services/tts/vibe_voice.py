"""VibeVoice TTS/ASR - unified voice service.

Supports TTS (0.5B streaming, 1.5B, 7B) and ASR (7B with diarization).
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0.01,
        "num_cpus": 0.5,
    },
)
class VibeVoiceDeployment(BaseGPUDeployment):
    """VibeVoice unified TTS/ASR."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        meta = registry.get_metadata("tts", "vibevoice-tts")
        model_path = registry.get_path("tts", "vibevoice-tts")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype="auto", device_map="auto",
        )
        self.model_name = model_name
        logger.info("VibeVoice loaded: %s", model_name)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None

    async def synthesize(self, text: str, voice: str = "default",
                         output_format: str = "wav") -> bytes:
        if not self.is_loaded():
            raise RuntimeError("No model loaded")
        # VibeVoice TTS inference
        # Implementation depends on specific VibeVoice API
        raise NotImplementedError("VibeVoice TTS inference TBD - needs model code")

    async def __call__(self, request):
        body = await request.json()
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "default"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
