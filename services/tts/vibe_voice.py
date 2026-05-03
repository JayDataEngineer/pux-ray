"""VibeVoice TTS - long-form multi-speaker speech synthesis.

Generates expressive, long-form audio (podcasts, conversations) from text
using the VibeVoice 7B model. Supports up to 4 speakers, up to 45 min output.

Runs in a Docker container (CUDA 12.4) accessed via HTTPToolMixin.

Code repo: https://github.com/microsoft/VibeVoice
Model weights: vibevoice/VibeVoice-7B on HuggingFace (18.7GB, community re-upload)
ASR is separate: services.asr.gpu_asr.VibeVoiceASRDeployment
"""

from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class VibeVoiceDeployment(BaseGPUDeployment, HTTPToolMixin):
    """VibeVoice long-form multi-speaker TTS via Docker worker."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        self._init_http(port=18403, service_name="vibevoice", timeout=600)
        self.model = True
        self.model_name = model_name
        logger.info("VibeVoice HTTP ready (port=18403)")

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        self._ensure_healthy(port=18403, service_name="vibevoice", timeout=600)

    async def __call__(self, request):
        self._ensure_loaded()
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = speaker_names.split(",")

        data = await self._call_worker(
            "generate",
            json={
                "input": text,
                "speaker_names": speaker_names,
            },
        )

        return Response(content=data, media_type="audio/wav")
