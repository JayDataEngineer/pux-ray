"""VibeVoice TTS - long-form multi-speaker speech synthesis.

Generates expressive, long-form audio (podcasts, conversations) from text
using the VibeVoice 7B model. Supports up to 4 speakers, up to 45 min output.

Runs in a Docker container with CUDA 12.4 for compiled flash-attn and
pinned transformers==4.51.3. Ray head calls it via HTTP.

Code repo: https://github.com/microsoft/VibeVoice
Model weights: vibevoice/VibeVoice-7B on HuggingFace (18.7GB, community re-upload)
ASR is separate: services.asr.gpu_asr.VibeVoiceASRDeployment
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "num_cpus": 0.5,
    },
)
class VibeVoiceDeployment(BaseGPUDeployment, HTTPToolMixin):
    """VibeVoice long-form multi-speaker TTS via Docker HTTP worker."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        self._init_http(port=18403, service_name="vibevoice")
        self.model = True
        self.model_name = model_name
        logger.info("VibeVoice HTTP worker ready (container managed by GPUScheduler)")

    def _unload(self) -> None:
        self.model = None

    async def synthesize(
        self,
        text: str,
        speaker_names: list[str] | None = None,
        model_path: str = "vibevoice/VibeVoice-7B",
    ) -> bytes:
        """Synthesize speech from text via HTTP worker."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        if not speaker_names:
            speaker_names = ["Andrew"]

        return await self._call_worker(
            "generate",
            json={
                "input": text,
                "speaker_names": speaker_names,
                "model_path": model_path,
            },
        )

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = speaker_names.split(",")

        audio = await self.synthesize(
            text=text,
            speaker_names=speaker_names,
            model_path=body.get("model_path", "vibevoice/VibeVoice-7B"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
