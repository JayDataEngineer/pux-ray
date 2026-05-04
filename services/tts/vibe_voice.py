"""VibeVoice TTS — Long-form multi-speaker speech synthesis.

Generates expressive, long-form audio (podcasts, conversations) from text
using the VibeVoice 7B model. Supports up to 4 speakers, up to 45 min output.

Runs inside Ray-managed container (tech-noir/vibevoice:latest).
Pipeline imports directly — no subprocess or HTTP layer needed.

Code repo: https://github.com/microsoft/VibeVoice
Model weights: vibevoice/VibeVoice-7B on HuggingFace (18.7GB)
ASR is separate: services.asr.gpu_asr.VibeVoiceASRDeployment
"""
from __future__ import annotations

import io
import logging
import os
import sys

import torch
from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment, _free_cuda_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/audio/vibevoice/VibeVoice-7B")


@serve.deployment(
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class VibeVoiceDeployment(BaseGPUDeployment):
    """VibeVoice long-form multi-speaker TTS via Ray native container."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        sys.path.insert(0, "/app/repo")

        from vibevoice.model import VibeVoicePipeline

        self.pipeline = VibeVoicePipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model_name = model_name
        self.model = True
        logger.info("VibeVoice loaded: %s", MODEL_PATH)

    def _unload(self) -> None:
        self.pipeline = None
        self.model = None
        _free_cuda_cache()

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = [s.strip() for s in speaker_names.split(",")]

        output = self.pipeline.run(text=text, speaker_names=speaker_names)

        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, output["audio"], output["sample_rate"], format="WAV")
        buf.seek(0)

        return Response(content=buf.read(), media_type="audio/wav")
