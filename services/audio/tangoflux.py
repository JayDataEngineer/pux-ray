"""TangoFlux — Super Fast Text-to-Audio generation.

Flow matching with DiT/MMDiT and CRPO alignment. Generates 44.1kHz audio
up to 30 seconds from text descriptions.
Requires ~6GB VRAM.
"""
from __future__ import annotations

import io
import logging
import os

from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment, _free_cuda_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("TANGOFLUX_MODEL_PATH", "/models/audio/tangoflux")


@serve.deployment(
    name="tangoflux",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class TangoFluxDeployment(BaseGPUDeployment):
    """TangoFlux text-to-audio."""

    def _load(self, model_name: str = "tangoflux") -> None:
        from tangoflux import TangoFluxInference

        self.model = TangoFluxInference(name="declare-lab/TangoFlux")
        self.model_name = model_name
        logger.info("TangoFlux loaded")

    def _unload(self) -> None:
        self.model = None
        _free_cuda_cache()

    async def __call__(self, request):
        if not self.is_loaded():
            self.load_model("tangoflux")

        body = await request.json()
        prompt = body.get("prompt", body.get("caption", ""))
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        steps = body.get("steps", 50)
        duration = body.get("duration", 10)

        try:
            import soundfile as sf

            audio = self.model.generate(prompt, steps=steps, duration=duration)
            buf = io.BytesIO()
            sf.write(buf, audio, 44100, format="WAV")
            buf.seek(0)
            return Response(content=buf.read(), media_type="audio/wav")

        except Exception as e:
            logger.exception("TangoFlux generation failed")
            return JSONResponse({"error": str(e)}, status_code=500)
