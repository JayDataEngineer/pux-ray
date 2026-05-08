"""TangoFlux — Super Fast Text-to-Audio generation.

Flow matching with DiT/MMDiT and CRPO alignment. Generates 44.1kHz audio
up to 30 seconds from text descriptions.
Requires ~6GB VRAM.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

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
        from safetensors.torch import load_file

        from diffusers import AutoencoderOobleck
        from tangoflux.model import TangoFlux

        path = MODEL_PATH

        self.vae = AutoencoderOobleck()
        vae_weights = load_file(os.path.join(path, "vae.safetensors"))
        self.vae.load_state_dict(vae_weights)

        weights = load_file(os.path.join(path, "tangoflux.safetensors"))
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        self.model = TangoFlux(config)
        self.model.load_state_dict(weights, strict=False)

        self.vae.to("cuda")
        self.model.to("cuda")
        self.model_name = model_name
        logger.info("TangoFlux loaded from %s", path)

    def _unload(self) -> None:
        self.model = None
        self.vae = None
        super()._unload()

    def _run_generate(self, prompt: str, steps: int, duration: float, guidance_scale: float) -> bytes:
        import soundfile as sf

        audio = self.model.inference_flow(
            prompt,
            duration=duration,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )
        wave = self.vae.decode(audio.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * self.vae.config.sampling_rate)
        audio_np = wave[:, :waveform_end].numpy().T

        buf = io.BytesIO()
        sf.write(buf, audio_np, int(self.vae.config.sampling_rate), format="WAV")
        buf.seek(0)
        return buf.read()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {prompt, steps, duration, guidance}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "tangoflux")

            prompt = extracted.get("prompt", "")
            if not prompt:
                return JSONResponse(self.handle_error("prompt required"), status_code=400)

            audio_bytes = await asyncio.to_thread(
                lambda: self._run_generate(
                    prompt,
                    steps=extracted.get("steps", 50),
                    duration=extracted.get("duration", 10.0),
                    guidance_scale=extracted.get("guidance", 4.5),
                ),
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio_bytes, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.exception("TangoFlux generation failed")
            return JSONResponse(self.handle_error(str(e)), status_code=500)