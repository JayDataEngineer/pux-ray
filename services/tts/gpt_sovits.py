"""GPT-SoVITS — Voice cloning TTS.

Clones voices from reference audio using GPT-SoVITS.
Runs in Docker container (tech-noir/gptsovits:latest) via HTTPToolMixin.
"""
from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class GPTSoVITSDeployment(BaseGPUDeployment, HTTPToolMixin):
    """GPT-SoVITS voice cloning TTS via Docker worker."""

    PORT = 18406

    def _load(self, model_name: str = "gpt-sovits") -> None:
        self._init_http(port=self.PORT, service_name="gptsovits", timeout=120)
        self.model = True
        self.model_name = model_name
        logger.info("GPT-SoVITS HTTP ready (port=%d)", self.PORT)

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        self._ensure_healthy(port=self.PORT, service_name="gptsovits", timeout=120)

    async def __call__(self, request):
        self._ensure_loaded()
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        # For reference audio from path (already on disk)
        refer_wav_path = body.get("refer_wav_path", "")

        import io
        if refer_wav_path:
            # Read file from host path
            import subprocess
            result = subprocess.run(
                ["curl", "-s", "--data-binary", f"@{refer_wav_path}"],
                capture_output=True,
            )
            data = await self._call_worker(
                "synthesize",
                data={
                    "text": text,
                    "prompt_text": body.get("prompt_text", ""),
                    "prompt_language": body.get("prompt_language", "en"),
                    "text_language": body.get("text_language", "en"),
                },
                files={"refer_wav": ("reference.wav", Path(refer_wav_path).read_bytes(), "audio/wav")},
            )
        else:
            data = await self._call_worker(
                "synthesize",
                data={
                    "text": text,
                    "prompt_text": body.get("prompt_text", ""),
                    "prompt_language": body.get("prompt_language", "en"),
                    "text_language": body.get("text_language", "en"),
                },
            )

        return Response(content=data, media_type="audio/wav")
