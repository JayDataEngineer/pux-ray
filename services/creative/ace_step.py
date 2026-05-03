"""ACE-STEP — Music generation from text prompts.

Generates music from text descriptions. Runs in Docker container
(tech-noir/acestep:latest) via HTTPToolMixin.

Requires ~8GB VRAM. Docker image runs FastAPI on port 8000 internally.
"""
from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="ace_step",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class ACEStepDeployment(BaseGPUDeployment, HTTPToolMixin):
    """ACE-STEP music generation via Docker worker."""

    PORT = 18405

    def _load(self, model_name: str = "ace-step") -> None:
        self._init_http(port=self.PORT, service_name="acestep", timeout=120)
        self.model = True
        self.model_name = model_name
        logger.info("ACE-STEP HTTP ready (port=%d)", self.PORT)

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        self._ensure_healthy(port=self.PORT, service_name="acestep", timeout=120)

    async def __call__(self, request):
        self._ensure_loaded()
        body = await request.json()
        prompt = body.get("prompt", body.get("caption", ""))
        if not prompt:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        data = await self._call_worker(
            "generate",
            json={
                "prompt": prompt,
                "duration": body.get("duration", 30),
                "bpm": body.get("bpm", 120),
                "instrumental": body.get("instrumental", True),
                "seed": body.get("seed", -1),
                "audio_format": body.get("audio_format", "wav"),
                "task_type": body.get("task_type", "text2music"),
            },
        )

        fmt = body.get("audio_format", "wav")
        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}
        return Response(content=data, media_type=media_types.get(fmt, "audio/wav"))
