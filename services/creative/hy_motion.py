"""HY-Motion 1.0 — Text-to-3D human motion generation.

Generates skeleton-based 3D character animations from text prompts.
Runs in Docker container (tech-noir/hymotion:latest) via HTTPToolMixin.

Requires ~26GB VRAM for HY-Motion-1.0, ~24GB for Lite variant.
"""
from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="hy_motion",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class HYMotionDeployment(BaseGPUDeployment, HTTPToolMixin):
    """HY-Motion text-to-3D motion via Docker worker."""

    PORT = 18407

    def _load(self, model_name: str = "hy-motion-1.0") -> None:
        self._init_http(port=self.PORT, service_name="hymotion", timeout=300)
        self.model = True
        self.model_name = model_name
        logger.info("HY-Motion HTTP ready (port=%d)", self.PORT)

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        self._ensure_healthy(port=self.PORT, service_name="hymotion", timeout=300)

    async def __call__(self, request):
        self._ensure_loaded()
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        data = await self._call_worker(
            "generate",
            json={
                "prompt": prompt,
                "duration": body.get("duration", 5.0),
                "seed": body.get("seed", 42),
                "format": body.get("format", "glb"),
            },
        )

        fmt = body.get("format", "glb")
        media_types = {
            "glb": "model/gltf-binary",
            "fbx": "application/octet-stream",
            "npz": "application/octet-stream",
        }
        return Response(content=data, media_type=media_types.get(fmt, "application/octet-stream"))
