"""See-Through — Layer decomposition for anime character illustrations.

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
Runs in Docker container (tech-noir/seethrough:latest) via HTTPToolMixin.

Requires ~4GB VRAM. Docker image runs FastAPI on port 8000 internally.
"""
from __future__ import annotations

import logging

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class SeeThroughDeployment(BaseGPUDeployment, HTTPToolMixin):
    """See-Through layer decomposition via Docker worker."""

    PORT = 18404

    def _load(self, model_name: str = "see-through") -> None:
        self._init_http(
            port=self.PORT,
            service_name="seethrough",
            timeout=120,
        )
        self.model = True
        self.model_name = model_name
        logger.info("See-Through HTTP ready (port=%d)", self.PORT)

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        self._ensure_healthy(port=self.PORT, service_name="seethrough", timeout=120)

    async def __call__(self, request):
        self._ensure_loaded()
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        resolution = int(form.get("resolution", "1280"))
        inference_steps = int(form.get("inference_steps", "30"))

        data = await self._call_worker(
            "decompose",
            files={"image": ("image.png", image_bytes, "image/png")},
            params={"resolution": resolution, "inference_steps": inference_steps},
        )

        import json
        return JSONResponse(json.loads(data))
