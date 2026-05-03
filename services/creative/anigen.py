"""AniGen - Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Runs in a Docker container (CUDA 12.1) accessed via HTTPToolMixin.
"""

from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class AniGenDeployment(BaseGPUDeployment, HTTPToolMixin):
    """AniGen animated 3D asset generation via Docker worker."""

    def _load(self, model_name: str = "anigen") -> None:
        self._init_http(port=18402, service_name="anigen", timeout=600)
        self.model = True
        self.model_name = model_name
        logger.info("AniGen HTTP ready (port=18402)")

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        if not hasattr(self, "_base_url"):
            self._load()

    async def __call__(self, request):
        self._ensure_loaded()
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        seed = form.get("seed", "42")

        data = await self._call_worker(
            "generate",
            files={"image": ("image.png", image_bytes, "image/png")},
            data={"seed": seed},
        )

        return Response(content=data, media_type="model/gltf-binary")
