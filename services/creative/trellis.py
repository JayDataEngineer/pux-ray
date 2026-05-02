"""TRELLIS.2 - Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Runs in a Docker container (CUDA 12.4) accessed via HTTPToolMixin.
"""

from __future__ import annotations

import logging

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class TRELLISDeployment(BaseGPUDeployment, HTTPToolMixin):
    """TRELLIS.2 image-to-3D generation via Docker worker."""

    def _load(self, model_name: str = "trellis") -> None:
        self._init_http(port=18401, service_name="trellis", timeout=600)
        self.model = True
        self.model_name = model_name
        logger.info("TRELLIS HTTP ready (port=18401)")

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
        output_format = form.get("output_format", "glb")
        resolution = form.get("resolution", "512")
        decimation = form.get("decimation", "1000000")

        data = await self._call_worker(
            "generate",
            files={"image": ("image.png", image_bytes, "image/png")},
            data={
                "output_format": output_format,
                "resolution": resolution,
                "decimation": decimation,
            },
        )

        return Response(
            content=data,
            media_type="model/gltf-binary" if output_format == "glb" else "application/octet-stream",
        )
