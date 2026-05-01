"""TRELLIS.2 - Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Runs in a Docker container with CUDA 12.4 for compiled extensions
(o_voxel, CuMesh, flash-attn, nvdiffrast). Ray head calls it via HTTP.

Requires ~12GB VRAM. Model stays loaded in container between requests.
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class TRELLISDeployment(BaseGPUDeployment, HTTPToolMixin):
    """TRELLIS.2 image-to-3D generation via Docker HTTP worker."""

    def _load(self, model_name: str = "trellis-base") -> None:
        self._init_http(port=18401, service_name="trellis")
        self.model = True
        self.model_name = model_name
        logger.info("TRELLIS HTTP worker ready (container managed by GPUScheduler)")

    def _unload(self) -> None:
        # Container lifecycle managed by GPUScheduler
        self.model = None

    async def generate_3d(
        self,
        image: bytes,
        output_format: str = "glb",
        resolution: int = 512,
        seed: int = 0,
    ) -> bytes:
        """Generate 3D mesh from image bytes via HTTP worker."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        result = await self._call_worker(
            "generate",
            files={"image": ("input.png", image, "image/png")},
            data={
                "output_format": output_format,
                "resolution": str(resolution),
            },
        )
        return result

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        output_format = form.get("output_format", "glb")
        resolution = int(form.get("resolution", "512"))

        glb_data = await self.generate_3d(
            image=image_bytes,
            output_format=output_format,
            resolution=resolution,
        )
        from starlette.responses import Response
        return Response(
            content=glb_data,
            media_type="model/gltf-binary" if output_format == "glb" else "application/octet-stream",
        )
