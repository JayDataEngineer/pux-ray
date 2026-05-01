"""AniGen - Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Runs in a Docker container with CUDA 12.1 for pytorch3d (source build).
Ray head calls it via HTTP.

Requires ~14GB VRAM. Model stays loaded in container between requests.
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class AniGenDeployment(BaseGPUDeployment, HTTPToolMixin):
    """AniGen animated 3D asset generation via Docker HTTP worker."""

    def _load(self, model_name: str = "anigen-base") -> None:
        self._init_http(port=18402, service_name="anigen")
        self.model = True
        self.model_name = model_name
        logger.info("AniGen HTTP worker ready (container managed by GPUScheduler)")

    def _unload(self) -> None:
        self.model = None

    async def generate_rigged(
        self,
        image: bytes,
        output_format: str = "glb",
        ss_model: str = "ckpts/anigen/ss_flow_duet",
        slat_model: str = "ckpts/anigen/slat_flow_auto",
        seed: int = 42,
    ) -> bytes:
        """Generate rigged 3D mesh from image via HTTP worker. Returns GLB bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        return await self._call_worker(
            "generate",
            files={"image": ("input.png", image, "image/png")},
            data={
                "ss_model": ss_model,
                "slat_model": slat_model,
                "seed": str(seed),
            },
        )

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        ss_model = form.get("ss_model", "ckpts/anigen/ss_flow_duet")
        slat_model = form.get("slat_model", "ckpts/anigen/slat_flow_auto")

        glb_data = await self.generate_rigged(
            image=image_bytes,
            ss_model=ss_model,
            slat_model=slat_model,
        )
        from starlette.responses import Response
        return Response(
            content=glb_data,
            media_type="model/gltf-binary",
        )
