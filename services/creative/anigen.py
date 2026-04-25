"""AniGen - Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes. Requires ~14GB VRAM.
Conflicts with TRELLIS dependencies - uses separate runtime_env.
"""

from __future__ import annotations

import io
import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

ANIGEN_DIR = "/home/ubuntu/Documents/programs/AniGen"


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "runtime_env": {
            "working_dir": ANIGEN_DIR,
            "pip": ["torch>=2.1", "torchvision", "einops", "trimesh",
                    "pygltflib", "accelerate", "safetensors", "pytorch3d"],
        },
    },
)
class AniGenDeployment(BaseGPUDeployment):
    """AniGen animated 3D asset generation."""

    def _load(self, model_name: str = "anigen-base") -> None:
        import sys
        sys.path.insert(0, ANIGEN_DIR)

        from registry.models import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.get_path("3d", model_name)

        # AniGen loads its own models from ckpts directory
        # Exact import depends on AniGen's codebase structure
        logger.info("AniGen loading from %s (stub - needs model code)", model_path)
        self.model = True  # placeholder until exact API is determined
        self.model_name = model_name

    def _unload(self) -> None:
        self.model = None

    async def generate_rigged(
        self,
        image: bytes,
        output_format: str = "glb",
    ) -> bytes:
        """Generate rigged 3D mesh from image."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        # AniGen inference - needs exact model code
        raise NotImplementedError("AniGen inference TBD - needs model code")

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        glb_data = await self.generate_rigged(image=image_bytes)
        from starlette.responses import Response
        return Response(content=glb_data, media_type="model/gltf-binary")
