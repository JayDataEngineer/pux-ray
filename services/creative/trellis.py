"""TRELLIS.2 - Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Requires ~12GB VRAM. Conflicts with AniGen dependencies.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

TRELLIS_DIR = "/home/ubuntu/Documents/programs/TRELLIS.2"


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "runtime_env": {
            "working_dir": TRELLIS_DIR,
            "pip": ["torch>=2.1", "torchvision", "einops", "omegaconf",
                    "plyfile", "imageio", "trimesh", "pygltflib",
                    "accelerate", "safetensors"],
        },
    },
)
class TRELLISDeployment(BaseGPUDeployment):
    """TRELLIS.2 image-to-3D generation."""

    def _load(self, model_name: str = "trellis-base") -> None:
        import sys
        sys.path.insert(0, TRELLIS_DIR)

        from registry.models import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.get_path("3d", model_name)

        # TRELLIS uses its own pipeline
        from trellis.pipelines import TrellisImageTo3DPipeline
        self.model = TrellisImageTo3DPipeline.from_pretrained(str(model_path))
        self.model.to("cuda")
        self.model_name = model_name
        logger.info("TRELLIS loaded from %s", model_path)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def generate_3d(
        self,
        image: bytes,
        output_format: str = "glb",
        resolution: int = 256,
        seed: int = 0,
    ) -> bytes:
        """Generate 3D mesh from image bytes. Returns GLB or OBJ bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        from PIL import Image
        img = Image.open(io.BytesIO(image)).convert("RGBA")

        # Run TRELLIS inference
        outputs = self.model.run(
            image=img,
            seed=seed,
            formats=[output_format],
            sparse_structure_resolution=resolution,
        )

        # Extract the GLB/OBJ data
        if output_format in outputs:
            mesh_data = outputs[output_format]
            if hasattr(mesh_data, 'read'):
                return mesh_data.read()
            return mesh_data

        raise RuntimeError(f"TRELLIS did not produce {output_format} output")

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        output_format = form.get("output_format", "glb")
        resolution = int(form.get("resolution", "256"))

        glb_data = await self.generate_3d(
            image=image_bytes,
            output_format=output_format,
            resolution=resolution,
        )
        return Response(
            content=glb_data,
            media_type="model/gltf-binary" if output_format == "glb" else "application/octet-stream",
        )
