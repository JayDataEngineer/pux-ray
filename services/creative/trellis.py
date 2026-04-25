"""TRELLIS.2 - Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Called via subprocess using TRELLIS.2's own venv Python because
it has compiled CUDA extensions (o_voxel, cumesh, nvdiffrast)
that can't be pip-installed dynamically.

Requires ~12GB VRAM. Model loads fresh per subprocess call (~30s overhead).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve

from services.base import BaseGPUDeployment, CLIToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01},
)
class TRELLISDeployment(BaseGPUDeployment, CLIToolMixin):
    """TRELLIS.2 image-to-3D generation via subprocess CLI."""

    def _load(self, model_name: str = "trellis-base") -> None:
        self._init_cli("services.creative.trellis")
        self.model = True
        self.model_name = model_name
        logger.info("TRELLIS CLI tool ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        # Nothing in-process to unload — subprocess handles its own cleanup
        self.model = None

    async def generate_3d(
        self,
        image: bytes,
        output_format: str = "glb",
        resolution: int = 512,
        seed: int = 0,
    ) -> bytes:
        """Generate 3D mesh from image bytes. Returns GLB or OBJ bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        with tempfile.TemporaryDirectory(prefix="trellis_") as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / f"output.{output_format}"

            input_path.write_bytes(image)

            args = [
                "--image", str(input_path),
                "--output", str(output_path),
                "--resolution", str(resolution),
            ]

            self._run_cli(args, timeout=600)

            if not output_path.exists():
                raise RuntimeError(f"TRELLIS did not produce output at {output_path}")

            return output_path.read_bytes()

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
