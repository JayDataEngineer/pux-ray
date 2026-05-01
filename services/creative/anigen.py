"""AniGen - Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Uses CLIToolMixin subprocess pattern — the tool's venv has compiled
CUDA extensions (pytorch3d, spconv, flash-attn).

Requires ~14GB VRAM. Model loads fresh per subprocess call (~60s overhead).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, CLIToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class AniGenDeployment(BaseGPUDeployment, CLIToolMixin):
    """AniGen animated 3D asset generation via subprocess CLI."""

    def _load(self, model_name: str = "anigen") -> None:
        self._init_cli("services.creative.anigen")
        self.model = True
        self.model_name = model_name
        logger.info("AniGen CLI ready")

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.glb"
            input_path.write_bytes(image_bytes)

            result = self._run_cli([
                "--image", str(input_path),
                "--output", str(output_path),
            ])

            if result.returncode != 0:
                raise RuntimeError(
                    f"AniGen failed (exit {result.returncode}): "
                    f"{result.stderr[-500:]}"
                )

            if not output_path.exists():
                raise RuntimeError("AniGen produced no output file")

            glb_data = output_path.read_bytes()

        return Response(
            content=glb_data,
            media_type="model/gltf-binary",
        )
