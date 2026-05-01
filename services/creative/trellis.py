"""TRELLIS.2 - Image-to-3D mesh generation.

Generates high-quality 3D meshes (GLB) from single images.
Uses CLIToolMixin subprocess pattern — the tool's venv has compiled
CUDA extensions (o_voxel, CuMesh, flash-attn, nvdiffrast, FlexGEMM, nvdiffrec).

Requires ~12GB VRAM. Model loads fresh per subprocess call (~30s overhead).
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
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class TRELLISDeployment(BaseGPUDeployment, CLIToolMixin):
    """TRELLIS.2 image-to-3D generation via subprocess CLI."""

    def _load(self, model_name: str = "trellis") -> None:
        self._init_cli("services.creative.trellis")
        from registry.config import Config
        self._model_path = Config().get("services.creative.trellis.model_path",
                                        "/models/3d/trellis/TRELLIS.2-4B")
        self.model = True
        self.model_name = model_name
        logger.info("TRELLIS CLI ready (model_path=%s)", self._model_path)

    def _unload(self) -> None:
        # Model loads/unloads per subprocess call — nothing in-process
        self.model = None

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        output_format = form.get("output_format", "glb")
        resolution = form.get("resolution", "512")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.glb"
            input_path.write_bytes(image_bytes)

            result = self._run_cli([
                "--image", str(input_path),
                "--output", str(output_path),
                "--resolution", resolution,
                "--model", self._model_path,
            ])

            if result.returncode != 0:
                raise RuntimeError(
                    f"TRELLIS failed (exit {result.returncode}): "
                    f"{result.stderr[-500:]}"
                )

            if not output_path.exists():
                raise RuntimeError("TRELLIS produced no output file")

            glb_data = output_path.read_bytes()

        return Response(
            content=glb_data,
            media_type="model/gltf-binary" if output_format == "glb" else "application/octet-stream",
        )
