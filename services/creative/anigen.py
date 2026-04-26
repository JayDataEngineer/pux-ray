"""AniGen - Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Called via subprocess using AniGen's own venv Python because it has
compiled extensions (pytorch3d, flash-attn, spconv) that can't be
pip-installed dynamically.

Requires ~14GB VRAM. Model loads fresh per subprocess call (~60s overhead).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve

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

    def _load(self, model_name: str = "anigen-base") -> None:
        self._init_cli("services.creative.anigen")
        self.model = True
        self.model_name = model_name
        logger.info("AniGen CLI tool ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        self.model = None

    async def generate_rigged(
        self,
        image: bytes,
        output_format: str = "glb",
        ss_model: str = "ckpts/anigen/ss_flow_duet",
        slat_model: str = "ckpts/anigen/slat_flow_auto",
        seed: int = 42,
    ) -> dict[str, bytes]:
        """Generate rigged 3D mesh from image. Returns {'mesh': bytes, 'skeleton': bytes}."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        with tempfile.TemporaryDirectory(prefix="anigen_") as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            input_path.write_bytes(image)

            args = [
                "--image_path", str(input_path),
                "--output_dir", str(output_dir),
                "--ss_flow_path", ss_model,
                "--slat_flow_path", slat_model,
                "--seed", str(seed),
            ]

            self._run_cli(args, timeout=600)

            # AniGen outputs mesh.glb and optionally skeleton data
            result = {}
            mesh_file = output_dir / "input" / "mesh.glb"
            if not mesh_file.exists():
                # Check alternate output structure
                for glb in output_dir.rglob("*.glb"):
                    if "mesh" in glb.name:
                        mesh_file = glb
                        break

            if mesh_file.exists():
                result["mesh"] = mesh_file.read_bytes()
            else:
                raise RuntimeError("AniGen did not produce mesh.glb output")

            # Check for skeleton
            for glb in output_dir.rglob("*.glb"):
                if "skeleton" in glb.name:
                    result["skeleton"] = glb.read_bytes()
                    break

            return result

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        ss_model = form.get("ss_model", "ckpts/anigen/ss_flow_duet")
        slat_model = form.get("slat_model", "ckpts/anigen/slat_flow_auto")

        result = await self.generate_rigged(
            image=image_bytes,
            ss_model=ss_model,
            slat_model=slat_model,
        )
        from starlette.responses import Response
        return Response(
            content=result["mesh"],
            media_type="model/gltf-binary",
        )
