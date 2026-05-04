"""See-Through — Layer decomposition for anime character illustrations.

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
Runs inside Ray-managed container (tech-noir/seethrough:latest).

Requires ~4GB VRAM.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class SeeThroughDeployment(BaseGPUDeployment):
    """See-Through layer decomposition via Ray native container."""

    def _load(self, model_name: str = "see-through") -> None:
        self.model_name = model_name
        self.model = True
        logger.info("See-Through ready")

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        resolution = int(form.get("resolution", "1280"))
        inference_steps = int(form.get("inference_steps", "30"))

        result = await asyncio.to_thread(
            self._decompose,
            image_bytes=image_bytes,
            resolution=resolution,
            inference_steps=inference_steps,
        )

        return JSONResponse(result)

    def _decompose(self, image_bytes: bytes, resolution: int, inference_steps: int) -> dict:
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            input_path.write_bytes(image_bytes)

            result = subprocess.run([
                "python", "/opt/seethrough/inference/scripts/inference_psd.py",
                "--srcp", str(input_path),
                "--save_dir", str(output_dir),
                "--resolution", str(resolution),
                "--inference_steps", str(inference_steps),
                "--save_to_psd",
            ], capture_output=True, text=True, timeout=600, cwd="/opt/seethrough")

            if result.returncode != 0:
                raise RuntimeError(f"See-Through failed: {result.stderr}")

            layers = []
            psd_data = None
            for png in sorted(output_dir.rglob("*.png")):
                if "layer" in png.name.lower() or "part" in png.name.lower():
                    layers.append({"name": png.stem})
            for psd in output_dir.rglob("*.psd"):
                psd_data = psd.read_bytes()
                break

            return {
                "layers": layers,
                "has_psd": psd_data is not None,
            }
