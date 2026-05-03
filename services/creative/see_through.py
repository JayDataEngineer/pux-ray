"""See-Through - Layer decomposition for anime character illustrations.

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
Called via subprocess using See-Through's own venv Python.

Requires ~4GB VRAM. Model loads fresh per subprocess call.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve

from services.base import BaseGPUDeployment, CLIToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01, "num_cpus": 0.5},
)
class SeeThroughDeployment(BaseGPUDeployment, CLIToolMixin):
    """See-Through layer decomposition via subprocess CLI."""

    def _load(self, model_name: str = "see-through") -> None:
        self._init_cli("services.creative.see_through")
        self.model = True
        self.model_name = model_name
        logger.info("See-Through CLI tool ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        self.model = None

    def _ensure_loaded(self) -> None:
        if not hasattr(self, "_venv_python"):
            self._load()

    async def decompose(
        self,
        image: bytes,
        resolution: int = 1280,
        inference_steps: int = 30,
        save_to_psd: bool = True,
    ) -> dict:
        """Decompose image into layers.

        Returns dict with:
          - 'layers': list of layer PNG bytes
          - 'psd': PSD file bytes (if save_to_psd=True)
        """
        self._ensure_loaded()

        with tempfile.TemporaryDirectory(prefix="seethrough_") as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            input_path.write_bytes(image)

            args = [
                "--srcp", str(input_path),
                "--save_dir", str(output_dir),
                "--resolution", str(resolution),
                "--inference_steps", str(inference_steps),
            ]
            if save_to_psd:
                args.append("--save_to_psd")

            self._run_cli(args, timeout=600)

            # Collect output layers
            result = {"layers": []}

            # Find all PNG files in output
            for png in sorted(output_dir.rglob("*.png")):
                if "layer" in png.name.lower() or "part" in png.name.lower():
                    result["layers"].append({
                        "name": png.stem,
                        "data": png.read_bytes(),
                    })

            # Find PSD output
            for psd in output_dir.rglob("*.psd"):
                result["psd"] = psd.read_bytes()
                result["psd_name"] = psd.name
                break

            return result

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        resolution = int(form.get("resolution", "1280"))
        save_to_psd = form.get("save_to_psd", "true").lower() == "true"

        result = await self.decompose(
            image=image_bytes,
            resolution=resolution,
            save_to_psd=save_to_psd,
        )

        from starlette.responses import JSONResponse
        return JSONResponse({
            "layers": [l["name"] for l in result.get("layers", [])],
            "has_psd": "psd" in result,
            "psd_name": result.get("psd_name"),
        })
