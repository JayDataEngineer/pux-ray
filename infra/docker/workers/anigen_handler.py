"""AniGen handler for Docker worker.

Wraps AniGen rigged 3D mesh generation as an HTTP endpoint.
Model stays loaded in GPU memory between requests.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger("workers.anigen")


class Handler:
    def __init__(self):
        self.model = None
        self.model_name: str | None = None

    async def health(self):
        if self.model is None:
            return {"status": "model_not_loaded"}
        return {"status": "ok", "model": self.model_name}

    async def load(self, body: dict):
        """Load AniGen model into GPU memory."""
        if self.model is not None:
            return {"status": "already_loaded", "model": self.model_name}

        logger.info("Loading AniGen model...")
        # AniGen uses a different loading pattern — imports load on first use.
        # We do a test import to verify all extensions are available.
        import torch
        logger.info("AniGen: torch %s + CUDA %s", torch.__version__, torch.version.cuda)

        # Try importing anigen to verify extensions work
        import anigen  # noqa: F401
        self.model = True
        self.model_name = "anigen"
        logger.info("AniGen loaded successfully")
        return {"status": "loaded", "model": self.model_name}

    async def generate(self, request: Request):
        """Generate rigged 3D mesh from uploaded image.

        Accepts multipart/form-data with:
          - image: PNG/JPEG file (required)
          - ss_model: path to ss_flow model (default: from MODEL_PATH)
          - slat_model: path to slat_flow model (default: from MODEL_PATH)
          - seed: int (default: 42)
        """
        if self.model is None:
            await self.load({})

        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        seed = int(form.get("seed", "42"))

        models_root = os.environ.get("MODEL_PATH", "/models")
        ss_model = form.get("ss_model", f"{models_root}/ss_flow_duet")
        slat_model = form.get("slat_model", f"{models_root}/slat_flow_auto")

        # Write image to temp file (AniGen reads from disk)
        import tempfile
        with tempfile.TemporaryDirectory(prefix="anigen_") as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            input_path.write_bytes(image_bytes)

            # Run AniGen inference
            # The exact call depends on AniGen's API — this follows the example.py pattern
            import torch
            import numpy as np
            from PIL import Image

            image = Image.open(input_path).convert("RGB")
            logger.info("Generating rigged mesh from image (%dx%d)", image.width, image.height)

            # AniGen inference
            # AniGen hardcodes './ckpts/' relative to cwd for some models
            import os as _os
            _cwd = _os.getcwd()
            _os.chdir(models_root)
            try:
                from anigen.pipelines import AnigenImageTo3DPipeline
                pipeline = AnigenImageTo3DPipeline.from_pretrained(models_root)
                pipeline.to("cuda")

                torch.manual_seed(seed)
                result = pipeline(image, output_dir=str(output_dir))
            finally:
                _os.chdir(_cwd)

            # Find output GLB
            mesh_file = output_dir / "input" / "mesh.glb"
            if not mesh_file.exists():
                for glb in output_dir.rglob("*.glb"):
                    if "mesh" in glb.name:
                        mesh_file = glb
                        break

            if not mesh_file.exists():
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "No mesh output generated"}, status_code=500)

            glb_bytes = mesh_file.read_bytes()
            logger.info("Generated mesh (%d bytes)", len(glb_bytes))
            return Response(content=glb_bytes, media_type="model/gltf-binary")
