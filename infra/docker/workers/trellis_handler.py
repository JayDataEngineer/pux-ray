"""TRELLIS.2 handler for Docker worker.

Wraps TRELLIS.2 image-to-3D pipeline as an HTTP endpoint.
Model stays loaded in GPU memory between requests (no per-call reload).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger("workers.trellis")

# Set env vars before importing torch/trellis
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Handler:
    def __init__(self):
        self.pipeline = None
        self.model_name: str | None = None

    async def health(self):
        if self.pipeline is None:
            return {"status": "model_not_loaded"}
        return {"status": "ok", "model": self.model_name}

    async def load(self, body: dict):
        """Load TRELLIS model into GPU memory."""
        if self.pipeline is not None:
            return {"status": "already_loaded", "model": self.model_name}

        model_name = body.get("model", os.environ.get("MODEL_NAME", "microsoft/TRELLIS.2-4B"))

        # Model path: if local, check /models mount first
        model_path = os.environ.get("MODEL_PATH", "")
        if model_path and Path(model_path).exists():
            model_name = model_path

        logger.info("Loading TRELLIS pipeline: %s", model_name)

        import o_voxel
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_name)
        self.pipeline.cuda()
        self.model_name = model_name

        logger.info("TRELLIS loaded successfully")
        return {"status": "loaded", "model": self.model_name}

    async def generate(self, request: Request):
        """Generate 3D mesh from uploaded image.

        Accepts multipart/form-data with:
          - image: PNG/JPEG file (required)
          - output_format: "glb" or "obj" (default: "glb")
          - resolution: int (default: 512)
          - decimation: int (default: 1000000)
        """
        if self.pipeline is None:
            await self.load({})

        import o_voxel
        from PIL import Image

        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        output_format = form.get("output_format", "glb")
        resolution = int(form.get("resolution", "512"))
        decimation = int(form.get("decimation", "1000000"))

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        texture_size = {256: 2048, 512: 4096, 1024: 4096, 2048: 8192}.get(resolution, 4096)

        logger.info("Generating 3D mesh (res=%d, decimation=%d)", resolution, decimation)
        mesh = self.pipeline.run(image)[0]
        mesh.simplify(16_777_216)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        buf = io.BytesIO()
        glb.export(buf, extension_webp=True)
        buf.seek(0)

        logger.info("Generated %s (%d bytes)", output_format, len(buf.getvalue()))
        return Response(
            content=buf.getvalue(),
            media_type="model/gltf-binary" if output_format == "glb" else "application/octet-stream",
        )
