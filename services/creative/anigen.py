"""AniGen — Animated 3D asset generation from images.

Generates rigged, skinned 3D meshes (GLB) from single character images.
Runs inside Ray-managed container (tech-noir/anigen:latest).

Pipeline imports directly — no subprocess or HTTP layer needed.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
from ray import serve
from starlette.responses import Response

from services.base import BaseGPUDeployment, _free_cuda_cache

logger = logging.getLogger(__name__)

SS_MODEL = os.environ.get("ANIGEN_SS_MODEL", "ckpts/anigen/ss_flow_duet")
SLAT_MODEL = os.environ.get("ANIGEN_SLAT_MODEL", "ckpts/anigen/slat_flow_auto")


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class AniGenDeployment(BaseGPUDeployment):
    """AniGen animated 3D asset generation via Ray native container."""

    def _load(self, model_name: str = "anigen") -> None:
        os.environ.setdefault("FORCE_CUDA", "1")
        sys.path.insert(0, "/opt/anigen")

        from anigen.pipelines import AnigenImageTo3DPipeline

        self.pipeline = AnigenImageTo3DPipeline.from_pretrained(
            ss_model=SS_MODEL,
            slat_model=SLAT_MODEL,
        )
        self.pipeline.cuda()
        self.model_name = model_name
        self.model = True
        logger.info("AniGen loaded: ss=%s, slat=%s", SS_MODEL, SLAT_MODEL)

    def _unload(self) -> None:
        self.pipeline = None
        self.model = None
        _free_cuda_cache()

    async def __call__(self, request):
        from PIL import Image

        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        seed = int(form.get("seed", "42"))

        img = Image.open(io.BytesIO(image_bytes))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        result = self.pipeline.run(img)

        for name, data in result.items():
            if hasattr(data, "export"):
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                    data.export(tmp.name)
                    glb_bytes = Path(tmp.name).read_bytes()
                    Path(tmp.name).unlink(missing_ok=True)
                return Response(content=glb_bytes, media_type="model/gltf-binary")

        from starlette.responses import JSONResponse
        return JSONResponse({"error": "No mesh in output"}, status_code=500)
