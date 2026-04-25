"""See-Through - Layer decomposition for anime character illustrations.

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
"""

from __future__ import annotations

import io
import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

SEETHROUGH_DIR = "/home/ubuntu/Documents/programs/creative/see-through"


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "runtime_env": {
            "working_dir": SEETHROUGH_DIR,
            "pip": ["torch>=2.1", "torchvision", "transformers>=4.40",
                    "accelerate", "safetensors", "Pillow"],
        },
    },
)
class SeeThroughDeployment(BaseGPUDeployment):
    """See-Through layer decomposition."""

    def _load(self, model_name: str = "see-through") -> None:
        import sys
        sys.path.insert(0, SEETHROUGH_DIR)

        # See-Through loads from its own pretrained models
        logger.info("See-Through loading (stub - needs model code)")
        self.model = True  # placeholder
        self.model_name = model_name

    def _unload(self) -> None:
        self.model = None

    async def decompose(self, image: bytes) -> dict:
        """Decompose image into layers. Returns dict of layer_name -> PNG bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        # See-Through inference
        raise NotImplementedError("See-Through inference TBD - needs model code")

    async def __call__(self, request):
        form = await request.form()
        image_file = form["image"]
        image_bytes = await image_file.read()
        result = await self.decompose(image=image_bytes)
        from starlette.responses import JSONResponse
        return JSONResponse({"layers": list(result.keys())})
