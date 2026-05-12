"""See-Through handler — anime layer decomposition.

LIMITATION: This service cannot be decomposed to raw nn.Modules. The vendor
code uses function-based inference (apply_layerdiff, apply_marigold, further_extr)
that create pipeline objects internally. The functions don't expose nn.Modules
for mmgp management. Kept as a thin wrapper until vendor code is forked.

Architecture: KDiffusionStableDiffusionXLPipeline (layerdiff) + MarigoldDepthPipeline
run sequentially, outputting PSD layers.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "see-through": ModelVariant(
        name="see-through", family="see_through", display_name="See-Through Layer Decomposition",
        vram_estimate_gb=6,
        defaults={"resolution": 1280, "steps": 30},
    ),
}


class SeeThroughHandler(BaseHandler):
    """See-Through handler — thin wrapper pending vendor fork.

    Cannot decompose to nn.Modules. Vendor functions create and manage
    pipeline objects internally without exposing them.
    """

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(f"Unknown See-Through type: {model_type}")
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LoadResult:
        from services.compat import apply
        apply()

        from registry.config import Config
        cfg = Config()

        model_dir = Path("/opt/seethrough")
        if not model_dir.is_dir():
            model_dir = Path(cfg.project_root) / "vendor" / "seethrough"
        if not model_dir.is_dir():
            raise FileNotFoundError("See-Through code not found")

        common_dir = str(model_dir / "common")
        if common_dir not in sys.path:
            sys.path.insert(0, common_dir)

        os.environ["SEETHROUGH_MODEL_DIR"] = str(model_dir)

        return LoadResult(
            pipeline=SeeThroughPipeline(model_dir),
            pipe={},  # No nn.Modules to manage — pipelines load lazily
            co_tenants={},
        )


class SeeThroughPipeline:

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def __call__(self, payload: dict) -> dict:
        return self.generate(payload)

    def generate(self, payload: dict) -> dict:
        import base64

        img_data = payload.get("image")
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        resolution = int(payload.get("resolution", 1280))
        inference_steps = int(payload.get("steps", 30))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            input_path.write_bytes(img_data)

            from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
            from utils.torch_utils import seed_everything

            seed_everything(42)

            apply_layerdiff(
                str(input_path),
                "layerdifforg/seethroughv0.0.2_layerdiff3d",
                save_dir=str(output_dir), seed=42,
                resolution=resolution,
                num_inference_steps=inference_steps,
                disable_progressbar=True,
            )

            apply_marigold(
                str(input_path),
                "24yearsold/seethroughv0.0.1_marigold",
                save_dir=str(output_dir), seed=42,
                resolution=768, disable_progressbar=True,
            )

            srcname = input_path.stem
            saved = output_dir / srcname
            further_extr(str(saved), rotate=False, save_to_psd=True, tblr_split=False)

            layers = []
            psd_data = None
            for png in sorted(output_dir.rglob("*.png")):
                if "layer" in png.name.lower() or "part" in png.name.lower():
                    layers.append({"name": png.stem})
            for psd in output_dir.rglob("*.psd"):
                psd_data = psd.read_bytes()
                break

            if psd_data:
                return {
                    "status": "success",
                    "data": base64.b64encode(psd_data).decode(),
                    "media_type": "image/vnd.adobe.photoshop",
                    "layers": layers,
                }

            return {
                "status": "success",
                "data": base64.b64encode(json.dumps({
                    "layers": layers, "has_psd": False
                }).encode()).decode(),
                "media_type": "application/json",
            }
