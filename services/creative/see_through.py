"""See-Through — Layer decomposition for anime illustrations (ForgeService).

Decomposes a character illustration into body part layers (body, arms, head,
hair, etc.) for sprite animation. Uses vendored inference at /opt/seethrough
or vendor/seethrough.

TODO: Wire to model_engine handler once SeeThroughOrchestrator is complete
(currently has a stub return).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


class SeeThroughService(ForgeService):
    """Forge-managed See-Through layer decomposition service."""
    vram_mb = 6_144
    service_name = "see_through"
    default_model = "see-through"

    def __init__(self):
        super().__init__()
        self._pipelines_loaded = False
        self.model_path = None

    def load(self, model_name: str) -> None:
        from services.compat import apply as _apply_compat
        _apply_compat()

        from registry.config import Config

        cfg = Config()
        model_path = Path("/opt/seethrough")
        if not model_path.is_dir():
            model_path = Path(cfg.project_root) / "vendor" / "seethrough"
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"See-Through code not found at /opt/seethrough or vendor/seethrough"
            )

        common_dir = str(model_path / "common")
        if common_dir not in sys.path:
            sys.path.insert(0, common_dir)

        os.environ["SEETHROUGH_MODEL_DIR"] = str(model_path)

        self.model_path = model_path
        self._pipelines_loaded = False
        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("See-Through ready (pipelines load on first request, VRAM=%.0fMB)", vram)

    def unload(self) -> None:
        self._pipelines_loaded = False
        self.model_path = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, payload: dict) -> dict:
        import base64 as _b64

        img_input = payload.get("image")
        if img_input is None:
            return {"status": "error", "error": "image required"}

        if isinstance(img_input, str):
            img_bytes = _b64.b64decode(img_input)
        else:
            img_bytes = img_input

        resolution = int(payload.get("resolution", 1280))
        inference_steps = int(payload.get("steps", 30))

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            input_path.write_bytes(img_bytes)

            try:
                from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
                from utils.torch_utils import seed_everything

                seed_everything(42)

                logger.info("See-Through: running layerdiff (res=%d, steps=%d)...", resolution, inference_steps)
                apply_layerdiff(
                    str(input_path),
                    "layerdifforg/seethroughv0.0.2_layerdiff3d",
                    save_dir=str(output_dir),
                    seed=42,
                    resolution=resolution,
                    num_inference_steps=inference_steps,
                    disable_progressbar=True,
                )

                logger.info("See-Through: running marigold...")
                apply_marigold(
                    str(input_path),
                    "24yearsold/seethroughv0.0.1_marigold",
                    save_dir=str(output_dir),
                    seed=42,
                    resolution=768,
                    disable_progressbar=True,
                )

                srcname = input_path.stem
                saved = output_dir / srcname
                further_extr(str(saved), rotate=False, save_to_psd=True, tblr_split=False)

                self._pipelines_loaded = True

            except Exception as e:
                logger.error("See-Through inference failed: %s", e, exc_info=True)
                return {
                    "status": "error",
                    "error": str(e),
                    "data": None,
                    "media_type": None,
                }

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
                    "data": _b64.b64encode(psd_data).decode(),
                    "media_type": "image/vnd.adobe.photoshop",
                    "layers": layers,
                }

            return {
                "status": "success",
                "data": _b64.b64encode(json.dumps({"layers": layers, "has_psd": False}).encode()).decode(),
                "media_type": "application/json",
                "layers": layers,
            }
