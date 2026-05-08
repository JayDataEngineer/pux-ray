"""See-Through — Layer decomposition for anime illustrations (Ray-native).

Decomposes a single character illustration into body part layers
(body, arms, head, hair, etc.) for sprite animation.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="see_through",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
    },
)
class SeeThroughDeployment(BaseGPUDeployment):
    """See-Through layer decomposition via native PyTorch inference."""

    def __init__(self):
        super().__init__()
        self._pipelines_loaded = False

    def _load(self, model_name: str = "see-through") -> None:
        from services.compat import apply as _apply_compat
        _apply_compat()

        from registry.config import Config

        cfg = Config()
        # See-through uses vendored code at /opt/seethrough, not a model from the PVC
        model_path = Path("/opt/seethrough")
        if not model_path.is_dir():
            # Fallback: check vendor dir in project
            model_path = Path(cfg.project_root) / "vendor" / "seethrough"
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"See-Through code not found at /opt/seethrough or vendor/seethrough"
            )

        # inference/scripts/ contains utils.inference_utils etc
        inf_scripts = str(model_path / "inference" / "scripts")
        if inf_scripts not in sys.path:
            sys.path.insert(0, inf_scripts)

        # Also keep vendor paths for local dev
        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        st_vendor = str(Path(cfg.project_root) / "vendor" / "seethrough")
        if st_vendor not in sys.path:
            sys.path.insert(0, st_vendor)

        if self.config.low_resource:
            logger.info("See-Through LOW_RESOURCE mode — reduced steps, lower resolution")
            self.config.precision = "fp16"

        os.environ["SEETHROUGH_MODEL_DIR"] = str(model_path)

        self._pipelines_loaded = False
        self.model = True
        self.model_name = model_name

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("See-Through ready (pipelines load on first request, VRAM=%.0fMB)", vram)

    def _unload(self) -> None:
        self._pipelines_loaded = False
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {image_b64, resolution, steps}, config}."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "precision": self.config.precision,
                "low_resource": self.config.low_resource,
            }

        start = time.perf_counter()

        try:
            content_type = request.headers.get("content-type", "")

            if "multipart/form-data" in content_type:
                form = await request.form()
                if "config" in form:
                    requested = InferenceConfig(**json.loads(str(form["config"])))
                    if requested != self.config:
                        self.config = requested

                image_file = form.get("image") or form.get("file")
                if not image_file:
                    return JSONResponse(self.handle_error("image file required"), status_code=400)

                img_bytes = await image_file.read()
                resolution = int(form.get("resolution", 1280))
                inference_steps = int(form.get("inference_steps", 30))
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                img_bytes = extracted.get("image")
                if not img_bytes:
                    return JSONResponse(self.handle_error("image_b64 required"), status_code=400)

                resolution = extracted.get("resolution", 1280)
                inference_steps = extracted.get("steps", 30)

            if self.config.low_resource:
                resolution = min(resolution, 768)
                inference_steps = min(inference_steps, 10)

            if not self.is_loaded():
                import asyncio
                await asyncio.to_thread(self.load_model, "see-through")

            result = await self._infer(img_bytes, resolution, inference_steps)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                    extra_metrics={"layer_count": len(result.get("layers", []))},
                )
            )
        except Exception as e:
            logger.error("see_through error: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    async def _infer(self, img_bytes: bytes, resolution: int, inference_steps: int) -> dict:
        def _run():
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
                    return {"data": json.dumps({"error": str(e)}).encode(), "media_type": "application/json", "layers": []}

                layers = []
                psd_data = None
                for png in sorted(output_dir.rglob("*.png")):
                    if "layer" in png.name.lower() or "part" in png.name.lower():
                        layers.append({"name": png.stem})
                for psd in output_dir.rglob("*.psd"):
                    psd_data = psd.read_bytes()
                    break

                if psd_data:
                    metadata = json.dumps({"layers": layers, "has_psd": True}).encode()
                    return {"data": psd_data, "media_type": "image/vnd.adobe.photoshop", "layers": layers}

                return {
                    "data": json.dumps({"layers": layers, "has_psd": False}).encode(),
                    "media_type": "application/json",
                    "layers": layers,
                }

        import asyncio
        return await asyncio.to_thread(_run)