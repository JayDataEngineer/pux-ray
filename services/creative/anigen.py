"""AniGen — Animated 3D asset generation from images (Ray-native).

Generates rigged, skinned 3D meshes (GLB) from single character images.
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

from PIL import Image

from services.base import BaseGPUDeployment, InferenceConfig, _b64_decode

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "FORCE_CUDA": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
    },
)
class AniGenDeployment(BaseGPUDeployment):
    """AniGen image-to-3D via native PyTorch inference."""

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def _load(self, model_name: str = "anigen") -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()

        model_path = registry.get_path("3d", model_name)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"AniGen model not found at {model_path}. "
                f"Check model_registry.yaml '3d.anigen' entry."
            )

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        if self.config.low_resource:
            logger.info("AniGen LOW_RESOURCE mode — fp16, reduced steps")
            self.config.precision = "fp16"

        logger.info("Loading AniGen pipeline from %s", model_path)

        from anigen.pipelines import AnigenImageTo3DPipeline

        ss_flow_path = str(model_path / "ckpts" / "anigen" / "ss_flow_duet")
        slat_flow_path = str(model_path / "ckpts" / "anigen" / "slat_flow_auto")

        self.pipeline = AnigenImageTo3DPipeline.from_pretrained(
            ss_flow_path=ss_flow_path,
            slat_flow_path=slat_flow_path,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(device)

        target_dtype = self.config.dtype()
        if target_dtype != torch.float32:
            for m in self.pipeline.models.values():
                if hasattr(m, "to"):
                    m.to(target_dtype)

        self.model = True
        self.model_name = model_name
        torch.cuda.empty_cache()
        gc.collect()

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("AniGen loaded (precision=%s, low_resource=%s, VRAM=%.0fMB)",
                    self.config.precision, self.config.low_resource, vram)

    def _unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {image_b64, seed}, config}."""
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
            img = None
            seed = 42

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
                img = Image.open(io.BytesIO(img_bytes))
                seed = int(form.get("seed", 42))
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                img_bytes = extracted.get("image")
                if not img_bytes:
                    return JSONResponse(self.handle_error("image_b64 required"), status_code=400)

                img = Image.open(io.BytesIO(img_bytes))
                seed = extracted.get("seed", 42)

            if not self.is_loaded():
                import asyncio
                await asyncio.to_thread(self.load_model, "anigen")

            path = request.url.path
            if path.endswith("/mesh"):
                result = await self._infer_mesh(img, seed)
            else:
                result = await self._infer(img, seed)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                )
            )
        except Exception as e:
            logger.error("AniGen error: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    async def _infer_mesh(self, img, seed: int) -> dict:
        def _run():
            result = self._run_pipeline(img, seed)
            mesh = result.get("mesh")
            if mesh is None:
                return {"data": json.dumps({"error": "No mesh produced"}).encode(), "media_type": "application/json"}

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                mesh.export(tmp.name, file_type="glb")
                data = Path(tmp.name).read_bytes()
                Path(tmp.name).unlink(missing_ok=True)

            logger.info("AniGen mesh done: %dKB GLB", len(data) // 1024)
            return {"data": data, "media_type": "model/gltf-binary"}

        import asyncio
        return await asyncio.to_thread(_run)

    async def _infer(self, img, seed: int) -> dict:
        def _run():
            result = self._run_pipeline(img, seed)
            keys = list(result.keys())
            mesh = result.get("mesh")
            mesh_info = None
            if mesh is not None:
                mesh_info = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}

            return {
                "data": json.dumps({"status": "ok", "seed": seed, "mesh": mesh_info, "keys": keys}).encode(),
                "media_type": "application/json",
            }

        import asyncio
        return await asyncio.to_thread(_run)

    def _run_pipeline(self, img, seed: int) -> dict:
        ss_steps = 25
        slat_steps = 25
        cfg_scale_ss = 7.5
        cfg_scale_slat = 3.0

        if self.config.low_resource:
            ss_steps = 4
            slat_steps = 4

        try:
            with torch.inference_mode():
                return self.pipeline.run(
                    img,
                    seed=seed,
                    cfg_scale_ss=cfg_scale_ss,
                    cfg_scale_slat=cfg_scale_slat,
                    ss_steps=ss_steps,
                    slat_steps=slat_steps,
                    texture_size=512 if self.config.low_resource else 1024,
                )
        except Exception as e:
            logger.error("AniGen inference failed: %s", e, exc_info=True)
            raise