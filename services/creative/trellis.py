"""TRELLIS.2 — Image-to-3D mesh generation (Ray-native).

Generates high-quality 3D meshes (GLB) from single images.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_FACES_BEFORE_SIMPLIFY = 2_000_000


@serve.deployment(
    name="trellis",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.65,expandable_segments:True",
                "OPENCV_IO_ENABLE_OPENEXR": "1",
                "TORCHDYNAMO_DISABLE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
            },
        },
    },
)
class TRELLISDeployment(BaseGPUDeployment):
    """TRELLIS image-to-3D via native PyTorch inference."""
    vram_mb = 10_240

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def _load(self, model_name: str = "trellis") -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("3d", model_name)

        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"TRELLIS model not found at {model_path}. "
                f"Check model_registry.yaml '3d.trellis' entry."
            )

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        if self.config.low_resource:
            logger.info("TRELLIS LOW_RESOURCE mode — fp16, 512 pipeline, minimal steps")
            self.config.precision = "fp16"

        logger.info("Loading TRELLIS pipeline: %s", model_path)

        os.environ["TRELLIS_PIPELINE_ROOT"] = str(model_path)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.quantize import apply_precision

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(device)

        precision = self.config.precision
        if precision != "fp16":
            apply_precision(self.pipeline, precision)

        self.model = True
        self.model_name = model_name
        torch.cuda.empty_cache()
        gc.collect()

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("TRELLIS loaded (precision=%s, low_resource=%s, VRAM=%.0fMB)",
                    precision, self.config.low_resource, vram)

    def _unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {image_b64, seed, steps, guidance, resolution}, config}."""
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
            import asyncio

            content_type = request.headers.get("content-type", "")
            img_bytes = None

            if "multipart/form-data" in content_type:
                form = await request.form()
                if "config" in form:
                    requested = InferenceConfig(**json.loads(str(form["config"])))
                    if requested != self.config:
                        self.config = requested
                        if self.pipeline is not None:
                            from trellis2.quantize import apply_precision
                            apply_precision(self.pipeline, requested.precision)
                            logger.info("TRELLIS precision switched to %s", requested.precision)

                image_file = form.get("image") or form.get("file")
                if not image_file:
                    return JSONResponse(self.handle_error("image file required"), status_code=400)

                img_bytes = await image_file.read()

                if len(img_bytes) > MAX_IMAGE_SIZE:
                    return JSONResponse(self.handle_error("Image too large"), status_code=400)

                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

                if max(img.size) > 4096:
                    scale = 4096 / max(img.size)
                    img = img.resize(
                        (int(img.width * scale), int(img.height * scale)),
                        Image.Resampling.LANCZOS,
                    )

                params = {
                    "seed": int(form.get("seed", 1)),
                    "steps": int(form.get("steps", 12)),
                    "guidance": float(form.get("guidance", 7.5)),
                    "resolution": str(form.get("resolution", "1024_cascade")),
                    "decimation": int(form.get("decimation", 50000)),
                    "texture_size": int(form.get("texture_size", 4096)),
                }
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                img_bytes = extracted.get("image")
                if not img_bytes:
                    return JSONResponse(self.handle_error("image_b64 required"), status_code=400)

                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

                if max(img.size) > 4096:
                    scale = 4096 / max(img.size)
                    img = img.resize(
                        (int(img.width * scale), int(img.height * scale)),
                        Image.Resampling.LANCZOS,
                    )

                params = {
                    "seed": extracted.get("seed", 1),
                    "steps": extracted.get("steps", 12),
                    "guidance": extracted.get("guidance", 7.5),
                    "resolution": extracted.get("resolution", "1024_cascade"),
                    "decimation": extracted.get("decimation", 50000),
                    "texture_size": extracted.get("texture_size", 4096),
                }

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "trellis")

            result = await self._infer(img, **params)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                    extra_metrics={"vertices": result.get("vertices", 0), "faces": result.get("faces", 0)},
                )
            )
        except Exception as e:
            logger.error("TRELLIS error: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    async def _infer(self, img, seed: int, steps: int, guidance: float,
                    resolution: str, decimation: int, texture_size: int) -> dict:
        def _run():
            import o_voxel

            _steps = steps
            _decimation = decimation
            _texture_size = texture_size
            if self.config.low_resource:
                resolution = "512"
                _steps = min(steps, 4)
                _decimation = min(decimation, 10000)
                _texture_size = min(texture_size, 1024)

            logger.info(
                "TRELLIS generate: seed=%d steps=%d res=%s guidance=%.1f low=%s",
                seed, _steps, resolution, guidance, self.config.low_resource,
            )

            ss_params = {
                "steps": _steps,
                "guidance_strength": guidance,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            }
            shape_params = {
                "steps": _steps,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            }
            tex_params = {
                "steps": _steps,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            }

            try:
                with torch.inference_mode():
                    shape_slat, tex_slat, res = self.pipeline.run(
                        img,
                        seed=seed,
                        preprocess_image=True,
                        sparse_structure_sampler_params=ss_params,
                        shape_slat_sampler_params=shape_params,
                        tex_slat_sampler_params=tex_params,
                        pipeline_type=resolution,
                        return_before_decode=True,
                    )

                    mesh = self.pipeline.decode_and_cleanup(shape_slat, tex_slat, res)[0]
                    del shape_slat, tex_slat
                    torch.cuda.empty_cache()

                    if mesh.faces.shape[0] > MAX_FACES_BEFORE_SIMPLIFY:
                        try:
                            mesh.simplify(MAX_FACES_BEFORE_SIMPLIFY)
                        except (AttributeError, ImportError, RuntimeError):
                            pass

                    _verts, _faces, _attrs = mesh.vertices, mesh.faces, mesh.attrs
                    _coords, _layout, _voxel_size = mesh.coords, mesh.layout, mesh.voxel_size
                    del mesh
                    torch.cuda.empty_cache()

                    glb = o_voxel.postprocess.to_glb(
                        vertices=_verts, faces=_faces, attr_volume=_attrs,
                        coords=_coords, attr_layout=_layout, voxel_size=_voxel_size,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=decimation,
                        texture_size=texture_size,
                        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
                    )

                    buf = io.BytesIO()
                    glb.export(buf, file_type="glb")
                    data = buf.getvalue()

                logger.info("TRELLIS done: %dKB GLB", len(data) // 1024)
                return {
                    "data": data,
                    "media_type": "model/gltf-binary",
                    "vertices": len(_verts) if '_verts' in dir() else 0,
                    "faces": len(_faces) if '_faces' in dir() else 0,
                }

            except Exception as e:
                logger.error("TRELLIS inference failed: %s", e, exc_info=True)
                return {"data": json.dumps({"error": str(e)}).encode(), "media_type": "application/json"}

        import asyncio
        return await asyncio.to_thread(_run)