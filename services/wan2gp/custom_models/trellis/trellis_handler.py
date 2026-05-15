"""TRELLIS.2 family handler — image-to-3D mesh with texture.

Follows Wan2GP handler pattern: pipeline object + pipe dict for mmgp.
TRELLIS's pipeline.run() has built-in low_vram VRAM management, so we
return an empty pipe dict — no mmgp interference. The pipeline handles
its own model swapping via .to(device)/.cpu() calls.

The sampler creates float32 timestep tensors that must be cast to match
model weights (bf16 flow models, fp16 decoders). We wrap pipeline.run()
in autocast('cuda', float16) which handles all mixed-precision casts.

Modules: ss_flow, ss_decoder, slat_flow_512, slat_flow_1024,
         tex_slat_flow_512, tex_slat_flow_1024, shape_decoder, tex_decoder, image_cond
"""
import os
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

import base64
import gc
import io
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["trellis"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "trellis"

    @staticmethod
    def query_family_infos():
        return {"trellis": (402, "TRELLIS 3D")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"image_outputs": True, "audio_only": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.models import ModelRegistry
        from registry.config import Config

        cfg = Config()
        model_root = Path(ModelRegistry().get_path("3d", "trellis"))

        import sys
        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        # Find pipeline.json
        pipeline_json = None
        for candidate in [model_root, *model_root.glob("*/ckpts"), *model_root.glob("*/*/ckpts")]:
            if (candidate / "pipeline.json").exists():
                pipeline_json = candidate
                break
        if pipeline_json is None:
            for p in model_root.rglob("pipeline.json"):
                pipeline_json = p.parent
                break
        if pipeline_json is None:
            raise FileNotFoundError(f"pipeline.json not found under {model_root}")

        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        os.environ["TRELLIS_PIPELINE_ROOT"] = str(pipeline_json)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(pipeline_json))

        # Pipeline manages its own VRAM via low_vram mode (default=True).
        # Set device so it creates CUDA tensors for inference.
        pipeline._device = torch.device("cuda")

        # Empty pipe dict — TRELLIS pipeline.run() manages its own VRAM.
        # mmgp hooks would conflict with the pipeline's .to(device)/.cpu() calls.
        pl = _Pipeline(pipeline)
        return pl, {"pipe": {}, "coTenantsMap": {}}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"steps": 12, "guidance": 7.5})


class _Pipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @torch.inference_mode()
    def generate(self, *, image=None, seed=1, steps=12, guidance=7.5,
                 resolution="1024_cascade", decimation=50000,
                 texture_size=4096, **kwargs):
        from PIL import Image

        img_data = image
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        pipeline_type = resolution
        if resolution not in ('512', '1024', '1024_cascade', '1536_cascade'):
            pipeline_type = '1024_cascade'

        sampler_params = {
            'steps': steps,
            'guidance_strength': guidance,
        }

        # Preprocess image outside autocast — BiRefNet outputs bf16 under
        # autocast and ToPILImage() cannot convert bf16 to numpy.
        img = self.pipeline.preprocess_image(img)

        # autocast('cuda', float16) so the sampler's float32 timestep tensors
        # are cast to match model weights. Flow models are bf16, decoders fp16 —
        # autocast handles mixed-precision for all eligible ops.
        # Note: spconv CUDA kernels require fp16 inputs (custom_fwd cast_inputs=fp16).
        with torch.autocast('cuda', dtype=torch.float16):
            results = self.pipeline.run(
                image=img,
                num_samples=1,
                seed=seed,
                sparse_structure_sampler_params=sampler_params,
                shape_slat_sampler_params=sampler_params,
                tex_slat_sampler_params={'steps': steps},
                preprocess_image=False,
                pipeline_type=pipeline_type,
            )

        if not results:
            return {"status": "error", "error": "Pipeline produced no output"}

        result = results[0]
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=result.vertices.cpu().numpy(),
            faces=result.faces.cpu().numpy(),
            process=False,
        )
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        data = buf.getvalue()

        torch.cuda.empty_cache()
        gc.collect()
        return {"status": "success", "data": base64.b64encode(data).decode(),
                "media_type": "model/gltf-binary"}
