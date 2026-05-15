"""TRELLIS.2 family handler — image-to-3D mesh with texture.

Follows Wan2GP handler pattern: extract nn.Modules → pipe dict → mmgp VRAM.
Modules are converted to float16 for mmgp dtype uniformity.
Pipeline.run() delegates to the pipeline's own inference logic.

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

        # Keep pipeline on CPU — mmgp handles device movement.
        # Set device pointer so pipeline.run() targets CUDA for tensor ops.
        pipeline._device = torch.device("cuda")

        # Extract nn.Modules into pipe dict — same pattern as index_tts2/kokoro
        key_map = {
            "ss_flow": "sparse_structure_flow_model",
            "ss_decoder": "sparse_structure_decoder",
            "slat_flow_512": "shape_slat_flow_model_512",
            "slat_flow_1024": "shape_slat_flow_model_1024",
            "tex_slat_flow_512": "tex_slat_flow_model_512",
            "tex_slat_flow_1024": "tex_slat_flow_model_1024",
            "shape_decoder": "shape_slat_decoder",
            "tex_decoder": "tex_slat_decoder",
        }
        pipe = {}
        for short, full in key_map.items():
            if full in pipeline.models:
                pipe[short] = pipeline.models[full]

        if hasattr(pipeline, 'image_cond_model') and pipeline.image_cond_model is not None:
            pipe['image_cond'] = pipeline.image_cond_model

        # Normalize all modules to float16 for mmgp dtype uniformity.
        # TRELLIS ships mixed bf16/fp16/f32. First bf16→fp16 via pipeline method
        # (also fixes internal self.dtype attrs), then convert remaining f32→fp16.
        pipeline._convert_bf16_to_fp16_if_needed()

        for name in list(pipe.keys()):
            mod = pipe[name]
            if not isinstance(mod, torch.nn.Module):
                continue
            has_non_fp16 = any(p.dtype != torch.float16 for p in mod.parameters())
            if not has_non_fp16:
                continue
            pipe[name] = mod.to(torch.float16)
            # Also handle unregistered tensor attrs (e.g. rope_phases)
            for sub in mod.modules():
                for attr_key, val in list(vars(sub).items()):
                    if isinstance(val, torch.Tensor) and val.dtype != torch.float16:
                        setattr(sub, attr_key, val.to(torch.float16))

        pl = _Pipeline(pipeline)
        return pl, {"pipe": pipe, "coTenantsMap": {}}

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

        results = self.pipeline.run(
            image=img,
            num_samples=1,
            seed=seed,
            sparse_structure_sampler_params=sampler_params,
            shape_slat_sampler_params=sampler_params,
            tex_slat_sampler_params={'steps': steps},
            preprocess_image=True,
            pipeline_type=pipeline_type,
        )

        if not results:
            return {"status": "error", "error": "Pipeline produced no output"}

        data = b""
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
