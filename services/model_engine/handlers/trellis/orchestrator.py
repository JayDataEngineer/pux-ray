"""TRELLIS.2 orchestrator — explicit forward() calls on each nn.Module.
 
Inference flow:
1. preprocess_image() — rembg removes background
2. get_cond() — dinov3 extracts features at 512 and 1024
3. sample_sparse_structure() — ss_flow_model + ss_decoder
4. sample_shape_slat() — shape_slat_flow_512/1024
5. decode_shape_slat() — shape_slat_decoder
6. sample_tex_slat() — tex_slat_flow_1024
7. decode_tex_slat() — tex_slat_decoder
8. to_glb() — export mesh
 
Each step calls .forward() directly on the module from TrellisModules.
"""
from __future__ import annotations

import gc
import io
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

MAX_FACES_BEFORE_SIMPLIFY = 2_000_000


class TrellisOrchestrator:
    """Runs TRELLIS inference via direct forward() calls on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def generate(
        self,
        *,
        image: Any = None,
        seed: int = 1,
        steps: int = 12,
        guidance: float = 7.5,
        resolution: str = "1024_cascade",
        decimation: int = 50000,
        texture_size: int = 4096,
    ) -> dict:
        import base64
        import o_voxel
        from PIL import Image

        img_data = image
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data)).convert("RGBA")
        if max(img.size) > 4096:
            scale = 4096 / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)),
                             Image.Resampling.LANCZOS)

        torch.manual_seed(seed)
        device = self.m.device

        with torch.inference_mode():
            if img.mode == "RGBA" and img.getextrema()[3][0] < 255:
                img = self.m.rembg(img)

            cond_512 = self._get_cond(img, 512)
            cond_1024 = self._get_cond(img, 1024)

            coords = self._sample_sparse_structure(cond_512, steps, guidance)

            shape_slat, subs = self._sample_and_decode_shape(
                coords, cond_512, cond_1024, steps, resolution,
            )

            tex_voxels = self._sample_and_decode_texture(
                shape_slat, subs, cond_1024, steps,
            )

            glb = o_voxel.postprocess.to_glb(
                vertices=shape_slat.coords if hasattr(shape_slat, 'coords') else coords,
                faces=None,
                attr_volume=tex_voxels,
                decimation_target=decimation,
                texture_size=texture_size,
                remesh=True, remesh_band=1, remesh_project=0, verbose=False,
            )

            buf = io.BytesIO()
            glb.export(buf, file_type="glb")
            data = buf.getvalue()

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("TRELLIS done: %dKB GLB", len(data) // 1024)
        return {
            "status": "success",
            "data": base64.b64encode(data).decode(),
            "media_type": "model/gltf-binary",
        }

    def _get_cond(self, image, resolution: int) -> dict:
        return {"cond": self.m.dinov3([image], resolution=resolution)}

    def _sample_sparse_structure(self, cond: dict, steps: int, guidance: float) -> torch.Tensor:
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

        device = self.m.device
        batch_size = 1
        in_channels = _get_in_channels(self.m.ss_flow_model)
        res = 32

        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.02)

        noise = torch.randn(batch_size, in_channels, res, res, res, device=device)
        ss_params = self.m.ss_sampler_config.get("params", {})
        result = sampler.sample(
            self.m.ss_flow_model,
            noise,
            steps=steps or ss_params.get("steps", 12),
            guidance_strength=guidance or ss_params.get("guidance_strength", 7.5),
            guidance_rescale=ss_params.get("guidance_rescale", 0.7),
            **cond,
        )

        z_s = result if isinstance(result, torch.Tensor) else result.get("samples", result)
        occupancy = self.m.ss_decoder(z_s)
        coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]]
        return coords

    def _sample_and_decode_shape(self, coords, cond_512, cond_1024, steps, resolution):
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

        device = self.m.device
        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.02)
        shape_params = self.m.shape_sampler_config.get("params", {})

        if resolution in ("512",):
            flow_model = self.m.shape_slat_flow_512
            cond = cond_512
        else:
            flow_model = self.m.shape_slat_flow_1024
            cond = cond_1024

        in_channels = _get_in_channels(flow_model)
        noise_feats = torch.randn(coords.shape[0], in_channels, device=device)

        import trellis2.utils.sparse as sp
        noise_slat = sp.SparseTensor(feats=noise_feats, coords=coords)

        result = sampler.sample(
            flow_model,
            noise_slat,
            steps=steps or shape_params.get("steps", 12),
            guidance_strength=shape_params.get("guidance_strength", 7.5),
            guidance_rescale=shape_params.get("guidance_rescale", 0.5),
            **cond,
        )

        slat = result if isinstance(result, sp.SparseTensor) else result.get("samples", result)

        decoded, subs = self.m.shape_slat_decoder(slat, return_subs=True)
        torch.cuda.empty_cache()

        return decoded, subs

    def _sample_and_decode_texture(self, shape_slat, subs, cond, steps):
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
        import trellis2.utils.sparse as sp

        device = self.m.device
        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.02)
        tex_params = self.m.tex_sampler_config.get("params", {})

        in_channels = _get_in_channels(self.m.tex_slat_flow_1024)
        shape_ch = shape_slat.feats.shape[-1] if hasattr(shape_slat, 'feats') else 8
        tex_ch = max(in_channels - shape_ch, in_channels)

        noise_feats = torch.randn(shape_slat.coords.shape[0], tex_ch, device=device)
        noise_tex = sp.SparseTensor(feats=noise_feats, coords=shape_slat.coords)

        result = sampler.sample(
            self.m.tex_slat_flow_1024,
            noise_tex,
            concat_cond=shape_slat,
            steps=steps or tex_params.get("steps", 12),
            guidance_strength=tex_params.get("guidance_strength", 1.0),
            guidance_rescale=tex_params.get("guidance_rescale", 0.0),
            **cond,
        )

        tex_slat = result if isinstance(result, sp.SparseTensor) else result.get("samples", result)

        tex_voxels = self.m.tex_slat_decoder(tex_slat, guide_subs=subs)
        torch.cuda.empty_cache()

        return tex_voxels


def _get_in_channels(model) -> int:
    for attr in ["in_channels", "input_channels", "num_classes"]:
        val = getattr(model, attr, None)
        if val is not None:
            return val
    if hasattr(model, "config") and hasattr(model.config, "in_channels"):
        return model.config.in_channels
    return 8
