"""Pixal3D family handler — high-fidelity image-to-3D with PBR textures.

Pixal3D is a fine-tune of TRELLIS.2 with projection-mode (pixel-aligned)
conditioning for near-reconstruction-level 3D generation. 5-stage cascade:
SS → Shape LR 512 → Shape HR 1024 → Texture → Decode → GLB.

13 nn.Modules in mmgp pipe:
- ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024, shape_decoder
- tex_slat_flow_512, tex_slat_flow_1024, tex_decoder
- image_cond_ss, image_cond_shape_512, image_cond_shape_1024, image_cond_tex_1024
- rembg (BiRefNet background removal)

Amendment A (Large Model Exception):
  Upstream source: vendor/pixal3d/ (symlinked as models/pixal3d/pixal3d/).
  Origin: https://github.com/TencentARC/Pixal3D — see vendor/pixal3d/ for commit details.
  Uses relative imports from the symlinked subpackage (no sys.path.insert).
  Dependencies: natten, o_voxel, moge (optional, for auto-FOV estimation).
"""
import os
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"

# Patch mmgp safetensors2 to support complex dtypes (C64, C128, C32)
try:
    import mmgp.safetensors2 as _st2
    import torch as _torch
    _st2._map_to_dtype.setdefault("C64", _torch.complex64)
    _st2._map_to_dtype.setdefault("C128", _torch.complex128)
    _st2._map_to_dtype.setdefault("C32", _torch.complex32)
except Exception:
    pass

import base64
import gc
import io
import logging
from pathlib import Path

import numpy as np
import torch
from models._shared import BaseFamilyHandler, resolve_model_path

from PIL import Image

logger = logging.getLogger(__name__)

from models.base_handler import HandlerHooks


class _Pixal3DHooks(HandlerHooks):
    needs_bf16_autocast = False


HANDLER_META = {
    "input_type": "image",
    "output_type": "model3d",
    "hooks": _Pixal3DHooks(),
}

_HANDLER_DIR = Path(__file__).parent

IMAGE_COND_CONFIGS = {
    "ss": {
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


def _find_dinov3_path(model_root: Path) -> str:
    """Resolve local DinoV3 weights from model root or models directory."""
    candidates = [
        model_root / "dinov3" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m",
        model_root / "dinov3",
        Path("/models/3d/pixal3d/dinov3/facebook/dinov3-vitl16-pretrain-lvd1689m"),
        Path("/models/3d/trellis/dinov3/facebook/dinov3-vitl16-pretrain-lvd1689m"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return str(c)
    # Fallback to HuggingFace repo ID
    return "facebook/dinov3-vitl16-pretrain-lvd1689m"



class family_handler(BaseFamilyHandler):
    FAMILY = "pixal3d"
    FAMILY_ID = 404
    DISPLAY_NAME = "Pixal3D"
    SUPPORTED_TYPES = ["pixal3d"]
    AUDIO_ONLY = False
    UI_DEFAULTS = {"steps": 12, "guidance": 7.5}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        model_root = resolve_model_path(
            "pixal3d", "pixal3d_path", model_def,
            spec_module="pipeline_root", category="3d",
            quant=kwargs.get("quant"),
        )

        from .pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

        pipeline_json = _find_pipeline_json(model_root)
        logger.info("Pixal3D: model_root=%s pipeline_json=%s", model_root, pipeline_json)
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(pipeline_json))
        dev = torch.device("cuda")

        from .pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )
        dinov3_path = _find_dinov3_path(model_root)
        image_cond_models = {}
        for stage, config in IMAGE_COND_CONFIGS.items():
            config = {**config, "model_name": dinov3_path}
            cond_model = DinoV3ProjFeatureExtractor(**config)
            cond_model.eval()
            image_cond_models[f"image_cond_{stage}"] = cond_model

        for cond_model in image_cond_models.values():
            if getattr(cond_model, 'use_naf_upsample', False):
                cond_model._load_naf()

        key_map = {
            "ss_flow_model": "sparse_structure_flow_model",
            "ss_decoder": "sparse_structure_decoder",
            "slat_flow_512": "shape_slat_flow_model_512",
            "slat_flow_1024": "shape_slat_flow_model_1024",
            "shape_decoder": "shape_slat_decoder",
            "tex_slat_flow_512": "tex_slat_flow_model_512",
            "tex_slat_flow_1024": "tex_slat_flow_model_1024",
            "tex_decoder": "tex_slat_decoder",
        }

        pipe = {}
        for short_key, full_key in key_map.items():
            if full_key in pipeline.models:
                pipe[short_key] = pipeline.models[full_key]

        pipe.update(image_cond_models)

        if hasattr(pipeline, 'rembg_model') and pipeline.rembg_model is not None:
            pipe['rembg'] = pipeline.rembg_model

        # Do NOT move modules to CUDA — keep on CPU so mmgp can manage VRAM
        # by swapping them just-in-time during forward(). Total weights ~25GB
        # exceeds 24GB VRAM; mmgp co-tenancy handles stage-by-stage loading.
        for k, v in pipe.items():
            if isinstance(v, torch.nn.Module):
                v.eval()
            elif hasattr(v, 'model') and isinstance(v.model, torch.nn.Module):
                v.model.eval()

        co_tenants = {
            "ss_flow_model": ["ss_decoder", "image_cond_ss"],
            "slat_flow_512": ["image_cond_shape_512"],
            "slat_flow_1024": ["image_cond_shape_1024"],
            "tex_slat_flow_1024": ["image_cond_tex_1024"],
        }

        pl = _Pipeline(
            modules=pipe,
            samplers={
                'ss': pipeline.sparse_structure_sampler,
                'shape': pipeline.shape_slat_sampler,
                'tex': pipeline.tex_slat_sampler,
            },
            normalization={
                'shape': pipeline.shape_slat_normalization,
                'tex': pipeline.tex_slat_normalization,
            },
            pbr_layout=getattr(pipeline, 'pbr_attr_layout', None),
            device=dev,
        )
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"steps": 12, "guidance": 7.5})


def _find_pipeline_json(model_root):
    for candidate in [model_root, *model_root.glob("*/ckpts"), *model_root.glob("*/*/ckpts")]:
        if (candidate / "pipeline.json").exists():
            return candidate
    for p in model_root.rglob("pipeline.json"):
        return p.parent
    raise FileNotFoundError(f"pipeline.json not found under {model_root}")


class _Pipeline:
    def __init__(self, modules, samplers, normalization, pbr_layout, device):
        self.m = modules
        self.samplers = samplers
        self.norm = normalization
        self.pbr_layout = pbr_layout
        self.device = device

    def _to_gpu(self, *keys):
        """Move modules to GPU for a stage."""
        for k in keys:
            mod = self.m.get(k)
            if mod is None:
                continue
            target = mod if isinstance(mod, torch.nn.Module) else getattr(mod, 'model', None)
            if target is not None:
                target.to(self.device)

    def _to_cpu(self, *keys):
        """Move modules back to CPU after a stage."""
        for k in keys:
            mod = self.m.get(k)
            if mod is None:
                continue
            target = mod if isinstance(mod, torch.nn.Module) else getattr(mod, 'model', None)
            if target is not None:
                target.cpu()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, *, image=None, image_b64=None, seed=1, steps=12, guidance=7.5,
                 resolution="1024_cascade", camera_angle_x=0.8575,
                 camera_distance=2.0, mesh_scale=1.0,
                 decimation=50000, texture_size=2048, **kwargs):
        from .pixal3d.modules.sparse.basic import SparseTensor

        if image is None and image_b64 is not None:
            image = image_b64
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
        dev = self.device

        img = self._preprocess_image(img)

        # Stage 1: Sparse Structure (proj mode)
        self._to_gpu('ss_flow_model', 'ss_decoder', 'image_cond_ss')
        cond_ss = self._get_proj_cond_ss(
            [img], camera_angle_x, camera_distance, mesh_scale)
        ss_res = 32
        coords = self._sample_sparse_structure(cond_ss, ss_res, steps, guidance)
        del cond_ss
        self._to_cpu('ss_flow_model', 'ss_decoder', 'image_cond_ss')

        # Stage 2: Shape LR 512 (proj mode)
        self._to_gpu('slat_flow_512', 'image_cond_shape_512')
        cond_shape_lr = self._get_proj_cond_shape(
            'image_cond_shape_512', [img], coords,
            camera_angle_x, camera_distance, mesh_scale)
        lr_slat = self._sample_shape_slat(
            cond_shape_lr, 'slat_flow_512', coords, steps, guidance)
        del cond_shape_lr
        self._to_cpu('slat_flow_512', 'image_cond_shape_512')

        # Stage 3: Upsample LR → HR coords
        target_res = 1536 if resolution == '1536_cascade' else 1024
        self._to_gpu('shape_decoder')
        shape_dec = self.m['shape_decoder']
        hr_coords = shape_dec.upsample(lr_slat, upsample_times=4)
        del lr_slat
        self._to_cpu('shape_decoder')

        lr_resolution = 512
        hr_resolution = target_res
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            if coords.shape[0] < 49152 or hr_resolution <= 1024:
                break
            hr_resolution -= 128
        del hr_coords, quant_coords
        torch.cuda.empty_cache()

        # Stage 4: Shape HR (proj mode)
        self._to_gpu('slat_flow_1024', 'image_cond_shape_1024')
        grid_res = hr_resolution // 16
        cond_shape_hr = self._get_proj_cond_shape(
            'image_cond_shape_1024', [img], coords,
            camera_angle_x, camera_distance, mesh_scale,
            grid_resolution_override=grid_res)
        shape_slat = self._sample_shape_slat_hr(
            cond_shape_hr, 'slat_flow_1024', coords, steps, guidance)
        del cond_shape_hr
        self._to_cpu('slat_flow_1024', 'image_cond_shape_1024')

        # Stage 5: Texture (proj mode)
        self._to_gpu('tex_slat_flow_1024', 'image_cond_tex_1024')
        tex_grid_res = hr_resolution // 16
        cond_tex = self._get_proj_cond_shape(
            'image_cond_tex_1024', [img], shape_slat.coords,
            camera_angle_x, camera_distance, mesh_scale,
            grid_resolution_override=tex_grid_res)
        tex_slat = self._sample_tex_slat(
            cond_tex, 'tex_slat_flow_1024', shape_slat, steps)
        del cond_tex
        self._to_cpu('tex_slat_flow_1024', 'image_cond_tex_1024')

        # Stage 6: Decode
        self._to_gpu('shape_decoder', 'tex_decoder')
        out_mesh = self._decode_latent(shape_slat, tex_slat, hr_resolution)
        self._to_cpu('shape_decoder', 'tex_decoder')

        data = b""
        if out_mesh:
            data = self._export_glb(out_mesh[0], hr_resolution)

        torch.cuda.empty_cache()
        gc.collect()
        return {"status": "success", "data": base64.b64encode(data).decode(),
                "media_type": "model/gltf-binary"}

    def _preprocess_image(self, img):
        has_alpha = img.mode == 'RGBA'
        if has_alpha:
            alpha = np.array(img)[:, :, 3]
            has_alpha = not np.all(alpha == 255)

        max_size = max(img.size)
        if max_size > 1024:
            scale = 1024 / max_size
            img = img.resize((int(img.width * scale), int(img.height * scale)),
                             Image.Resampling.LANCZOS)

        if not has_alpha:
            rembg = self.m.get('rembg')
            if rembg is not None:
                target = rembg if isinstance(rembg, torch.nn.Module) else getattr(rembg, 'model', None)
                if target is not None:
                    target.to(self.device)
                img = rembg(img.convert('RGB'))
                if target is not None:
                    target.cpu()
                    torch.cuda.empty_cache()

        if has_alpha or (self.m.get('rembg') and not has_alpha):
            arr = np.array(img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                alpha = arr[:, :, 3]
                bbox = np.argwhere(alpha > 0.8 * 255)
                if len(bbox) > 0:
                    y0, x0 = bbox.min(axis=0)
                    y1, x1 = bbox.max(axis=0) + 1
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    size = max(x1 - x0, y1 - y0)
                    x0 = int(cx - size / 2)
                    y0 = int(cy - size / 2)
                    img = img.crop((x0, y0, x0 + size, y0 + size))
                bg = Image.new("RGB", img.size, (0, 0, 0))
                if img.mode == 'RGBA':
                    bg.paste(img, mask=img.split()[3])
                img = bg

        return img.convert("RGB")

    def _get_proj_cond_ss(self, images, camera_angle_x, distance, mesh_scale):
        cond_model = self.m['image_cond_ss']
        device = self.device
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = cond_model(
            images, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor)
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    def _get_proj_cond_shape(self, cond_key, images, coords,
                             camera_angle_x, distance, mesh_scale,
                             grid_resolution_override=None):
        from .pixal3d.modules.sparse.basic import SparseTensor

        cond_model = self.m[cond_key]
        device = self.device

        orig_grid_res = cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            cond_model.grid_resolution = grid_resolution_override
            cond_model.proj_grid = cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=cond_model.proj_grid.image_resolution,
            ).to(device)

        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = cond_model(
            images, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor)

        B = 1
        grid_res = cond_model.grid_resolution
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        batch_indices = coords[:, 0].long()
        x_coords = coords[:, 1].long()
        y_coords = coords[:, 2].long()
        z_coords = coords[:, 3].long()
        z_proj_sparse = z_proj_grid[batch_indices, x_coords, y_coords, z_coords]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            cond_model.grid_resolution = orig_grid_res
            cond_model.proj_grid = cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=cond_model.proj_grid.image_resolution,
            ).to(device)

        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {
                'global': torch.zeros_like(z_global),
                'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords),
            },
        }

    def _sample_sparse_structure(self, cond, resolution, steps, guidance):
        from .pixal3d.modules.sparse.basic import SparseTensor

        ss_flow = self.m['ss_flow_model']
        reso = ss_flow.resolution
        in_ch = ss_flow.in_channels
        noise = torch.randn(1, in_ch, reso, reso, reso, device=self.device)

        z_s = self.samplers['ss'].sample(
            ss_flow, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling SS (proj)",
        ).samples

        ss_dec = self.m['ss_decoder']
        decoded = ss_dec(z_s) > 0
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def _sample_shape_slat(self, cond, flow_key, coords, steps, guidance):
        from .pixal3d.modules.sparse.basic import SparseTensor

        flow_model = self.m[flow_key]
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        slat = self.samplers['shape'].sample(
            flow_model, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling shape SLat (proj, 512)",
        ).samples
        std = torch.tensor(self.norm['shape']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=slat.device)[None]
        return slat * std + mean

    def _sample_shape_slat_hr(self, cond, flow_key, coords, steps, guidance):
        from .pixal3d.modules.sparse.basic import SparseTensor

        flow_model = self.m[flow_key]
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        slat = self.samplers['shape'].sample(
            flow_model, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling HR shape SLat (proj)",
        ).samples
        std = torch.tensor(self.norm['shape']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=slat.device)[None]
        return slat * std + mean

    def _sample_tex_slat(self, cond, flow_key, shape_slat, steps):
        from .pixal3d.modules.sparse.basic import SparseTensor

        std = torch.tensor(self.norm['shape']['std'], device=shape_slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=shape_slat.device)[None]
        normed_slat = (shape_slat - mean) / std

        flow_model = self.m[flow_key]
        in_ch = flow_model.in_channels
        extra_ch = in_ch - shape_slat.feats.shape[1]
        noise = normed_slat.replace(
            feats=torch.randn(shape_slat.coords.shape[0], extra_ch, device=self.device))

        slat = self.samplers['tex'].sample(
            flow_model, noise,
            concat_cond=normed_slat,
            **cond,
            steps=steps,
            verbose=True, tqdm_desc="Sampling texture SLat (proj)",
        ).samples
        std = torch.tensor(self.norm['tex']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['tex']['mean'], device=slat.device)[None]
        return slat * std + mean

    def _decode_latent(self, shape_slat, tex_slat, resolution):
        from .pixal3d.representations.mesh import MeshWithVoxel

        tex_slat = tex_slat.to('cpu')
        torch.cuda.empty_cache()

        shape_dec = self.m['shape_decoder']
        if shape_dec.dtype != torch.float16:
            shape_dec.convert_to_fp16()
            shape_dec.dtype = torch.float16
        shape_dec.set_resolution(resolution)
        meshes, subs = shape_dec(shape_slat, return_subs=True)
        del shape_slat
        torch.cuda.empty_cache()

        subs = [s.to('cpu') for s in subs]
        for m in meshes:
            m.vertices = m.vertices.cpu()
            m.faces = m.faces.cpu()
        torch.cuda.empty_cache()

        tex_slat = tex_slat.to(self.device)
        tex_dec = self.m['tex_decoder']
        if tex_dec.dtype != torch.float16:
            tex_dec.convert_to_fp16()
            tex_dec.dtype = torch.float16
        subs_gpu = [s.to(self.device) for s in subs]
        tex_voxels = tex_dec(tex_slat, guide_subs=subs_gpu) * 0.5 + 0.5
        tex_voxels._spatial_cache.clear()
        del tex_slat, subs_gpu, subs
        torch.cuda.empty_cache()

        out = []
        for m, v in zip(meshes, tex_voxels):
            m.vertices = m.vertices.cuda()
            m.faces = m.faces.cuda()
            out.append(MeshWithVoxel(
                m.vertices, m.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=v.coords[:, 1:].contiguous(),
                attrs=v.feats.half(),
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                layout=self.pbr_layout,
            ))
        return out

    def _export_glb(self, result, resolution):
        # Try o_voxel for high-quality PBR export
        try:
            import o_voxel.postprocess
            glb = o_voxel.postprocess.to_glb(
                vertices=result.vertices, faces=result.faces,
                attr_volume=result.attrs, coords=result.coords,
                attr_layout=self.pbr_layout, grid_size=resolution,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=200000, texture_size=2048,
                remesh=True, remesh_band=1, remesh_project=0,
            )
            rot = np.array([
                [-1,  0,  0,  0],
                [ 0,  0, -1,  0],
                [ 0, -1,  0,  0],
                [ 0,  0,  0,  1],
            ], dtype=np.float64)
            glb.apply_transform(rot)
            buf = io.BytesIO()
            glb.export(buf, file_type="glb")
            return buf.getvalue()
        except ImportError:
            pass

        # Fallback: trimesh-only export (no PBR textures)
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=result.vertices.cpu().numpy(),
            faces=result.faces.cpu().numpy(),
            process=False,
        )
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        return buf.getvalue()
