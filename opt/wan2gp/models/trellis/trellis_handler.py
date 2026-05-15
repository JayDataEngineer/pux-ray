"""TRELLIS.2 family handler — image-to-3D mesh with texture.

Native Wan2GP model family. Decomposes TRELLIS pipeline into nn.Modules
for mmgp VRAM management. Own inference logic with SageAttention sparse
attention — zero external attention dependencies.

Modules:
  ss_flow_model, ss_decoder, slat_flow_512, slat_flow_1024,
  tex_slat_flow_1024, shape_decoder, tex_decoder, image_cond, rembg
"""
import os
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sage")
os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

import base64
import gc
import io
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

TRELLIS_MODEL_ROOT = os.environ.get("TRELLIS_MODEL_ROOT", "/models/3d/trellis")


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
        import sys

        # Add this handler's directory to sys.path so trellis2 package resolves locally
        handler_dir = Path(__file__).parent
        trellis2_pkg = str(handler_dir / "trellis2")
        if handler_dir not in sys.path:
            sys.path.insert(0, str(handler_dir))
        if trellis2_pkg not in sys.path:
            sys.path.insert(0, trellis2_pkg)

        model_root = Path(TRELLIS_MODEL_ROOT)

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
        dev = torch.device("cuda")
        pipeline.to(dev)

        # Extract individual nn.Modules for mmgp
        key_map = {
            "ss_flow_model": "sparse_structure_flow_model",
            "ss_decoder": "sparse_structure_decoder",
            "slat_flow_512": "shape_slat_flow_model_512",
            "slat_flow_1024": "shape_slat_flow_model_1024",
            "tex_slat_flow_512": "tex_slat_flow_model_512",
            "tex_slat_flow_1024": "tex_slat_flow_model_1024",
            "shape_decoder": "shape_slat_decoder",
            "tex_decoder": "tex_slat_decoder",
        }
        pipe = {}
        for short_key, full_key in key_map.items():
            if full_key in pipeline.models:
                pipe[short_key] = pipeline.models[full_key]

        if hasattr(pipeline, 'image_cond_model') and pipeline.image_cond_model is not None:
            pipe['image_cond'] = pipeline.image_cond_model

        if hasattr(pipeline, 'rembg_model') and pipeline.rembg_model is not None:
            pipe['rembg'] = pipeline.rembg_model

        for k, v in list(pipe.items()):
            if isinstance(v, torch.nn.Module):
                pipe[k] = v.to(dev)
            elif hasattr(v, 'to') and callable(v.to):
                try:
                    v.to(str(dev))
                except Exception:
                    pass

        co_tenants = {
            "ss_flow_model": ["ss_decoder"],
            "slat_flow_512": ["shape_decoder"],
            "slat_flow_1024": ["tex_slat_flow_1024"],
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


class _Pipeline:
    def __init__(self, modules, samplers, normalization, pbr_layout, device):
        self.m = modules
        self.samplers = samplers
        self.norm = normalization
        self.pbr_layout = pbr_layout
        self.device = device

    @torch.inference_mode()
    def generate(self, *, image=None, seed=1, steps=12, guidance=7.5,
                 resolution="1024_cascade", decimation=50000,
                 texture_size=4096, **kwargs):
        from PIL import Image
        import numpy as np
        from trellis2.modules.sparse.basic import SparseTensor

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

        cond_512 = self._get_cond([img], 512)
        cond_1024 = self._get_cond([img], 1024) if resolution != '512' else None

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[resolution]
        coords = self._sample_sparse_structure(cond_512, ss_res, steps, guidance)

        if resolution == '512':
            shape_slat = self._sample_shape_slat(cond_512, 'slat_flow_512', coords, steps, guidance)
            del cond_512
            res = 512
        elif resolution == '1024':
            shape_slat = self._sample_shape_slat(cond_1024, 'slat_flow_1024', coords, steps, guidance)
            del cond_512
            res = 1024
        else:
            target_res = 1536 if resolution == '1536_cascade' else 1024
            shape_slat, res = self._sample_shape_slat_cascade(
                cond_512, cond_1024, 'slat_flow_512', 'slat_flow_1024',
                512, target_res, coords, steps, guidance)
            del cond_512
            shape_slat._spatial_cache = {}
            torch.cuda.empty_cache()

        tex_slat = self._sample_tex_slat(
            cond_1024, 'tex_slat_flow_1024', shape_slat, steps)
        del cond_1024
        torch.cuda.empty_cache()

        shape_slat._spatial_cache = {}
        tex_slat._spatial_cache = {}

        out_mesh = self._decode_latent(shape_slat, tex_slat, res)

        data = b""
        if out_mesh:
            result = out_mesh[0]
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

    def _preprocess_image(self, img):
        import numpy as np
        from PIL import Image

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
                img = rembg(img.convert('RGB'))

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
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg

        return img.convert("RGB")

    def _get_cond(self, images, resolution):
        cond_model = self.m.get('image_cond')
        if cond_model is None:
            raise RuntimeError("image_cond model not loaded")
        cond_model.image_size = resolution
        cond = cond_model(images)
        if cond.dtype == torch.bfloat16:
            cond = cond.half()
        neg_cond = torch.zeros_like(cond)
        return {'cond': cond, 'neg_cond': neg_cond}

    def _sample_sparse_structure(self, cond, resolution, steps, guidance):
        from trellis2.modules.sparse.basic import SparseTensor

        ss_flow = self.m['ss_flow_model']
        reso = ss_flow.resolution
        in_ch = ss_flow.in_channels
        noise = torch.randn(1, in_ch, reso, reso, reso, device=self.device)

        z_s = self.samplers['ss'].sample(
            ss_flow, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling SS",
        ).samples

        ss_dec = self.m['ss_decoder']
        decoded = ss_dec(z_s) > 0

        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def _sample_shape_slat(self, cond, flow_key, coords, steps, guidance):
        from trellis2.modules.sparse.basic import SparseTensor

        flow_model = self.m[flow_key]
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        slat = self.samplers['shape'].sample(
            flow_model, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling shape SLat",
        ).samples

        std = torch.tensor(self.norm['shape']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=slat.device)[None]
        return slat * std + mean

    def _sample_shape_slat_cascade(self, lr_cond, cond, lr_flow_key, hr_flow_key,
                                    lr_res, target_res, coords, steps, guidance,
                                    max_tokens=49152):
        from trellis2.modules.sparse.basic import SparseTensor

        lr_flow = self.m[lr_flow_key]
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], lr_flow.in_channels, device=self.device),
            coords=coords,
        )
        slat = self.samplers['shape'].sample(
            lr_flow, noise,
            **lr_cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        std = torch.tensor(self.norm['shape']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=slat.device)[None]
        slat = slat * std + mean

        shape_dec = self.m['shape_decoder']
        hr_coords = shape_dec.upsample(slat, upsample_times=4)
        del slat
        torch.cuda.empty_cache()

        hr_resolution = target_res
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_res * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            if coords.shape[0] < max_tokens or hr_resolution <= 1024:
                break
            hr_resolution -= 128
        del hr_coords, quant_coords
        torch.cuda.empty_cache()

        hr_flow = self.m[hr_flow_key]
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], hr_flow.in_channels, device=self.device),
            coords=coords,
        )
        slat = self.samplers['shape'].sample(
            hr_flow, noise,
            **cond,
            steps=steps, guidance_strength=guidance,
            verbose=True, tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        std = torch.tensor(self.norm['shape']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['shape']['mean'], device=slat.device)[None]
        slat = slat * std + mean

        return slat, hr_resolution

    def _sample_tex_slat(self, cond, flow_key, shape_slat, steps):
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
            verbose=True, tqdm_desc="Sampling texture SLat",
        ).samples

        std = torch.tensor(self.norm['tex']['std'], device=slat.device)[None]
        mean = torch.tensor(self.norm['tex']['mean'], device=slat.device)[None]
        return slat * std + mean

    def _decode_latent(self, shape_slat, tex_slat, resolution):
        from trellis2.representations.mesh import MeshWithVoxel

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
