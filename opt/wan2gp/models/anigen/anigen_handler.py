"""AniGen family handler — image-to-rigged-3D with skeleton and skin weights.

Decomposes AniGen pipeline into individual nn.Modules for mmgp VRAM management.
Own inference logic: DSINE normals → DINOv2 conditioning → SS flow → SLAT flow → decode → GLB.

Modules:
  ss_flow_model, ss_decoder, slat_flow_model, slat_decoder,
  image_cond (DINOv2), dsine (normal estimation)
"""
import os
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sage")

import contextlib
import sys

import base64
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _isolated_import(dominant_path, hidden_prefixes=("models",)):
    """Temporarily hide Wan2GP's sys.modules entries so a conflicting package resolves.

    DSINE's hubconf.py does ``from models import dsine`` which collides with
    Wan2GP's ``models`` package.  This context manager briefly removes the
    colliding entries from ``sys.modules`` and ``sys.path`` so that the
    *dominant_path* wins during the managed block, then restores everything
    in the ``finally`` — safe because model loading is single-threaded.
    """
    dominant_path = str(dominant_path)

    # Save sys.modules entries that conflict
    saved_modules = {}
    for prefix in hidden_prefixes:
        for key in list(sys.modules.keys()):
            if key == prefix or key.startswith(prefix + "."):
                saved_modules[key] = sys.modules.pop(key)

    # Save and reorder sys.path: inject dominant, remove wan2gp vendor paths
    saved_path = sys.path.copy()
    wan2gp_paths = [p for p in sys.path
                    if p.endswith("/vendor/wan2gp") or p == "/opt/wan2gp"]
    for p in wan2gp_paths:
        sys.path.remove(p)
    if dominant_path not in sys.path:
        sys.path.insert(0, dominant_path)

    try:
        yield
    finally:
        # Restore sys.path
        sys.path[:] = saved_path

        # Purge anything new that was imported during the managed block
        for key in list(sys.modules.keys()):
            for prefix in hidden_prefixes:
                if key == prefix or key.startswith(prefix + "."):
                    if key not in saved_modules:
                        del sys.modules[key]

        # Restore original entries
        sys.modules.update(saved_modules)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["anigen"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "anigen"

    @staticmethod
    def query_family_infos():
        return {"anigen": (400, "AniGen 3D")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"image_outputs": True, "audio_only": False}

    @staticmethod
    def load_model(
        model_filename, model_type, base_model_type, model_def,
        quantizeTransformer=False, text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0, **kwargs,
    ):
        _src = str(Path(__file__).parent / "_src")
        if _src not in sys.path:
            sys.path.insert(0, _src)

        model_path = Path((model_def or {}).get("anigen_path", ""))
        if not model_path.is_dir():
            raise FileNotFoundError(f"anigen model path not found: {model_path}")

        ckpts_dir = model_path / "ckpts"
        os.environ.setdefault("TORCH_HOME", str(model_path.parent))

        from anigen.pipelines.anigen_image_to_3d import AnigenImageTo3DPipeline

        ss_flow_path = str(ckpts_dir / "ss_flow_solo")
        slat_flow_path = str(ckpts_dir / "slat_flow_control")

        # Search deeper if not found
        if not Path(ss_flow_path).is_dir():
            for p in ckpts_dir.rglob("ss_flow_solo"):
                ss_flow_path = str(p)
                break
        if not Path(ss_flow_path).is_dir():
            for p in ckpts_dir.rglob("ss*"):
                ss_flow_path = str(p)
                break
        if not Path(slat_flow_path).is_dir():
            for p in ckpts_dir.rglob("slat_flow_control"):
                slat_flow_path = str(p)
                break
        if not Path(slat_flow_path).is_dir():
            for p in ckpts_dir.rglob("slat*"):
                slat_flow_path = str(p)
                break

        # DSINE's hubconf.py does `from models import dsine` which conflicts
        # with Wan2GP's `models` package. Use isolated import context.
        dsine_hub_dir = model_path / "hub" / "hugoycj_DSINE-hub_main"
        prev_cwd = os.getcwd()
        os.chdir(str(model_path))
        try:
            with _isolated_import(dsine_hub_dir):
                pipeline = AnigenImageTo3DPipeline.from_pretrained(
                    ss_flow_path=ss_flow_path,
                    slat_flow_path=slat_flow_path,
                    device="cpu",
                )
        finally:
            os.chdir(prev_cwd)

        # Extract nn.Modules for mmgp
        pipe = {
            "ss_flow_model": pipeline.models["ss_flow_model"],
            "ss_decoder": pipeline.models["ss_decoder"],
            "slat_flow_model": pipeline.models["slat_flow_model"],
            "slat_decoder": pipeline.models["slat_decoder"],
        }
        if "image_cond_model" in pipeline.models:
            pipe["image_cond"] = pipeline.models["image_cond_model"]
        if "dsine" in pipeline.models:
            pipe["dsine"] = pipeline.models["dsine"]

        co_tenants = {
            "ss_flow_model": ["ss_decoder"],
            "slat_flow_model": ["slat_decoder"],
        }

        pl = _Pipeline(
            modules=pipe,
            ss_config=pipeline.ss_config,
            slat_config=pipeline.slat_config,
        )
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": ""})


class _Pipeline:
    def __init__(self, modules, ss_config, slat_config):
        self.m = modules
        self.ss_config = ss_config
        self.slat_config = slat_config

    @property
    def device(self):
        return next(self.m["ss_flow_model"].parameters()).device

    @torch.no_grad()
    def generate(self, *, image=None, seed=42, ss_steps=25, slat_steps=25,
                 cfg_scale_ss=7.5, cfg_scale_slat=3.0, simplify_ratio=0.95,
                 **kwargs):
        img_data = image
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = self.device

        # 1. Preprocess: DSINE normal estimation
        from anigen.utils.image_utils import preprocess_image
        img_rgb, img_normal = preprocess_image(img, self.m["dsine"], str(dev))

        # 2. Conditioning: DINOv2 features
        from anigen.utils.image_utils import encode_image
        cond_normal = encode_image(img_normal, self.m["image_cond"], dev)
        cond_rgb = encode_image(img_rgb, self.m["image_cond"], dev)

        normal_tensor = torch.from_numpy(np.array(img_normal)).float() / 255.0
        normal_tensor = normal_tensor.permute(2, 0, 1).unsqueeze(0).to(dev)
        rgb_tensor = torch.from_numpy(np.array(img_rgb.convert("RGB"))).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(dev)

        cond_dict_ss = {"cond": cond_normal, "neg_cond": torch.zeros_like(cond_rgb),
                        "normal": normal_tensor}
        cond_dict_slat = {"cond": cond_rgb, "neg_cond": torch.zeros_like(cond_rgb),
                          "normal": rgb_tensor}

        # 3. SS sampling
        coords, coords_skl = self._sample_ss(cond_dict_ss, cfg_scale_ss, ss_steps)
        del cond_dict_ss
        torch.cuda.empty_cache()

        if coords.shape[0] == 0:
            return {"status": "error", "message": "SS produced no features — try a different image"}

        if coords_skl.shape[0] == 0:
            return {"status": "error", "message": "SS produced no skeleton — try a different image"}

        # 4. SLAT sampling
        slat, slat_skl = self._sample_slat(cond_dict_slat, coords, coords_skl,
                                            cfg_scale_slat, slat_steps)
        del cond_dict_slat
        torch.cuda.empty_cache()

        # 5. Decode
        mesh_result, skeleton_result = self.m["slat_decoder"](slat, slat_skl)
        del slat, slat_skl
        torch.cuda.empty_cache()

        # 6. Post-processing → GLB
        data = self._postprocess_and_export(
            mesh_result, skeleton_result, img_rgb, simplify_ratio)

        return {"status": "success", "data": base64.b64encode(data).decode(),
                "media_type": "model/gltf-binary"}

    def _sample_ss(self, cond_dict_ss, strength, steps):
        from anigen.pipelines import samplers
        from anigen.utils.general_utils import _keep_largest_connected_component_3d

        ss_model = self.m["ss_flow_model"]
        ss_decoder = self.m["ss_decoder"]
        dev = self.device
        reso = ss_model.resolution

        ss_sampler = samplers.AniGenFlowEulerCfgSampler(sigma_min=1e-5)

        noise = torch.randn(1, ss_model.in_channels, reso, reso, reso, device=dev)
        if ss_model.z_is_global:
            noise = torch.randn(1, ss_model.global_token_num, ss_model.in_channels, device=dev)
        noise_skl = torch.randn(1, ss_model.in_channels_skl, reso, reso, reso, device=dev)
        if ss_model.z_skl_is_global:
            noise_skl = torch.randn(1, ss_model.global_token_num_skl, ss_model.in_channels_skl, device=dev)

        out = ss_sampler.sample(
            ss_model, noise, noise_skl,
            **cond_dict_ss,
            steps=steps, cfg_strength=strength, verbose=True,
        )
        z_s, z_s_skl = out.samples, out.samples_skl
        decoded_ss, decoded_ss_skl = ss_decoder(z_s, z_s_skl)

        # Keep largest connected component for skeleton
        bsz, ch, d, h, w = decoded_ss_skl.shape
        for b in range(bsz):
            occ_3d = (decoded_ss_skl[b] > 0).any(dim=0).detach().cpu().numpy()
            if not np.any(occ_3d):
                continue
            mainland_3d = _keep_largest_connected_component_3d(occ_3d)
            mainland_t = torch.from_numpy(mainland_3d).to(device=decoded_ss_skl.device)
            mainland_cd = mainland_t.unsqueeze(0).expand(ch, -1, -1, -1)
            decoded_ss_skl[b] = torch.where(
                mainland_cd, decoded_ss_skl[b],
                torch.full_like(decoded_ss_skl[b], -1e9),
            )

        coords = torch.argwhere(decoded_ss > 0)[:, [0, 2, 3, 4]].int()
        coords_skl = torch.argwhere(decoded_ss_skl > 0)[:, [0, 2, 3, 4]].int()
        return coords, coords_skl

    def _sample_slat(self, cond_dict_slat, coords, coords_skl, strength, steps):
        from anigen.pipelines import samplers
        from anigen.modules import sparse as sp

        slat_model = self.m["slat_flow_model"]
        dev = self.device

        gsn_enabled = False
        gsn_iters = 0
        gsn_alpha = 0.7
        if self.slat_config is not None:
            trainer_args = getattr(getattr(self.slat_config, 'trainer', None), 'args', None)
            if trainer_args is not None:
                gsn_enabled = bool(getattr(trainer_args, 'geodesic_smooth_noise', False))
                gsn_iters = int(getattr(trainer_args, 'geodesic_smooth_noise_iters', 0))
                gsn_alpha = float(getattr(trainer_args, 'geodesic_smooth_noise_alpha', 0.7))

        slat_sampler = samplers.AniGenFlowEulerCfgSampler(
            sigma_min=1e-5,
            geodesic_smooth_noise=gsn_enabled,
            geodesic_smooth_noise_iters=gsn_iters,
            geodesic_smooth_noise_alpha=gsn_alpha,
        )

        noise_slat = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], slat_model.in_channels + slat_model.in_channels_vert_skin, device=dev),
            coords=coords,
        )
        noise_skl = sp.SparseTensor(
            feats=torch.randn(coords_skl.shape[0], slat_model.in_channels_skl, device=dev),
            coords=coords_skl,
        )

        cond = cond_dict_slat.copy()
        use_joint_num_cond = bool(getattr(slat_model, 'use_joint_num_cond', False))
        if use_joint_num_cond:
            cond['joints_num'] = 10
            cond['neg_joints_num'] = 0

        out = slat_sampler.sample(
            slat_model, noise_slat, noise_skl,
            **cond,
            steps=steps, cfg_strength=strength, verbose=True,
        )
        slat, slat_skl = out.samples, out.samples_skl

        # Denormalize
        if self.slat_config is not None:
            norm_stats = getattr(getattr(self.slat_config, 'dataset', None), 'args', None)
            if norm_stats and hasattr(norm_stats, 'normalization'):
                ns = norm_stats.normalization
                if isinstance(ns, dict):
                    for key, tensor_attr in [('slat', slat), ('slat_skl', slat_skl),
                                             ('slat_skel', slat_skl)]:
                        if key in ns:
                            t = tensor_attr if key.startswith('slat_s') else slat
                            mean = torch.tensor(ns[key]['mean'], device=t.device)
                            std = torch.tensor(ns[key]['std'], device=t.device)
                            if key.endswith('_skl') or key.endswith('_skel'):
                                slat_skl = slat_skl.replace(feats=slat_skl.feats * std + mean)
                            else:
                                slat = slat.replace(feats=slat.feats * std + mean)

        return slat, slat_skl

    def _postprocess_and_export(self, mesh_result, skeleton_result, img_rgb,
                                simplify_ratio):
        import trimesh
        from anigen.utils.skin_utils import repair_skeleton_parents, filter_skinning_weights, smooth_skin_weights_on_mesh
        from anigen.utils.postprocessing_utils import postprocess_mesh, parametrize_mesh, barycentric_transfer_attributes, bake_texture
        from anigen.utils.render_utils import render_multiview
        from anigen.utils.export_utils import convert_to_glb_from_data

        joints = skeleton_result.joints_grouped.cpu().numpy()
        parents = skeleton_result.parents_grouped.cpu().numpy().astype(np.int32)
        parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=False).astype(np.int32)

        skin_weights = skeleton_result.skin_pred.cpu().numpy()
        vertex_colors = None
        if hasattr(mesh_result, 'vertex_attrs') and mesh_result.vertex_attrs is not None:
            vertex_colors = mesh_result.vertex_attrs.cpu().numpy()

        orig_vertices = mesh_result.vertices.cpu().numpy()
        orig_faces = mesh_result.faces.cpu().numpy()
        del skeleton_result
        torch.cuda.empty_cache()

        # Simplify
        new_vertices, new_faces = postprocess_mesh(
            orig_vertices, orig_faces,
            simplify=(simplify_ratio > 0), simplify_ratio=simplify_ratio,
            fill_holes=True, verbose=True,
        )

        # Transfer skin weights
        if new_vertices.shape[0] != orig_vertices.shape[0]:
            orig_mesh = trimesh.Trimesh(vertices=orig_vertices, faces=orig_faces, process=False)
            skin_weights = barycentric_transfer_attributes(orig_mesh, skin_weights, new_vertices)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        skin_weights = filter_skinning_weights(mesh, skin_weights, joints, parents)
        skin_weights = smooth_skin_weights_on_mesh(mesh, skin_weights, iterations=100, alpha=1.0)

        # Texture baking
        uv_vertices, uv_faces, uvs, vmapping = parametrize_mesh(new_vertices, new_faces)
        skin_weights = skin_weights[vmapping]

        observations, extrinsics_mv, intrinsics_mv = render_multiview(
            mesh_result, resolution=1024, nviews=100,
        )
        masks = [np.any(obs > 0, axis=-1) for obs in observations]
        extrinsics_np = [e.cpu().numpy() for e in extrinsics_mv]
        intrinsics_np = [i.cpu().numpy() for i in intrinsics_mv]
        del extrinsics_mv, intrinsics_mv, mesh_result
        torch.cuda.empty_cache()

        with torch.enable_grad():
            texture_image = bake_texture(
                uv_vertices, uv_faces, uvs,
                observations, masks, extrinsics_np, intrinsics_np,
                texture_size=1024, mode='opt', lambda_tv=0.01, verbose=True,
            )

        mesh = trimesh.Trimesh(
            vertices=uv_vertices, faces=uv_faces,
            visual=trimesh.visual.TextureVisuals(uv=uvs),
            process=False,
        )

        # Export GLB
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            convert_to_glb_from_data(
                mesh, joints, parents, skin_weights, tmp.name,
                vertex_colors=vertex_colors, texture_image=texture_image,
            )
            data = Path(tmp.name).read_bytes()
            Path(tmp.name).unlink(missing_ok=True)

        return data
