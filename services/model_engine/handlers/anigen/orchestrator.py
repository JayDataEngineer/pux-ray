"""AniGen orchestrator — explicit forward() calls on decomposed modules.

Inference flow (all FP32):
1. preprocess_image() — dsine.forward() → normal map
2. encode_image() — dinov2.forward() → conditioning features
3. sample_ss() — ss_flow_model in Euler ODE loop (CFG), then ss_decoder
4. sample_slat() — slat_flow_model in Euler ODE loop (CFG), then slat_decoder
5. postprocess() — simplify, UV, texture bake, GLB export
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SIGMA_MIN = 1e-5


class AniGenOrchestrator:
    """AniGen inference via direct forward() calls on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def __call__(self, payload: dict) -> dict:
        return self.generate(payload)

    def generate(self, payload: dict) -> dict:
        import base64
        import io
        import json
        import tempfile
        import sys
        from pathlib import Path
        from registry.config import Config
        cfg = Config()
        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        img_data = payload.get("image")
        if isinstance(img_data, str):
            img_data = base64.b64decode(img_data)
        if not img_data:
            raise ValueError("image required")

        img = Image.open(io.BytesIO(img_data))
        seed = int(payload.get("seed", 42))
        ss_steps = int(payload.get("ss_steps", 25))
        slat_steps = int(payload.get("slat_steps", 25))
        cfg_scale_ss = float(payload.get("cfg_scale_ss", 7.5))
        cfg_scale_slat = float(payload.get("cfg_scale_slat", 3.0))
        texture_size = int(payload.get("texture_size", 1024))
        simplify_ratio = float(payload.get("simplify_ratio", 0.95))

        torch.manual_seed(seed)
        np.random.seed(seed)
        device = self.m.device

        with torch.no_grad():
            # Stage 0: Preprocess image + condition
            from anigen.utils.image_utils import preprocess_image, encode_image

            processed_image, processed_normal = preprocess_image(img, self.m.dsine, str(device))

            cond_rgb = encode_image(processed_image, self.m.dinov2, device)
            cond_normal = encode_image(processed_normal, self.m.dinov2, device)
            neg_cond = torch.zeros_like(cond_rgb)

            # Stage 1: Sparse Structure sampling
            coords, coords_skl = self._sample_ss(
                cond=cond_normal,
                neg_cond=neg_cond,
                cfg_strength=cfg_scale_ss,
                steps=ss_steps,
            )

            # Stage 2: Structured Latent sampling
            slat, slat_skl, _, _ = self._sample_slat(
                cond=cond_rgb,
                neg_cond=neg_cond,
                coords=coords,
                coords_skl=coords_skl,
                cfg_strength=cfg_scale_slat,
                steps=slat_steps,
            )

            coords_cpu = coords.cpu()
            coords_skl_cpu = coords_skl.cpu()
            del coords, coords_skl, cond_rgb, cond_normal, neg_cond

            # Stage 3: Decode SLat → mesh + skeleton
            meshes, skeletons = self.m.slat_decoder(slat, slat_skl)
            mesh_result = meshes[0]
            skeleton_result = skeletons[0]
            del slat, slat_skl
            _cuda_cleanup()

        # Post-processing (CPU)
        from anigen.utils.skin_utils import repair_skeleton_parents, filter_skinning_weights, smooth_skin_weights_on_mesh
        from anigen.utils.export_utils import _extract_vertex_rgb, convert_to_glb_from_data, visualize_skeleton_as_mesh
        from anigen.utils.postprocessing_utils import postprocess_mesh, barycentric_transfer_attributes, parametrize_mesh, bake_texture

        joints = skeleton_result.joints_grouped.cpu().numpy()
        parents = skeleton_result.parents_grouped.cpu().numpy().astype(np.int32)
        parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=False).astype(np.int32)

        skin_weights = skeleton_result.skin_pred.cpu().numpy()
        vertex_colors = _extract_vertex_rgb(getattr(mesh_result, 'vertex_attrs', None))

        orig_vertices = mesh_result.vertices.cpu().numpy()
        orig_faces = mesh_result.faces.cpu().numpy()
        del skeleton_result
        _cuda_cleanup()

        import trimesh
        new_vertices, new_faces = postprocess_mesh(
            orig_vertices, orig_faces,
            simplify=(simplify_ratio > 0),
            simplify_ratio=simplify_ratio,
            fill_holes=True,
            verbose=False,
        )

        if new_vertices.shape[0] != orig_vertices.shape[0]:
            orig_mesh = trimesh.Trimesh(vertices=orig_vertices, faces=orig_faces, process=False)
            skin_weights = barycentric_transfer_attributes(orig_mesh, skin_weights, new_vertices)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        skin_weights = filter_skinning_weights(mesh, skin_weights, joints, parents)
        skin_weights = smooth_skin_weights_on_mesh(mesh, skin_weights, iterations=100, alpha=1.0)

        if payload.get("endpoint", "").endswith("/mesh"):
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                convert_to_glb_from_data(
                    mesh, joints, parents, skin_weights,
                    tmp.name, vertex_colors=vertex_colors,
                )
                data = Path(tmp.name).read_bytes()
                Path(tmp.name).unlink(missing_ok=True)

            return {
                "status": "success",
                "data": base64.b64encode(data).decode(),
                "media_type": "model/gltf-binary",
            }

        return {
            "status": "success",
            "data": base64.b64encode(json.dumps({
                "status": "ok", "seed": seed,
                "mesh": {"vertices": len(mesh.vertices), "faces": len(mesh.faces)},
            }).encode()).decode(),
            "media_type": "application/json",
        }

    # ------------------------------------------------------------------ SS Euler ODE

    def _sample_ss(self, cond, neg_cond, cfg_strength, steps):
        """Sparse Structure: Euler ODE flow matching with CFG on dense 3D volume."""
        model = self.m.ss_flow_model
        decoder = self.m.ss_decoder
        device = self.m.device
        reso = model.resolution

        noise = torch.randn(1, model.in_channels, reso, reso, reso, device=device)
        noise_skl = torch.randn(1, model.in_channels_skl, reso, reso, reso, device=device)

        x_t = noise
        x_skl_t = noise_skl

        t_seq = np.linspace(1, 0, steps + 1)
        for i in range(steps):
            t = t_seq[i]
            t_prev = t_seq[i + 1]
            t_scaled = torch.tensor([1000 * t], device=device)

            # Conditional + unconditional forward
            v, v_skl = model(x_t, x_skl_t, t_scaled, cond)
            v_neg, v_skl_neg = model(x_t, x_skl_t, t_scaled, neg_cond)

            # CFG blend
            v_cfg = (1 + cfg_strength) * v - cfg_strength * v_neg
            v_skl_cfg = (1 + cfg_strength) * v_skl - cfg_strength * v_skl_neg

            # Euler step
            dt = t - t_prev
            x_t = x_t - dt * v_cfg
            x_skl_t = x_skl_t - dt * v_skl_cfg

        # Decode
        decoded_ss, decoded_ss_skl = decoder(x_t, x_skl_t)

        # Filter skeleton to largest connected component
        from anigen.utils.general_utils import _keep_largest_connected_component_3d
        bsz, ch, d, h, w = decoded_ss_skl.shape
        for b in range(bsz):
            occ_3d = (decoded_ss_skl[b] > 0).any(dim=0).detach().cpu().numpy()
            if np.any(occ_3d):
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

    # ------------------------------------------------------------------ SLat Euler ODE

    def _sample_slat(self, cond, neg_cond, coords, coords_skl, cfg_strength, steps):
        """Structured Latent: Euler ODE flow matching with CFG on sparse 3D features."""
        from anigen.modules import sparse as sp

        model = self.m.slat_flow_model
        device = self.m.device

        # Geodesic smooth noise (auto-detect from config)
        gsn_iters = 0
        if self.m.slat_config is not None:
            trainer_args = getattr(getattr(self.m.slat_config, 'trainer', None), 'args', None)
            if trainer_args is not None and bool(getattr(trainer_args, 'geodesic_smooth_noise', False)):
                gsn_iters = int(getattr(trainer_args, 'geodesic_smooth_noise_iters', 0))

        noise_slat = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], model.in_channels + model.in_channels_vert_skin, device=device),
            coords=coords,
        )
        noise_skl = sp.SparseTensor(
            feats=torch.randn(coords_skl.shape[0], model.in_channels_skl, device=device),
            coords=coords_skl,
        )

        if gsn_iters > 0:
            noise_slat = self._geodesic_smooth_noise(noise_slat, coords, gsn_iters)

        x_t = noise_slat
        x_skl_t = noise_skl

        # Joint count conditioning
        use_joint_num = bool(getattr(model, 'use_joint_num_cond', False))
        joints_num_kwargs = {}
        neg_joints_kwargs = {}
        if use_joint_num:
            joints_num_kwargs['joints_num'] = 15
            neg_joints_kwargs['joints_num'] = 0

        t_seq = np.linspace(1, 0, steps + 1)
        for i in range(steps):
            t = t_seq[i]
            t_prev = t_seq[i + 1]
            t_scaled = torch.tensor([1000 * t], device=device)

            # Conditional + unconditional forward
            v, v_skl = model(x_t, x_skl_t, t_scaled, cond, **joints_num_kwargs)
            v_neg, v_skl_neg = model(x_t, x_skl_t, t_scaled, neg_cond, **neg_joints_kwargs)

            # CFG blend on sparse features
            v_cfg_feats = (1 + cfg_strength) * v.feats - cfg_strength * v_neg.feats
            v_skl_cfg_feats = (1 + cfg_strength) * v_skl.feats - cfg_strength * v_skl_neg.feats

            # Euler step (elementwise on feats)
            dt = t - t_prev
            x_t = x_t.replace(x_t.feats - dt * v_cfg_feats)
            x_skl_t = x_skl_t.replace(x_skl_t.feats - dt * v_skl_cfg_feats)

        # Denormalize from config stats
        slat = x_t
        slat_skl = x_skl_t

        if self.m.slat_config is not None:
            norm = _get_norm_stats(self.m.slat_config)
            slat = self._denorm_sparse(slat, norm, 'slat')
            slat_skl = self._denorm_sparse(slat_skl, norm, 'slat_skl')

        return slat, slat_skl, slat, slat_skl

    def _denorm_sparse(self, tensor, norm_stats, key):
        """Denormalize sparse tensor features using config stats."""
        if norm_stats is None:
            return tensor
        mean_k = key if key in norm_stats else ('slat_skel' if key == 'slat_skl' and 'slat_skel' in norm_stats else None)
        if mean_k is None and 'mean' in norm_stats:
            mean_k = 'mean'
        if mean_k and isinstance(norm_stats.get(mean_k), dict):
            mean = torch.tensor(norm_stats[mean_k]['mean'], device=tensor.device)
            std = torch.tensor(norm_stats[mean_k]['std'], device=tensor.device)
            return tensor.replace(tensor.feats * std + mean)
        return tensor

    def _geodesic_smooth_noise(self, noise_slat, coords, iters, alpha=0.7):
        """Apply geodesic smooth noise to skin branch of SLat noise.

        Smooths the last in_channels_vert_skin channels using 6-neighbor voxel adjacency.
        """
        model = self.m.slat_flow_model
        vert_skin_c = model.in_channels_vert_skin
        if vert_skin_c <= 0:
            return noise_slat

        feats = noise_slat.feats.clone()
        skin_feats = feats[:, -vert_skin_c:]
        coords_t = coords[:, 1:].long()

        coord_to_idx = {}
        for idx, c in enumerate(coords_t):
            coord_to_idx[tuple(c.tolist())] = idx

        neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        for _ in range(iters):
            new_skin = skin_feats.clone()
            for i, c in enumerate(coords_t):
                c_tuple = tuple(c.tolist())
                neighbor_idxs = [
                    coord_to_idx[(c_tuple[0] + dx, c_tuple[1] + dy, c_tuple[2] + dz)]
                    for dx, dy, dz in neighbors_6
                    if (c_tuple[0] + dx, c_tuple[1] + dy, c_tuple[2] + dz) in coord_to_idx
                ]
                if neighbor_idxs:
                    neighbor_mean = skin_feats[neighbor_idxs].mean(dim=0)
                    new_skin[i] = (1 - alpha) * skin_feats[i] + alpha * neighbor_mean

            skin_feats = new_skin

        # Re-normalize variance
        skin_std = skin_feats.std(dim=0, keepdim=True).clamp_min(0.1)
        skin_feats = skin_feats / skin_std * skin_feats.std(dim=0, unbiased=False).clamp_min(0.1)

        feats[:, -vert_skin_c:] = skin_feats
        return noise_slat.replace(feats)


def _get_norm_stats(config):
    """Extract normalization stats from slat_config."""
    try:
        return config.dataset.args.normalization
    except (AttributeError, KeyError):
        return None


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
