"""SOMA body model — SMPL params -> SOMA 77-joint poses + mesh rendering.

SOMA takes posed SMPL vertices (not raw parameters). This service runs
the SMPL forward pass first, then feeds vertices to PoseInversion for
the SMPL->SOMA conversion. The 1279 FPS claim is for analytical-only mode.

SOMA stays resident in VRAM (<1GB) while GEM and FluxRT are staged.
"""
from __future__ import annotations

import gc
import logging
import os
import time

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SOMAService:
    """SMPL params -> SOMA 77-joint poses + skeleton rendering."""

    vram_mb: int = 0  # <1GB, stays resident
    service_name: str = "soma"
    default_model: str = "soma_smpl"

    def __init__(self):
        self._loaded = False
        self._soma_layer = None
        self._pose_inv = None
        self._smpl_model = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        from soma.soma import SOMALayer
        from soma.pose_inversion import PoseInversion
        from services.avatar import models_root

        smpl_path = os.path.join(models_root(), "avatar", "gem", "body_models")

        # Load SOMA layer with SMPL identity model
        self._soma_layer = SOMALayer(
            identity_model_type="smpl",
            identity_model_kwargs={"model_path": smpl_path},
            device="cuda",
            mode="warp",
        )

        # Create pose inverter (analytical mode for speed)
        self._pose_inv = PoseInversion(self._soma_layer, low_lod=True)

        # Load SMPL model for forward pass (vertices from params)
        import smplx
        self._smpl_model = smplx.create(
            model_type="smpl",
            model_path=smpl_path,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to("cuda").eval()

        self._loaded = True
        logger.info("SOMA: loaded (py-soma-x, analytical mode)")

    def unload(self) -> None:
        self._soma_layer = None
        self._pose_inv = None
        self._smpl_model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._loaded

    def actual_vram_mb(self) -> int:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))

    def convert_smpl_to_soma(self, smpl_params: dict) -> dict:
        """Convert SMPL params to SOMA 77-joint poses.

        Steps:
          1. SMPL forward pass -> vertices
          2. Prepare identity from betas
          3. PoseInversion.fit(vertices) -> SOMA rotations + root translation

        Args:
            smpl_params: Dict with body_pose (L,69), global_orient (L,3),
                         transl (L,3), betas (L,10).

        Returns:
            Dict with soma_rotations (L,77,3,3), root_translation (L,3),
                         vertices (L,V,3), conversion_time_ms.
        """
        if not self._loaded:
            raise RuntimeError("SOMA not loaded")

        t0 = time.time()

        body_pose = smpl_params["body_pose"]
        global_orient = smpl_params["global_orient"]
        transl = smpl_params.get("transl", torch.zeros_like(global_orient))
        betas = smpl_params.get(
            "betas",
            torch.zeros(body_pose.shape[0], 10, device=body_pose.device),
        )

        # Ensure tensors on CUDA
        device = torch.device("cuda")
        for name in ("body_pose", "global_orient", "transl", "betas"):
            t = locals()[name]
            if isinstance(t, torch.Tensor) and t.device != device:
                locals()[name]  # noqa — just for the check

        # Step 1: SMPL forward pass to get vertices
        with torch.no_grad():
            smpl_out = self._smpl_model(
                body_pose=body_pose.to(device),
                global_orient=global_orient.to(device),
                transl=transl.to(device),
                betas=betas.to(device),
            )
        vertices = smpl_out.vertices  # (L, V, 3)

        # Step 2: Prepare identity once from first frame's betas
        self._pose_inv.prepare_identity(betas[:1].to(device))

        # Step 3: Pose inversion — batch through for memory efficiency
        soma_rotations = []
        root_translations = []
        batch_size = 32
        for i in range(0, len(vertices), batch_size):
            batch_verts = vertices[i:i + batch_size]
            with torch.no_grad():
                result = self._pose_inv.fit(
                    batch_verts,
                    body_iters=2,
                    finger_iters=0,
                    full_iters=1,
                    autograd_iters=0,
                )
            soma_rotations.append(result["rotations"].cpu())
            root_translations.append(result["root_translation"].cpu())

        soma_rotations = torch.cat(soma_rotations, dim=0)  # (L, 77, 3, 3)
        root_translations = torch.cat(root_translations, dim=0)  # (L, 3)
        vertices_cpu = vertices.cpu()
        conv_ms = (time.time() - t0) * 1000

        return {
            "soma_rotations": soma_rotations,
            "root_translation": root_translations,
            "vertices": vertices_cpu,
            "conversion_time_ms": conv_ms,
            "num_frames": len(soma_rotations),
        }

    def render_skeleton(
        self,
        vertices: torch.Tensor | np.ndarray,
        resolution: tuple[int, int] = (512, 512),
    ) -> list[np.ndarray]:
        """Render skeleton images from SMPL vertices for rendering conditioning.

        Uses orthographic frontal projection. Draws bones + joints on black bg.
        """
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()

        H, W = resolution
        frames = []
        skeleton_pairs = self._skeleton_pairs()

        for frame_verts in vertices:
            img = np.zeros((H, W, 3), dtype=np.uint8)

            # Orthographic frontal projection (XY plane)
            xy = frame_verts[:, :2]
            xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
            extent = max((xy_max - xy_min).max(), 1e-6)
            xy_norm = (xy - xy_min) / extent
            margin = 0.1
            xy_px = (xy_norm * [W * (1 - 2 * margin), H * (1 - 2 * margin)]
                     + [W * margin, H * margin]).astype(int)

            for i, j in skeleton_pairs:
                if i < len(xy_px) and j < len(xy_px):
                    cv2.line(img, tuple(xy_px[i]), tuple(xy_px[j]),
                             (0, 255, 0), 2, cv2.LINE_AA)

            for pt in xy_px[::3]:  # Draw every 3rd joint to avoid clutter
                cv2.circle(img, tuple(pt), 3, (0, 200, 255), -1, cv2.LINE_AA)

            frames.append(img)

        return frames

    @staticmethod
    def _skeleton_pairs() -> list[tuple[int, int]]:
        """SMPL skeleton connectivity for visualization (23 joints)."""
        return [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (4, 5), (5, 6),
            (2, 7), (7, 8), (8, 9),
            (3, 10), (10, 11), (11, 12),
            (9, 13), (9, 14),
            (13, 16), (14, 17),
            (16, 18), (17, 19),
            (18, 20), (19, 21),
            (20, 22), (21, 23),
        ]
