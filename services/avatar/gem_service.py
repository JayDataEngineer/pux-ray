"""GEM (GENMO) motion generation — text/audio -> SMPL motion.

Wraps GEM for Forge-compatible inference. Uses Hydra config and the demo
pipeline's data assembly pattern to generate SMPL body parameters.

Output: SMPL params dict with body_pose (L,69), global_orient (L,3),
        transl (L,3), betas (L,10) at 30 FPS.

Note: GEM requires at least one video segment for camera intrinsics.
For text-only generation, we create a dummy reference video segment.
"""
from __future__ import annotations

import builtins
import gc
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

GEM_FPS = 30

# Default camera intrinsics for text-only mode (1280x720)
_REF_WIDTH, _REF_HEIGHT = 1280, 720


class GEMService:
    """Text/audio -> SMPL motion generation via GEM."""

    vram_mb: int = 0  # Self-managed
    service_name: str = "gem"
    default_model: str = "gem_smpl"

    def __init__(self):
        self._loaded = False
        self._model = None
        self._cfg = None
        self._ref_K = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        ckpt_path = self._resolve_ckpt(model_name)
        if not ckpt_path:
            raise FileNotFoundError(f"GEM checkpoint not found: {model_name}")

        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)

        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from gem.utils.net_utils import load_pretrained_model

        # Config dir: check vendor/ first, then opt/ (Docker image)
        gem_root = Path(__file__).parent.parent.parent / "vendor" / "GENMO"
        if not (gem_root / "configs").exists():
            gem_root = Path("/opt/genmo")
        config_dir = str(gem_root / "configs")

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="demo", overrides=[
                "exp=gem_smpl",
                "ckpt_path=null",
                "video_name=demo",
                "model.model_cfg.text_encoder.load_llm=true",
            ])

        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        load_pretrained_model(model, ckpt_path)
        model.cuda().eval()

        # Pre-compute reference camera intrinsics for text-only mode
        from gem.utils.cam_utils import estimate_K
        self._ref_K = estimate_K(_REF_WIDTH, _REF_HEIGHT)  # (3, 3)

        self._model = model
        self._cfg = cfg
        self._loaded = True
        logger.info("GEM: loaded from %s", ckpt_path)

    def unload(self) -> None:
        self._model = None
        self._cfg = None
        self._ref_K = None
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

    def generate(
        self,
        text: str,
        audio_path: str | None = None,
        duration_frames: int = 150,
        fps: int = GEM_FPS,
    ) -> dict:
        """Generate SMPL motion from text (+ optional audio)."""
        if not self._loaded:
            raise RuntimeError("GEM model not loaded")

        t0 = time.time()
        data = self._build_text_data(text, duration_frames)

        with torch.no_grad():
            pred = self._model.predict(data, static_cam=True)

        smpl_params = self._extract_smpl_params(pred)
        gen_ms = (time.time() - t0) * 1000

        return {
            "smpl_params": smpl_params,
            "duration_frames": duration_frames,
            "fps": fps,
            "generation_time_ms": gen_ms,
        }

    def _resolve_ckpt(self, model_name: str) -> str | None:
        from services.avatar import models_root
        for p in [
            Path(models_root()) / "avatar" / "gem" / f"{model_name}.ckpt",
            Path(models_root()) / "avatar" / "gem" / "gem_smpl.ckpt",
        ]:
            if p.exists():
                return str(p)
        return None

    def _build_text_data(self, text: str, duration_frames: int) -> dict:
        """Build the data dict matching GEM's assemble_mixed_data() format.

        Since GEM requires at least one video segment for camera intrinsics,
        we create a minimal video segment (1 frame of zeros) followed by
        the text segment. This matches the demo's create_text_segment() pattern.
        """
        from gem.utils.geo_transform import compute_cam_angvel
        from gem.utils.net_utils import get_valid_mask

        REF_L = 1  # Minimal reference video (1 frame)
        TXT_L = duration_frames
        L = REF_L + TXT_L
        ref_K = self._ref_K  # (3, 3)

        # Reference video segment (1 frame, all zeros except camera)
        R_w2c_ref = torch.eye(3).unsqueeze(0)
        cam_angvel_ref = compute_cam_angvel(R_w2c_ref, padding_last=True)  # (1, 6)

        # Text segment
        R_w2c_txt = torch.eye(3).unsqueeze(0).expand(TXT_L, -1, -1).clone()
        cam_angvel_txt = compute_cam_angvel(R_w2c_txt, padding_last=True)

        # Concatenate ref + text
        data = {
            "kp2d": torch.zeros(L, 17, 3),
            "bbx_xys": torch.cat([
                torch.tensor([[_REF_WIDTH / 2, _REF_HEIGHT / 2, max(_REF_WIDTH, _REF_HEIGHT)]]),
                torch.zeros(TXT_L, 3),
            ]),
            "K_fullimg": ref_K.unsqueeze(0).expand(L, -1, -1).clone(),
            "cam_angvel": torch.cat([cam_angvel_ref, cam_angvel_txt]),
            "cam_tvel": torch.zeros(L, 3),
            "R_w2c": torch.cat([
                R_w2c_ref,
                R_w2c_txt,
            ]),
            "f_imgseq": torch.cat([
                torch.ones(REF_L, 1024),   # Reference frame has "image" features
                torch.zeros(TXT_L, 1024),  # Text frames have none
            ]),
            "has_text": torch.tensor([True]),
            "caption": text,
            "mask": {
                "has_img_mask": torch.cat([
                    torch.ones(REF_L, dtype=torch.bool),   # Ref has image
                    torch.zeros(TXT_L, dtype=torch.bool),  # Text does not
                ]),
                "has_2d_mask": torch.cat([
                    torch.ones(REF_L, dtype=torch.bool),   # Ref has 2D keypoints
                    torch.zeros(TXT_L, dtype=torch.bool),
                ]),
                "has_cam_mask": torch.zeros(L, dtype=torch.bool),  # Static cam
                "has_audio_mask": get_valid_mask(L, 0),
                "has_music_mask": get_valid_mask(L, 0),
            },
            "length": torch.tensor(L),
            "meta": [{
                "mode": "default",
                "multi_text_data": {
                    "vid": ["text0"],
                    "caption": [text],
                    "text_ind": [0],
                    "window_start": torch.tensor([REF_L / L]),
                    "window_end": torch.tensor([1.0]),
                },
            }],
        }
        return data

    def _extract_smpl_params(self, pred: dict) -> dict:
        """Extract SMPL params from GEM prediction, excluding the reference frame."""
        body_params = pred.get("body_params_global", pred.get("body_params_incam", {}))
        # Slice off the reference frame (index 0) to get text-only output
        return {
            "body_pose": body_params["body_pose"][1:],
            "global_orient": body_params["global_orient"][1:],
            "transl": body_params["transl"][1:],
            "betas": body_params.get("betas", torch.zeros_like(body_params["body_pose"][:, :10]))[1:],
        }
