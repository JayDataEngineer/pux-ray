"""HY-Motion family handler — text-to-3D motion.

Pipeline: text_encoder.encode → ODE sampling (torchdiffeq.odeint) →
decode_motion_from_latent (smoothing + body model)

nn.Modules:
- motion_transformer: HunyuanMotionMMDiT
- text_encoder: HYTextModel (Qwen3-8B + CLIP)

Workspace: Creates temp dir with symlinks for Qwen3-8B, CLIP, stats.
Patches config.yml with absolute paths.
"""
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from models.base_handler import BaseFamilyHandler, _make_handler_cls

logger = logging.getLogger(__name__)

STATS_VENDOR = "/opt/hymotion/stats"


@contextlib.contextmanager
def _cwd(path):
    """Temporarily change working directory. Restores original on exit."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["hy-motion-1.0", "hy-motion-1.0-lite"]
    FAMILY = "hy_motion"
    FAMILY_INFOS = {"hy_motion": (401, "HY-Motion")}
    MODEL_DEF = {"image_outputs": False, "audio_only": False}
    DEFAULTS = {"prompt": "a person waves hello"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.config import Config
        cfg = Config()
        models_root = Path(cfg.models_root)
        model_path = models_root / "motion" / model_type

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        config_yml = model_path / "config.yml"
        ckpt_file = model_path / "latest.ckpt"
        if not config_yml.exists():
            raise FileNotFoundError(f"HY-Motion config.yml not found at {config_yml}")
        if not ckpt_file.exists():
            raise FileNotFoundError(f"HY-Motion checkpoint not found at {ckpt_file}")

        # Workspace with symlinks
        workspace = Path(tempfile.mkdtemp(prefix="hymotion_"))
        ckpts_dir = workspace / "ckpts"
        ckpts_dir.mkdir()

        qwen_src = model_path / "ckpts" / "Qwen3-8B"
        if not qwen_src.is_dir():
            qwen_src = models_root / "motion" / model_type / "ckpts" / "Qwen3-8B"
        if not qwen_src.is_dir():
            qwen_src = models_root / "motion" / "hy-motion-1.0" / "ckpts" / "Qwen3-8B"
        if qwen_src.is_dir():
            (ckpts_dir / "Qwen3-8B").symlink_to(qwen_src)

        clip_src = model_path / "ckpts" / "clip-vit-large-patch14"
        if not clip_src.is_dir():
            clip_src = models_root / "image-gen" / "comfyui" / "HY-Motion" / "ckpts" / "clip-vit-large-patch14"
        if clip_src.is_dir():
            (ckpts_dir / "clip-vit-large-patch14").symlink_to(clip_src)

        if Path(STATS_VENDOR).is_dir():
            (workspace / "stats").symlink_to(STATS_VENDOR)

        # Patch config with absolute mean_std_dir
        with open(config_yml) as f:
            config_dict = yaml.safe_load(f)
        import os
        test_cfg = config_dict.get("train_pipeline_args", {}).get("test_cfg", {})
        msd = test_cfg.get("mean_std_dir", "")
        if msd and not os.path.isabs(msd):
            config_dict["train_pipeline_args"]["test_cfg"]["mean_std_dir"] = str(workspace / msd)
        patched_yml = workspace / "config.yml"
        with open(patched_yml, "w") as f:
            yaml.dump(config_dict, f)

        # Extract mean/std from checkpoint
        import numpy as np
        ckpt_data = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        ckpt_sd = ckpt_data.get("model_state_dict", ckpt_data)
        stats_dir = workspace / msd.strip("./")
        if "mean" in ckpt_sd or "std" in ckpt_sd:
            stats_dir.mkdir(parents=True, exist_ok=True)
            if "mean" in ckpt_sd:
                np.save(str(stats_dir / "Mean.npy"), ckpt_sd["mean"].numpy())
            if "std" in ckpt_sd:
                np.save(str(stats_dir / "Std.npy"), ckpt_sd["std"].numpy())

        # Load T2MRuntime (requires CWD = workspace for relative path resolution)
        from hymotion.utils.t2m_runtime import T2MRuntime
        with _cwd(workspace):
            runtime = T2MRuntime(
                config_path=str(patched_yml), ckpt_name=str(ckpt_file),
                skip_text=False, device_ids=None, force_cpu=False,
                disable_prompt_engineering=True,
            )

        pipeline = runtime.pipelines[0]
        motion_transformer = pipeline.motion_transformer
        text_encoder = getattr(pipeline, "text_encoder", None)

        pipe = {"motion_transformer": motion_transformer}
        if text_encoder is not None:
            pipe["text_encoder"] = text_encoder
        co_tenants = {"motion_transformer": ["text_encoder"]} if text_encoder else {}

        pl = _Pipeline(pipeline, config_dict)
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}


class _Pipeline:
    def __init__(self, pipeline, config_dict):
        self.pipeline = pipeline
        tp = config_dict.get("train_pipeline_args", {})
        self._train_frames = tp.get("train_frames", 360)
        self._fps = tp.get("output_mesh_fps", 30)
        self._n_steps = tp.get("infer_noise_scheduler_cfg", {}).get("validation_steps", 50)
        self._input_dim = config_dict.get("network_module_args", {}).get("input_dim", 201)

    @property
    def device(self):
        return next(self.pipeline.motion_transformer.parameters()).device

    def generate(self, *, input_prompt="", guidance=3.0, duration=3.0,
                 seeds_csv="42", seed=-1, **kwargs):
        text = input_prompt or kwargs.get("text", "")
        if not text:
            raise ValueError("text required")
        if seed is not None and seed >= 0:
            seeds_csv = str(seed)

        from torchdiffeq import odeint
        from hymotion.pipeline.motion_diffusion import length_to_mask

        with torch.no_grad():
            vtxt_input, ctxt_input, ctxt_length = self.pipeline.text_encoder.encode([text])
            motion_latent = self._sample_motion(vtxt_input, ctxt_input, ctxt_length, duration, guidance)

        output = self.pipeline.decode_motion_from_latent(motion_latent, should_apply_smooothing=True)
        motion_data = {}
        rot6d = None
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                motion_data[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                if k == "rot6d":
                    rot6d = v

        import io, base64, numpy as np
        if rot6d is not None:
            buf = io.BytesIO()
            np.savez_compressed(buf, rot6d=rot6d.cpu().numpy())
            npz_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            npz_b64 = ""

        return {
            "status": "success", "text": text, "duration": duration,
            "cfg_scale": guidance, "motion_data": motion_data,
            "seeds": [int(s.strip()) for s in seeds_csv.split(",") if s.strip()],
            "data": npz_b64,
            "media_type": "application/x-npz",
        }

    def _sample_motion(self, vtxt_input, ctxt_input, ctxt_length, duration, cfg_scale):
        from torchdiffeq import odeint
        from hymotion.pipeline.motion_diffusion import length_to_mask
        device = self.device
        dtype = next(self.pipeline.motion_transformer.parameters()).dtype
        n_frames = min(max(int(duration * self._fps), 1), self._train_frames)
        x_length = torch.tensor([n_frames], device=device, dtype=torch.long)
        x_mask_temporal = length_to_mask(x_length, self._train_frames)
        ctxt_mask_temporal = length_to_mask(ctxt_length.to(device), ctxt_input.shape[1])
        vtxt_input = vtxt_input.to(dtype=dtype)
        ctxt_input = ctxt_input.to(dtype=dtype)

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            null_vtxt = self.pipeline.null_vtxt_feat.expand(*vtxt_input.shape).to(dtype=dtype)
            null_ctxt = self.pipeline.null_ctxt_input.expand(*ctxt_input.shape).to(dtype=dtype)
            vtxt_input = torch.cat([null_vtxt, vtxt_input], dim=0)
            ctxt_input = torch.cat([null_ctxt, ctxt_input], dim=0)
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        def ode_fn(t, x):
            x_input = torch.cat([x] * 2, dim=0) if do_cfg else x
            pred = self.pipeline.motion_transformer(
                x=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal, ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if do_cfg:
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            return pred

        y0 = torch.randn(1, self._train_frames, self._input_dim, device=device, dtype=dtype)
        t = torch.linspace(0, 1, self._n_steps + 1, device=device, dtype=dtype)
        trajectory = odeint(ode_fn, y0, t, method="euler")
        return trajectory[-1][:, :n_frames, ...].clone()
