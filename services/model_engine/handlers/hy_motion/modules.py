"""HY-Motion raw nn.Module loading + mmgp setup.

Decomposes T2MRuntime into:
- motion_transformer: HunyuanMotionMMDiT (input_encoder, ctxt_encoder, vtxt_encoder,
  timestep_encoder, double_blocks, single_blocks, final_layer)
- text_encoder: HYTextModel (Qwen3-8B LLM + CLIP sentence encoder)
- body_model: WoodenMesh (vertex template, skin weights, joint hierarchy)

The workspace setup (symlinks for Qwen3-8B, CLIP, stats) is handled here.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

CLIP_PVC_PATH = "/models/image-gen/comfyui/HY-Motion/ckpts/clip-vit-large-patch14"
CLIP_LOCAL_PATH = "/home/user/Documents/models/image-gen/comfyui/HY-Motion/ckpts/clip-vit-large-patch14"
QWEN_PVC_BASE = "/models/motion/{model}/ckpts/Qwen3-8B"
STATS_VENDOR_PATH = "/opt/hymotion/stats"


@dataclass
class HYMotionModules:
    """All raw nn.Modules for HY-Motion inference."""

    motion_transformer: Any   # HunyuanMotionMMDiT
    text_encoder: Any         # HYTextModel (Qwen3-8B + CLIP)
    body_model: Any           # WoodenMesh

    # CFG null features
    null_vtxt_feat: Any = None
    null_ctxt_input: Any = None

    # Pipeline reference — for decode_motion_from_latent() (smoothing, body model)
    pipeline: Any = None

    # Pipeline-level config (samplers, normalization)
    pipeline_cfg: dict = field(default_factory=dict)
    mean: Any = None
    std: Any = None

    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")

    # mmgp
    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_type: str = "hy-motion-1.0",
        disable_prompt_eng: bool = False,
    ) -> HYMotionModules:
        from registry.config import Config
        cfg = Config()

        vendor_path = Path(cfg.project_root) / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        config_yml = model_path / "config.yml"
        ckpt_file = model_path / "latest.ckpt"

        if not config_yml.exists():
            raise FileNotFoundError(f"HY-Motion config.yml not found at {config_yml}")
        if not ckpt_file.exists():
            raise FileNotFoundError(f"HY-Motion checkpoint not found at {ckpt_file}")

        prompter_path = os.environ.get(
            "TEXT2MOTION_PROMPTER_PATH",
            str(Path(cfg.models_root) / "motion" / "text2motion-prompter"),
        )
        if not Path(prompter_path).is_dir():
            disable_prompt_eng = True

        # Set up workspace with symlinks
        workspace = Path(tempfile.mkdtemp(prefix="hymotion_"))
        ckpts_dir = workspace / "ckpts"
        ckpts_dir.mkdir()

        # Resolve Qwen3-8B path: model_path/ckpts/Qwen3-8B, or PVC fallback
        qwen_src = model_path / "ckpts" / "Qwen3-8B"
        if not qwen_src.is_dir():
            qwen_src = Path(QWEN_PVC_BASE.format(model=model_type))
        if not qwen_src.is_dir():
            qwen_src = Path(QWEN_PVC_BASE.format(model="hy-motion-1.0"))
        if qwen_src.is_dir():
            (ckpts_dir / "Qwen3-8B").symlink_to(qwen_src)

        # Resolve CLIP path: local or PVC
        clip_src = Path(CLIP_LOCAL_PATH) if Path(CLIP_LOCAL_PATH).is_dir() else Path(CLIP_PVC_PATH)
        if clip_src.is_dir():
            (ckpts_dir / "clip-vit-large-patch14").symlink_to(clip_src)

        if Path(STATS_VENDOR_PATH).is_dir():
            (workspace / "stats").symlink_to(STATS_VENDOR_PATH)

        # Patch config for absolute mean_std_dir
        with open(config_yml) as f:
            config_dict = yaml.safe_load(f)
        test_cfg = config_dict.get("train_pipeline_args", {}).get("test_cfg", {})
        msd = test_cfg.get("mean_std_dir", "")
        if msd and not os.path.isabs(msd):
            config_dict["train_pipeline_args"]["test_cfg"]["mean_std_dir"] = str(workspace / msd)
        patched_yml = workspace / "config.yml"
        with open(patched_yml, "w") as f:
            yaml.dump(config_dict, f)

        # Extract mean/std from checkpoint to create stats files for the pipeline.
        # The vendor code expects Mean.npy/Std.npy in the mean_std_dir.
        # Without these, it falls back to torch.zeros(1)/torch.ones(1) which
        # causes a shape mismatch when load_state_dict loads the checkpoint's [201]-shaped buffers.
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
            logger.info("Extracted mean/std from checkpoint to %s", stats_dir)

        # Load T2MRuntime to get the pipeline
        from hymotion.utils.t2m_runtime import T2MRuntime

        prev_cwd = os.getcwd()
        os.chdir(str(workspace))
        try:
            runtime = T2MRuntime(
                config_path=str(patched_yml),
                ckpt_name=str(ckpt_file),
                skip_text=False,
                device_ids=None,
                force_cpu=False,
                disable_prompt_engineering=disable_prompt_eng,
                prompt_engineering_model_path=prompter_path if not disable_prompt_eng else None,
            )
        finally:
            os.chdir(prev_cwd)

        # Extract modules from the first pipeline
        pipeline = runtime.pipelines[0]
        motion_transformer = pipeline.motion_transformer
        text_encoder = getattr(pipeline, "text_encoder", None)
        body_model = getattr(pipeline, "body_model", None)
        mean = getattr(pipeline, "mean", None)
        std = getattr(pipeline, "std", None)

        # CFG null features (nn.Parameters on the pipeline)
        null_vtxt_feat = getattr(pipeline, "null_vtxt_feat", None)
        null_ctxt_input = getattr(pipeline, "null_ctxt_input", None)

        # Build pipe dict for mmgp
        pipe = {"motion_transformer": motion_transformer}
        if text_encoder is not None:
            pipe["text_encoder"] = text_encoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vram = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0
        logger.info("HY-Motion loaded: modules=%s VRAM=%.0fMB", list(pipe.keys()), vram)

        return cls(
            motion_transformer=motion_transformer,
            text_encoder=text_encoder,
            body_model=body_model,
            null_vtxt_feat=null_vtxt_feat,
            null_ctxt_input=null_ctxt_input,
            pipeline=pipeline,
            pipeline_cfg=config_dict,
            mean=mean,
            std=std,
            device=device,
            pipe=pipe,
            co_tenants={"motion_transformer": ["text_encoder"]} if text_encoder else {},
        )
