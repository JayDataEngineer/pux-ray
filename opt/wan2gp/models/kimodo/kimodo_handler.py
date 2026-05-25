"""Kimodo family handler — text-to-3D motion (NVIDIA, Apache-2.0).

Pipeline: text_encoder.encode → diffusion denoising → motion decode

Follows the trellis/hy_motion pattern:
  - family_handler with static methods (Wan2GP discovery contract)
  - _Pipeline wraps the kimodo model
  - load_model() handles HuggingFace auto-download (gated Llama-3-8B)

Model variants (all from https://github.com/nv-tlabs/kimodo):
  - kimodo-soma-rp:  SOMA skeleton, random prompt dataset
  - kimodo-soma-seed: SOMA skeleton, SEED benchmark
  - kimodo-g1-rp:     Unitree G1 Robot skeleton
  - kimodo-smplx-rp:  SMPL-X skeleton
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Variant → HF model name mapping
VARIANTS = {
    "kimodo-soma-rp": "Kimodo-SOMA-RP-v1.1",
    "kimodo-soma-seed": "Kimodo-SOMA-SEED-v1.1",
    "kimodo-g1-rp": "Kimodo-G1-RP-v1",
    "kimodo-smplx-rp": "Kimodo-SMPLX-RP-v1",
}

HANDLER_META = {
    "input_type": "text",
    "output_type": "motion",
}


class family_handler:
    @staticmethod
    def query_supported_types():
        return list(VARIANTS)

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "kimodo"

    @staticmethod
    def query_family_infos():
        return {"kimodo": (403, "Kimodo Motion")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"image_outputs": False, "audio_only": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN required for Kimodo (gated Meta-Llama-3-8B-Instruct). "
                "Set HF_TOKEN in secrets and deploy."
            )

        resolved = VARIANTS.get(base_model_type)
        if resolved is None:
            raise ValueError(f"Unknown Kimodo variant: {base_model_type}")

        # Run text encoder (Llama-3-8B via LLM2VecEncoder) on CPU.
        # LLM2VecEncoder is NOT an nn.Module so .half() doesn't convert it,
        # leaving it in bfloat16 which causes dtype mismatch errors on RTX 4090.
        # CPU avoids the dtype issue entirely and frees ~14GB VRAM.
        os.environ["TEXT_ENCODER_DEVICE"] = "cpu"

        from kimodo import load_model

        logger.info("Kimodo: loading %s (%s, text_encoder=cpu)...", base_model_type, resolved)
        model = load_model(resolved, device="cuda")
        model.eval()

        pipe = {}
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            pipe['text_encoder'] = model.text_encoder
        if hasattr(model, 'denoiser') and model.denoiser is not None:
            pipe['denoiser'] = model.denoiser

        co_tenants = {}
        if 'text_encoder' in pipe and 'denoiser' in pipe:
            co_tenants = {'text_encoder': ['denoiser']}

        for v in pipe.values():
            if isinstance(v, torch.nn.Module):
                v.eval()

        pl = _Pipeline(model)
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "prompts": "A person walks forward confidently",
            "num_frames": 150,
            "num_denoising_steps": 100,
            "post_processing": False,
        })


class _Pipeline:
    """Wraps the Kimodo model for Wan2GP's generate() → response dict contract."""

    def __init__(self, model):
        self._model = model

    @property
    def device(self):
        p = next(self._model.parameters(), None)
        return p.device if p is not None else torch.device("cuda")

    @torch.inference_mode()
    def generate(self, *, prompts="", num_frames=150, num_denoising_steps=100,
                 seed=None, post_processing=False, cfg_weight=None, **kwargs):
        import base64
        import gc
        import io

        text = prompts or kwargs.get("prompt", kwargs.get("input_prompt", ""))
        if not text:
            raise ValueError("'prompts' (text description) required")

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        gen_kwargs = {
            "prompts": text,
            "num_frames": int(num_frames),
            "num_denoising_steps": int(num_denoising_steps),
            "post_processing": bool(post_processing),
        }
        if cfg_weight is not None:
            gen_kwargs["cfg_weight"] = float(cfg_weight)

        for opt_key in ("cfg_type", "multi_prompt", "num_samples",
                        "num_transition_frames"):
            if opt_key in kwargs:
                gen_kwargs[opt_key] = kwargs[opt_key]

        gen_kwargs["progress_bar"] = lambda x: x

        output = self._model(**gen_kwargs)

        npz_buf = io.BytesIO()
        np_tensors = {}
        posed_joints = None
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                np_tensors[key] = value.detach().cpu().numpy()
                if key == "posed_joints":
                    posed_joints = np_tensors[key]
            elif isinstance(value, np.ndarray):
                np_tensors[key] = value
                if key == "posed_joints":
                    posed_joints = value
        np.savez(npz_buf, **np_tensors)
        npz_bytes = npz_buf.getvalue()
        npz_b64 = base64.b64encode(npz_bytes).decode()

        # MP4 preview for web display
        preview_b64 = ""
        if posed_joints is not None:
            # posed_joints is (B, T, J, 3) or (T, J, 3)
            pj = posed_joints[0] if posed_joints.ndim == 4 else posed_joints
            try:
                from services.motion.preview import render_motion_to_mp4_b64
                preview_b64 = render_motion_to_mp4_b64(pj, fps=30)
            except Exception as e:
                logger.warning("Motion preview render failed: %s", e)

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "status": "success",
            "data": preview_b64 or npz_b64,
            "media_type": "video/mp4" if preview_b64 else "application/x-npz",
            "npz_data": npz_b64,
            "num_frames": gen_kwargs["num_frames"],
            "tensor_keys": list(np_tensors.keys()),
            "tensor_shapes": {k: list(v.shape) for k, v in np_tensors.items()},
        }
