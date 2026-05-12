"""HY-Motion orchestrator — explicit forward() calls on decomposed modules.

Inference flow:
1. encode_text() — text_encoder.encode() → vtxt_input, ctxt_input, ctxt_length
2. sample_motion() — motion_transformer.forward() in ODE loop (torchdiffeq)
3. decode_motion() — body_model.forward() → vertices, keypoints3d
4. smooth() — Savitzky-Golay + SLERP post-processing
"""
from __future__ import annotations

import logging

import torch
from torchdiffeq import odeint
from hymotion.pipeline.motion_diffusion import length_to_mask

logger = logging.getLogger(__name__)


class HYMotionOrchestrator:
    """HY-Motion inference via direct forward() calls."""

    def __init__(self, modules):
        self.m = modules
        # Cache config values to avoid repeated dict lookups
        tp = modules.pipeline_cfg.get("train_pipeline_args", {})
        self._train_frames = tp.get("train_frames", 360)
        self._fps = tp.get("output_mesh_fps", 30)
        self._n_steps = tp.get("infer_noise_scheduler_cfg", {}).get("validation_steps", 50)
        self._input_dim = modules.pipeline_cfg.get("network_module_args", {}).get("input_dim", 201)

    def __call__(self, payload: dict) -> dict:
        return self.generate(payload)

    def generate(self, payload: dict) -> dict:
        text = payload.get("text", "")
        if not text:
            raise ValueError("text required")

        seeds_csv = str(payload.get("seed", payload.get("seeds_csv", "42")))
        duration = float(payload.get("duration", 3.0))
        cfg_scale = float(payload.get("cfg_scale", payload.get("guidance", 3.0)))

        logger.info("HY-Motion: text=%r dur=%.1fs cfg=%.1f", text[:80], duration, cfg_scale)

        with torch.no_grad():
            # 1. Text encoding (outputs already on GPU from T2MRuntime)
            vtxt_input, ctxt_input, ctxt_length = self.m.text_encoder.encode([text])

            # 2. Motion denoising via ODE
            motion_latent = self._sample_motion(
                vtxt_input, ctxt_input, ctxt_length, duration, cfg_scale,
            )

        # 3. Motion decoding
        output = self._decode_motion(motion_latent)

        motion_data = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                motion_data[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}

        return {
            "status": "success",
            "text": text,
            "duration": duration,
            "cfg_scale": cfg_scale,
            "seeds": [int(s.strip()) for s in seeds_csv.split(",") if s.strip()],
            "motion_data": motion_data,
        }

    def _sample_motion(self, vtxt_input, ctxt_input, ctxt_length, duration, cfg_scale):
        """ODE-based flow matching with CFG, matching vendor's torchdiffeq approach."""
        device = self.m.device

        n_frames = min(max(int(duration * self._fps), 1), self._train_frames)

        # Build masks
        x_length = torch.tensor([n_frames], device=device, dtype=torch.long)
        x_mask_temporal = length_to_mask(x_length, self._train_frames)
        ctxt_mask_temporal = length_to_mask(ctxt_length.to(device), ctxt_input.shape[1])

        do_cfg = cfg_scale > 1.0

        # CFG: batch uncond + cond for single forward pass
        if do_cfg:
            null_vtxt = self.m.null_vtxt_feat.expand(*vtxt_input.shape)
            null_ctxt = self.m.null_ctxt_input.expand(*ctxt_input.shape)
            vtxt_input = torch.cat([null_vtxt, vtxt_input], dim=0)
            ctxt_input = torch.cat([null_ctxt, ctxt_input], dim=0)
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        def ode_fn(t, x):
            x_input = torch.cat([x] * 2, dim=0) if do_cfg else x
            pred = self.m.motion_transformer(
                x=x_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if do_cfg:
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            return pred

        # Initialize noise on GPU
        y0 = torch.randn(1, self._train_frames, self._input_dim, device=device)

        # Solve ODE from t=0 to t=1 — Euler method (vendor default)
        t = torch.linspace(0, 1, self._n_steps + 1, device=device)
        trajectory = odeint(ode_fn, y0, t, method="euler")
        result = trajectory[-1][:, :n_frames, ...].clone()
        return result

    def _decode_motion(self, latent):
        """Decode motion latent via pipeline's method (handles 22→52 joint padding,
        Savitzky-Golay translation smoothing, SLERP rotation smoothing)."""
        return self.m.pipeline.decode_motion_from_latent(
            latent, should_apply_smooothing=True,
        )
