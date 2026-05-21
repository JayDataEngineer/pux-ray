"""Avatar Forge Service — orchestrates Kimodo + FluxRT pipeline.

Text-to-avatar pipeline where the only human input is text.
The AI generates speech (TTS), co-speech gestures (Kimodo),
and rendered video (FluxRT).

Uses the Wan2GP model engine for Kimodo inference — gets mmgp VRAM
management, shared GPU pool, and proper loading/unloading for free.

VRAM budget (single RTX 4090, 24GB):
  - Kimodo: ~1.2GB via Wan2GP mmgp pool
  - FluxRT int8: ~10GB
  - Total: ~11.2GB peak — no staging needed

FluxRT has no pose conditioning — rendering uses text prompts derived
from SOMA joint position analysis to describe each frame's body position.
"""
from __future__ import annotations

import base64
import gc
import io
import logging
import os
import time
import uuid

import numpy as np
import torch

from services.forge_base import ForgeService
from services.avatar.fluxrt_service import FluxRTService
from services.avatar.pose_describer import describe_poses

logger = logging.getLogger(__name__)

KIMODO_FPS = 30
CHUNK_DURATION_S = 5.0
OUTPUT_ROOT = "/tmp/avatar"


class AvatarForgeService(ForgeService):
    """Forge-managed avatar pipeline. Orchestrates Kimodo -> FluxRT."""

    vram_mb: int = 0  # Self-managed
    service_name: str = "avatar"
    default_model: str = "kimodo-soma-rp"

    def __init__(self):
        super().__init__()
        self._fluxrt = FluxRTService()
        self._wan2gp = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        self._loaded = True
        logger.info("Avatar: ready (Kimodo via Wan2GP, FluxRT loaded on demand)")

    def unload(self) -> None:
        self._fluxrt.unload()
        self._wan2gp = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_wan2gp(self):
        """Lazily get the Wan2GP service reference for Kimodo inference."""
        if self._wan2gp is not None:
            return self._wan2gp

        from services.wan2gp.deployment import Wan2GPDeployment
        # The Wan2GP service is a Ray Serve deployment — get its handle
        import ray
        handle = ray.serve.get_deployment("wan2gp").get_handle()
        self._wan2gp = handle
        return handle

    def _call_kimodo(self, text: str, duration_s: float, emotion: str,
                      denoising_steps: int, model: str) -> dict:
        """Call Kimodo through the Wan2GP service (mmgp-managed GPU)."""
        prompt = text
        if emotion:
            prompt = f"a person expressing {emotion}, {text}"

        num_frames = int(duration_s * KIMODO_FPS)

        # Call Wan2GP service which routes to the kimodo handler
        payload = {
            "model": model,
            "prompts": prompt,
            "num_frames": num_frames,
            "num_denoising_steps": denoising_steps,
            "post_processing": True,
        }

        handle = self._get_wan2gp()
        result = ray.get(handle.infer.remote(payload))
        return result

    def infer(self, payload: dict) -> dict:
        """Run the full avatar pipeline.

        Required: text (str)
        Optional: reference_image, style_prompt, emotion, audio_path,
                  duration_seconds, output_dir, render, no_render
        """
        text = payload.get("text", "")
        if not text:
            return {"status": "error", "error": "Missing required field: text"}

        reference_image = payload.get("reference_image", "")
        style_prompt = payload.get("style_prompt", "anime character, high quality")
        emotion = payload.get("emotion", "")
        audio_path = payload.get("audio_path")
        duration_s = payload.get("duration_seconds", CHUNK_DURATION_S)
        output_dir = payload.get("output_dir")
        should_render = not payload.get("no_render", False) and payload.get("render", True)
        denoising_steps = payload.get("denoising_steps", 100)
        model = payload.get("model", "kimodo-soma-rp")

        chunk_id = uuid.uuid4().hex[:8]
        if not output_dir:
            output_dir = os.path.join(OUTPUT_ROOT, f"chunk_{chunk_id}")
        os.makedirs(output_dir, exist_ok=True)

        t_total = time.time()
        timings = {}

        # ── Stage 1: Kimodo via Wan2GP — text -> SOMA 77-joint motion ────
        t0 = time.time()
        kimodo_result = self._call_kimodo(
            text=text,
            duration_s=duration_s,
            emotion=emotion,
            denoising_steps=denoising_steps,
            model=model,
        )

        if kimodo_result.get("status") != "success":
            return {
                "status": "error",
                "error": f"Kimodo failed: {kimodo_result.get('error', 'unknown')}",
            }
        timings["kimodo_total_ms"] = (time.time() - t0) * 1000

        # Decode motion data from NPZ
        npz_b64 = kimodo_result.get("npz_data", "")
        if not npz_b64:
            return {"status": "error", "error": "Kimodo returned no motion data"}

        npz_bytes = base64.b64decode(npz_b64)
        motion_data = dict(np.load(io.BytesIO(npz_bytes)))

        posed_joints = motion_data["posed_joints"]
        # Handle batch dimension
        if posed_joints.ndim == 4:
            posed_joints = posed_joints[0]
        foot_contacts = motion_data.get("foot_contacts")
        if foot_contacts is not None and foot_contacts.ndim == 3:
            foot_contacts = foot_contacts[0]

        num_frames = len(posed_joints)

        # Save motion data
        np.savez(os.path.join(output_dir, "motion_data.npz"), **motion_data)

        # ── Stage 2: FluxRT — reference + prompt -> video frames ──────────
        render_result = {}
        if should_render and reference_image:
            t0 = time.time()

            if not self._fluxrt.is_loaded():
                self._fluxrt.load(
                    payload.get("fluxrt_model", "flux_klein_int8"),
                    quant="int8",
                )

            pose_descriptions = describe_poses(
                posed_joints=posed_joints,
                foot_contacts=foot_contacts,
                emotion=emotion,
            )

            frames_dir = os.path.join(output_dir, "frames")
            render_result = self._fluxrt.render_to_disk(
                reference_image=reference_image,
                pose_descriptions=pose_descriptions,
                style_prompt=style_prompt,
                output_dir=frames_dir,
            )
            timings["fluxrt_ms"] = render_result.get("render_time_ms", 0)
            timings["fluxrt_total_ms"] = (time.time() - t0) * 1000

        total_ms = (time.time() - t_total) * 1000

        # Skeleton preview from Kimodo result
        preview_b64 = kimodo_result.get("data", "")
        media_type = kimodo_result.get("media_type", "application/x-npz")

        return {
            "status": "success",
            "chunk_id": chunk_id,
            "output_dir": output_dir,
            "motion_data_path": os.path.join(output_dir, "motion_data.npz"),
            "frame_count": render_result.get("frame_count", 0),
            "frames_dir": render_result.get("frames_dir", ""),
            "fps": KIMODO_FPS,
            "duration_seconds": num_frames / KIMODO_FPS,
            "num_frames": num_frames,
            "pipeline_latency_ms": total_ms,
            "timings": timings,
            "model": kimodo_result.get("model", model),
            "prompt": kimodo_result.get("prompt", text),
            "tensor_shapes": kimodo_result.get("tensor_shapes", {}),
            "data": preview_b64,
            "media_type": media_type,
        }
