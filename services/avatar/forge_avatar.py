"""Avatar Forge Service — orchestrates Kimodo + FluxRT pipeline.

Text-to-avatar pipeline where the only human input is text.
The AI generates speech (TTS), co-speech gestures (Kimodo),
and rendered video (FluxRT).

VRAM budget (single RTX 4090, 24GB):
  - Kimodo: <3GB (CPU text offload)
  - FluxRT int8: ~10GB
  - Total: ~13GB peak — no staging needed

FluxRT has no pose conditioning — rendering uses text prompts derived
from SOMA joint position analysis to describe each frame's body position.
"""
from __future__ import annotations

import gc
import logging
import os
import time
import uuid

import numpy as np
import torch

from services.forge_base import ForgeService
from services.avatar.kimodo_service import KimodoService, KIMODO_FPS
from services.avatar.fluxrt_service import FluxRTService
from services.avatar.pose_describer import describe_poses

logger = logging.getLogger(__name__)

CHUNK_DURATION_S = 5.0
OUTPUT_ROOT = "/tmp/avatar"


class AvatarForgeService(ForgeService):
    """Forge-managed avatar pipeline. Orchestrates Kimodo -> FluxRT."""

    vram_mb: int = 0  # Self-managed
    service_name: str = "avatar"
    default_model: str = "kimodo-soma-rp"

    def __init__(self):
        super().__init__()
        self._kimodo = KimodoService()
        self._fluxrt = FluxRTService()

    def load(self, model_name: str, quant: str | None = None) -> None:
        self._loaded = True
        logger.info("Avatar: ready (Kimodo + FluxRT loaded on demand)")

    def unload(self) -> None:
        self._kimodo.unload()
        self._fluxrt.unload()
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        chunk_id = uuid.uuid4().hex[:8]
        if not output_dir:
            output_dir = os.path.join(OUTPUT_ROOT, f"chunk_{chunk_id}")
        os.makedirs(output_dir, exist_ok=True)

        t_total = time.time()
        timings = {}

        # ── Stage 1: Kimodo — text -> SOMA 77-joint motion ──────────────
        t0 = time.time()
        if not self._kimodo.is_loaded():
            self._kimodo.load(payload.get("model", "kimodo-soma-rp"))

        motion_result = self._kimodo.generate(
            text=text,
            duration_seconds=duration_s,
            num_denoising_steps=denoising_steps,
            emotion=emotion,
        )
        timings["kimodo_ms"] = motion_result["generation_time_ms"]
        timings["kimodo_vram_mb"] = self._kimodo.actual_vram_mb()

        # Save motion data
        np.savez(
            os.path.join(output_dir, "motion_data.npz"),
            posed_joints=motion_result["posed_joints"],
            global_rot_mats=motion_result["global_rot_mats"],
            local_rot_mats=motion_result["local_rot_mats"],
            foot_contacts=motion_result["foot_contacts"],
            root_positions=motion_result["root_positions"],
            root_heading=motion_result["root_heading"],
        )
        timings["kimodo_total_ms"] = (time.time() - t0) * 1000

        posed_joints = motion_result["posed_joints"]
        foot_contacts = motion_result["foot_contacts"]
        num_frames = motion_result["num_frames"]

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
            "model": motion_result["model_name"],
            "prompt": motion_result["prompt"],
        }
