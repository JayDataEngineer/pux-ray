"""Avatar Forge Service — orchestrates GEM + SOMA + FluxRT pipeline.

Text-to-avatar pipeline where the only human input is text.
The AI generates speech (TTS), co-speech gestures (GEM), body mesh
(SOMA), and rendered video (FluxRT).

VRAM staging strategy (single RTX 4090, 24GB):
  1. SOMA (<1GB, stays resident)
  2. GEM (~?GB, load -> generate -> unload)
  3. FluxRT int8 (~10GB, load -> render -> unload)

GEM and FluxRT are never loaded simultaneously.

FluxRT has no pose conditioning — rendering uses text prompts derived
from SOMA pose analysis to describe each frame's body position.
"""
from __future__ import annotations

import gc
import logging
import os
import time
import uuid

import torch

from services.forge_base import ForgeService
from services.avatar.gem_service import GEMService, GEM_FPS
from services.avatar.soma_service import SOMAService
from services.avatar.fluxrt_service import FluxRTService
from services.avatar.pose_describer import describe_poses

logger = logging.getLogger(__name__)

CHUNK_DURATION_S = 5.0
OUTPUT_ROOT = "/tmp/avatar"


class AvatarForgeService(ForgeService):
    """Forge-managed avatar pipeline. Orchestrates GEM -> SOMA -> FluxRT."""

    vram_mb: int = 0  # Self-managed — stages sub-services
    service_name: str = "avatar"
    default_model: str = "gem_smpl"

    def __init__(self):
        super().__init__()
        self._gem = GEMService()
        self._soma = SOMAService()
        self._fluxrt = FluxRTService()
        self._soma_loaded = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        if not self._soma_loaded:
            self._soma.load("soma_smpl")
            self._soma_loaded = True
            logger.info("Avatar: SOMA loaded (resident)")
        self._loaded = True

    def unload(self) -> None:
        self._gem.unload()
        self._fluxrt.unload()
        if self._soma_loaded:
            self._soma.unload()
            self._soma_loaded = False
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

        chunk_id = uuid.uuid4().hex[:8]
        if not output_dir:
            output_dir = os.path.join(OUTPUT_ROOT, f"chunk_{chunk_id}")
        os.makedirs(output_dir, exist_ok=True)

        t_total = time.time()
        timings = {}

        # ── Stage 1: GEM — text (+ audio) -> SMPL motion ──────────────────
        t0 = time.time()
        self._ensure_gem_loaded(payload.get("model", "gem_smpl"))

        gesture_text = text
        if emotion:
            gesture_text = f"a person expressing {emotion}, {text}"

        duration_frames = int(duration_s * GEM_FPS)
        gem_result = self._gem.generate(
            text=gesture_text,
            audio_path=audio_path,
            duration_frames=duration_frames,
        )
        timings["gem_ms"] = gem_result["generation_time_ms"]

        # Free GEM VRAM for FluxRT
        self._gem.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        timings["gem_total_ms"] = (time.time() - t0) * 1000

        smpl_params = gem_result["smpl_params"]
        if any(v is None for v in smpl_params.values()):
            return {
                "status": "error",
                "error": "GEM generation returned incomplete SMPL params",
                "smpl_params": {k: "None" if v is None else v.shape
                                for k, v in smpl_params.items()},
            }

        torch.save(smpl_params, os.path.join(output_dir, "smpl_params.pt"))

        # ── Stage 2: SOMA — SMPL -> mesh + skeleton renders ───────────────
        t0 = time.time()
        soma_result = self._soma.convert_smpl_to_soma(smpl_params)
        timings["soma_ms"] = soma_result["conversion_time_ms"]

        # Save SOMA output
        torch.save({
            "soma_rotations": soma_result["soma_rotations"],
            "root_translation": soma_result["root_translation"],
            "vertices": soma_result["vertices"],
        }, os.path.join(output_dir, "soma_output.pt"))

        # Generate skeleton renders (used for debugging even if FluxRT can't use them)
        pose_images = []
        if should_render and "vertices" in soma_result:
            pose_images = self._soma.render_skeleton(
                soma_result["vertices"], resolution=(512, 512),
            )
            # Save skeleton frames for inspection
            skel_dir = os.path.join(output_dir, "skeleton")
            os.makedirs(skel_dir, exist_ok=True)
            import cv2
            for i, img in enumerate(pose_images):
                cv2.imwrite(os.path.join(skel_dir, f"skel_{i:05d}.png"), img)

        timings["soma_total_ms"] = (time.time() - t0) * 1000

        # ── Stage 3: FluxRT — reference + prompt -> video frames ──────────
        render_result = {}
        if should_render and reference_image:
            t0 = time.time()
            self._ensure_fluxrt_loaded(payload.get("fluxrt_model", "flux_klein_int8"))

            # Generate per-frame pose descriptions from SMPL params
            pose_descriptions = describe_poses(smpl_params["body_pose"], emotion)

            frames_dir = os.path.join(output_dir, "frames")
            render_result = self._fluxrt.render_to_disk(
                reference_image=reference_image,
                pose_descriptions=pose_descriptions,
                style_prompt=style_prompt,
                output_dir=frames_dir,
            )

            self._fluxrt.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timings["fluxrt_ms"] = render_result.get("render_time_ms", 0)
            timings["fluxrt_total_ms"] = (time.time() - t0) * 1000

        total_ms = (time.time() - t_total) * 1000

        return {
            "status": "success",
            "chunk_id": chunk_id,
            "output_dir": output_dir,
            "smpl_params_path": os.path.join(output_dir, "smpl_params.pt"),
            "frame_count": render_result.get("frame_count", 0),
            "frames_dir": render_result.get("frames_dir", ""),
            "fps": GEM_FPS,
            "duration_seconds": duration_frames / GEM_FPS,
            "pipeline_latency_ms": total_ms,
            "timings": timings,
        }

    def _ensure_gem_loaded(self, model_name: str) -> None:
        if not self._gem.is_loaded():
            self._gem.load(model_name)

    def _ensure_fluxrt_loaded(self, model_name: str) -> None:
        if not self._fluxrt.is_loaded():
            self._fluxrt.load(model_name)
