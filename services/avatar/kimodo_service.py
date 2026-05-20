"""Kimodo motion generation — text -> SOMA 77-joint motion.

Wraps NVIDIA Kimodo for Forge-compatible inference. Uses Kimodo-SOMA-RP-v1.1
which natively outputs SOMA 77-joint skeleton — no SMPL conversion needed.

VRAM: <3GB with TEXT_ENCODER_DEVICE=cpu. The 282M diffusion model stays on GPU,
the LLM2Vec text encoder (Meta-Llama-3-8B-Instruct) runs on CPU (~15GB RAM).

Output: dict with posed_joints (T,77,3), global_rot_mats (T,77,3,3),
        local_rot_mats (T,77,3,3), foot_contacts (T,6), root_positions (T,3).
"""
from __future__ import annotations

import gc
import logging
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

KIMODO_FPS = 30
MAX_DURATION_S = 10.0  # Kimodo max is 300 frames (10s at 30fps)


class KimodoService:
    """Text -> SOMA 77-joint motion generation via Kimodo."""

    vram_mb: int = 0  # Self-managed, <3GB with CPU text offload
    service_name: str = "kimodo"
    default_model: str = "kimodo-soma-rp"

    def __init__(self):
        self._loaded = False
        self._model = None
        self._model_name = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        from kimodo import load_model

        os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

        logger.info("Kimodo: loading %s (text encoder on CPU)...", model_name)
        self._model, self._model_name = load_model(
            model_name,
            device="cuda:0",
            return_resolved_name=True,
        )
        self._model.eval()
        self._loaded = True
        logger.info("Kimodo: loaded %s", self._model_name)

    def unload(self) -> None:
        self._model = None
        self._model_name = None
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
        duration_seconds: float = 5.0,
        num_denoising_steps: int = 100,
        emotion: str = "",
        seed: int | None = None,
    ) -> dict:
        """Generate SOMA 77-joint motion from text.

        Args:
            text: Motion description (e.g. "a person waves hello").
            duration_seconds: Motion duration (1-10 seconds).
            num_denoising_steps: DDIM denoising steps (50-200, more = better quality).
            emotion: Optional emotion to prepend.
            seed: Random seed for reproducibility.

        Returns:
            Dict with motion arrays, timing, and metadata.
        """
        if not self._loaded:
            raise RuntimeError("Kimodo model not loaded")

        duration_seconds = min(duration_seconds, MAX_DURATION_S)
        num_frames = int(duration_seconds * KIMODO_FPS)

        prompt = text
        if emotion:
            prompt = f"a person expressing {emotion}, {text}"

        t0 = time.time()

        kwargs = dict(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=num_denoising_steps,
            num_samples=1,
            cfg_type="separated",
            cfg_weight=[2.0, 2.0],
            return_numpy=True,
            post_processing=True,
        )
        if seed is not None:
            kwargs["seed"] = seed

        with torch.no_grad():
            output = self._model(**kwargs)

        gen_ms = (time.time() - t0) * 1000

        # Extract first (and only) sample
        posed_joints = output["posed_joints"][0]        # (T, 77, 3)
        global_rot_mats = output["global_rot_mats"][0]  # (T, 77, 3, 3)
        local_rot_mats = output["local_rot_mats"][0]    # (T, 77, 3, 3)
        foot_contacts = output["foot_contacts"][0]      # (T, 6)
        root_positions = output["root_positions"][0]    # (T, 3)
        root_heading = output["global_root_heading"][0]  # (T, 2)

        return {
            "posed_joints": posed_joints,
            "global_rot_mats": global_rot_mats,
            "local_rot_mats": local_rot_mats,
            "foot_contacts": foot_contacts,
            "root_positions": root_positions,
            "root_heading": root_heading,
            "num_frames": num_frames,
            "fps": KIMODO_FPS,
            "duration_seconds": duration_seconds,
            "generation_time_ms": gen_ms,
            "model_name": self._model_name,
            "prompt": prompt,
        }

    def generate_multi_prompt(
        self,
        prompts: list[str],
        durations: list[float],
        num_denoising_steps: int = 100,
    ) -> dict:
        """Generate multi-segment motion with transitions.

        Args:
            prompts: List of text descriptions for each segment.
            durations: List of durations in seconds for each segment.
            num_denoising_steps: DDIM steps.

        Returns:
            Same format as generate() but with stitched segments.
        """
        if not self._loaded:
            raise RuntimeError("Kimodo model not loaded")

        num_frames = [int(d * KIMODO_FPS) for d in durations]

        t0 = time.time()
        with torch.no_grad():
            output = self._model(
                prompts=prompts,
                num_frames=num_frames,
                num_denoising_steps=num_denoising_steps,
                multi_prompt=True,
                num_samples=1,
                cfg_type="separated",
                cfg_weight=[2.0, 2.0],
                return_numpy=True,
                post_processing=True,
            )
        gen_ms = (time.time() - t0) * 1000

        return {
            "posed_joints": output["posed_joints"][0],
            "global_rot_mats": output["global_rot_mats"][0],
            "local_rot_mats": output["local_rot_mats"][0],
            "foot_contacts": output["foot_contacts"][0],
            "root_positions": output["root_positions"][0],
            "root_heading": output["global_root_heading"][0],
            "num_frames": sum(num_frames),
            "fps": KIMODO_FPS,
            "duration_seconds": sum(durations),
            "generation_time_ms": gen_ms,
            "model_name": self._model_name,
            "prompts": prompts,
        }
