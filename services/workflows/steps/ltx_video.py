"""LTX Video step executors — native Director, spatial upscale, and multi-stage generation.

Provides step executors that leverage Wan2GP's LTX2 handler directly (no ComfyUI bridge)
with the WDC Director's prompt relay for temporal segment control.

Step types:
  ltx_generate  — Full LTX video generation with prompt relay support
  ltx_upscale   — Latent spatial/temporal upscaling
  ltx_director  — Director-only step: encode segments + temporal masks (preprocessing)

The ltx_generate step is the primary entry point. It supports:
  - Single-prompt video generation (basic mode)
  - Prompt relay with temporal segments (Director mode)
  - First-frame / last-frame conditioning (FFLF)
  - Audio conditioning (custom audio or generated)
  - Spatial upscaling (2x latent-space upscale before decode)
  - Multi-stage generation (2-phase guidance)
  - LoRA selection (dynamic, runtime)
  - Frame injection at specific positions
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from ray import serve

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)


class LTXGenerateStep(StepExecutor):
    """LTX video generation with Director-level controls.

    Calls Wan2GP's LTX2 handler through the Forge with full parameter support.
    When prompt relay is enabled (segments provided), applies temporal attention
    masking for per-segment prompt control.

    Params:
      input_prompt: Global prompt for the entire video
      image_b64: Start frame image (base64 or file path)
      image_end_b64: End frame image for FFLF conditioning
      n_prompt: Negative prompt
      seed: Random seed
      fps: Frames per second (default: 24)
      frame_num: Number of frames (default: 121 = ~5s)
      guide_scale: CFG guidance scale (default: 3.0)
      sampling_steps: Number of denoising steps
      loras_selected: List of LoRA filenames to load
      audio_b64: Audio conditioning data
      audio_scale: Audio strength (0-1)
      audio_prompt_type: Audio mode ('' for generated, 'A' for soundtrack)

      # Director / Prompt Relay params:
      local_prompts: Pipe-separated segment prompts (enables Director mode)
      segment_lengths: Comma-separated frame counts per segment
      epsilon: Prompt relay boundary sharpness (default: 0.001)

      # Upscale params:
      spatial_upscale: bool — run 2x spatial upscaler after generation

      # Advanced:
      guide_phases: Number of guidance phases (1 or 2, default: 2)
      perturbation_switch: Perturbation mode (0=off, 1=skip layer, 2=skip self-attn)
      perturbation_layers: Layer indices for perturbation
      denoising_strength: Control video strength
      video_prompt_type: Video conditioning mode
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()

        # Extract and clean params
        model = params.pop("_model", params.pop("model", "ltx2"))

        # Set LTX2-specific defaults if not provided
        params.setdefault("sample_solver", "euler")
        params.setdefault("fps", 24)
        params.setdefault("frame_num", 121)

        # Handle prompt relay preprocessing if segments are provided
        relay_config = self._build_relay_config(params)

        # Prepare the forge payload
        forge_params = await self._prepare_params(params, context)

        # If relay is enabled, attach the relay config for the forge to use
        if relay_config:
            forge_params["_relay_config"] = json.dumps(relay_config)

        # Call Forge via Ray handle
        forge = serve.get_deployment_handle("forge", "forge")
        result = await forge.invoke.remote("wan2gp", forge_params, model)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        status = result.get("status", "")
        if status not in ("ok", "success"):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"LTX generation failed: {error}")

        # Store output
        outputs = await self._store_outputs(result, context)
        metadata = {
            "model": model,
            "prompt_relay": bool(relay_config),
            "segments": len(relay_config.get("segments", [])) if relay_config else 0,
        }

        return StepResult(outputs=outputs, duration_ms=elapsed_ms, metadata=metadata)

    def _build_relay_config(self, params: dict) -> dict | None:
        """Build prompt relay config if Director mode params are present."""
        local_prompts = params.pop("local_prompts", "")
        segment_lengths_str = params.pop("segment_lengths", "")
        epsilon = float(params.pop("epsilon", "0.001"))

        if not local_prompts:
            return None

        prompts = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not prompts:
            return None

        lengths = None
        if segment_lengths_str:
            lengths = [int(x.strip()) for x in segment_lengths_str.split(",") if x.strip()]

        return {
            "local_prompts": prompts,
            "segment_lengths": lengths,
            "epsilon": epsilon,
        }

    async def _prepare_params(self, params: dict, context: StepContext) -> dict:
        """Resolve artifact paths and encode images."""
        resolved = {}

        for key, value in params.items():
            if key.startswith("_"):
                continue
            if key in ("image_b64", "image_end_b64", "audio_b64") and isinstance(value, (str, Path)):
                path = Path(value) if isinstance(value, str) and value.startswith("/") else None
                if path and path.exists():
                    resolved[key] = base64.b64encode(path.read_bytes()).decode()
                else:
                    resolved[key] = value
            elif key == "loras_selected" and isinstance(value, str) and value.strip():
                # Normalize comma-separated string to list, preserving
                # "name:strength" entries intact (deployment.py parses them).
                resolved[key] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                resolved[key] = value

        # Drop None values
        return {k: v for k, v in resolved.items() if v is not None}

    async def _store_outputs(self, result: dict, context: StepContext) -> dict[str, str]:
        """Store video/audio output as artifacts."""
        outputs = {}
        data = result.get("data")
        media_type = result.get("media_type", "video/mp4")

        if data and isinstance(data, str):
            artifact = await context.artifacts.store(
                run_id=context.run_id,
                step_id=context.step_id,
                name="output",
                data=base64.b64decode(data),
                media_type=media_type,
            )
            outputs["output"] = str(artifact.file_path)

        # Store metadata fields
        _SKIP = {"status", "data", "media_type", "model"}
        for k, v in result.items():
            if k not in _SKIP and isinstance(v, (str, int, float, bool)):
                outputs[k] = str(v)

        return outputs


class LTXSpatialUpscaleStep(StepExecutor):
    """Latent spatial upscaler for LTX video.

    Takes a generated video, re-encodes it into latent space, runs the
    LTX spatial upscaler (2x), then decodes back to pixel space.

    This is much higher quality than post-decode upscaling because it
    operates in the latent space before the VAE decoder.

    Params:
      video: Path to input video file
      model: LTX model variant (default: ltx2)
      seed: Random seed
      fps: Output FPS
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()

        model = params.pop("_model", params.pop("model", "ltx2"))

        # Read video and encode to base64
        video_path = params.get("video", "")
        if video_path and Path(video_path).exists():
            params["video_b64"] = base64.b64encode(
                Path(video_path).read_bytes()
            ).decode()
            params.pop("video", None)

        params["_mode"] = "spatial_upscale"

        forge = serve.get_deployment_handle("forge", "forge")
        result = await forge.invoke.remote("wan2gp", params, model)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        status = result.get("status", "")
        if status not in ("ok", "success"):
            raise RuntimeError(f"Spatial upscale failed: {result.get('error')}")

        outputs = {}
        data = result.get("data")
        if data and isinstance(data, str):
            artifact = await context.artifacts.store(
                context.run_id, context.step_id, "output",
                base64.b64decode(data),
                result.get("media_type", "video/mp4"),
            )
            outputs["output"] = str(artifact.file_path)

        return StepResult(outputs=outputs, duration_ms=elapsed_ms)
