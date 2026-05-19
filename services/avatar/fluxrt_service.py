"""FluxRT rendering — reference image + text prompt -> avatar frames.

FluxRT (FLUX.2-Klein-4B) is a real-time stream editor. It uses text +
reference image conditioning only — NO pose/ControlNet support. The API
is StreamProcessor with JSON config, shared memory I/O.

For the avatar pipeline, FluxRT renders character frames from:
  - A reference character image (the avatar portrait)
  - Text prompt describing the desired pose/expression
  - Built-in LivePortrait for face reenactment

Pose conditioning is handled via prompt engineering until a ControlNet
adapter becomes available:
  "person gesturing with left hand, leaning forward, arms raised"

Target: 15-30 FPS at 512x512 with int8 on RTX 4090.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FluxRTService:
    """FluxRT real-time avatar rendering."""

    vram_mb: int = 0  # Self-managed via FluxRT internals
    service_name: str = "fluxrt"
    default_model: str = "flux_klein_int8"

    def __init__(self):
        self._loaded = False
        self._processor = None
        self._config_path = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        from fluxrt import StreamProcessor
        from services.avatar import models_root

        model_dir = self._resolve_model(model_name, models_root())

        use_int8 = "int8" in model_name or quant == "int8"
        config = self._build_config(model_dir, use_int8)

        # Write config to temp file (FluxRT requires file-based config)
        cfg_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="fluxrt_",
        )
        json.dump(config, cfg_file)
        cfg_file.close()
        self._config_path = cfg_file.name

        self._processor = StreamProcessor(self._config_path)
        if use_int8:
            self._processor.enable_quantization()

        self._processor.start()
        self._loaded = True
        logger.info("FluxRT: loaded %s (int8=%s)", model_name, use_int8)

    def unload(self) -> None:
        if self._processor:
            self._processor.stop()
        self._processor = None
        self._loaded = False
        if self._config_path and os.path.exists(self._config_path):
            os.unlink(self._config_path)
        self._config_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._loaded

    def actual_vram_mb(self) -> int:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))

    def render(
        self,
        reference_image: str | np.ndarray,
        pose_descriptions: list[str],
        style_prompt: str = "",
        width: int = 512,
        height: int = 512,
        fps: int = 30,
        steps: int = 4,
    ) -> list[np.ndarray]:
        """Render avatar frames using FluxRT.

        Since FluxRT has no pose image conditioning, we render each frame
        using the pose description as the text prompt. The reference image
        provides character identity.

        Args:
            reference_image: Path to character portrait.
            pose_descriptions: Per-frame text descriptions of desired pose.
            style_prompt: Style prefix appended to each frame prompt.
            width/height: Output resolution.
            fps: Target FPS metadata.
            steps: Denoising steps (2-4 for distilled models).

        Returns:
            List of RGB numpy arrays (H, W, 3), uint8.
        """
        if not self._loaded:
            raise RuntimeError("FluxRT not loaded")

        self._processor.set_reference_image(self._load_reference(reference_image))
        self._processor.set_steps(steps)

        frames = []
        input_tensor = self._processor.get_input_tensor()
        output_tensor = self._processor.get_output_tensor()

        for desc in pose_descriptions:
            prompt = f"{style_prompt}, {desc}" if style_prompt else desc
            self._processor.set_prompt(prompt)

            # Feed a black frame to trigger generation
            input_tensor.copy_from(np.zeros((height, width, 3), dtype=np.uint8))
            frame = output_tensor.to_numpy()
            frames.append(frame.copy())

        return frames

    def render_to_disk(
        self,
        reference_image: str | np.ndarray,
        pose_descriptions: list[str],
        style_prompt: str = "",
        output_dir: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 4,
    ) -> dict:
        """Render frames and save directly to disk."""
        import cv2

        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()

        frames = self.render(
            reference_image, pose_descriptions, style_prompt,
            width, height, steps=steps,
        )

        frame_paths = []
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"frame_{i:05d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(path)

        render_ms = (time.time() - t0) * 1000
        return {
            "frames_dir": output_dir,
            "frame_count": len(frame_paths),
            "render_time_ms": render_ms,
        }

    def _build_config(self, model_dir: str, use_int8: bool) -> dict:
        return {
            "model_path": model_dir,
            "resolution": [512, 512],
            "use_reference_image": True,
            "enable_spatial_cache": True,
            "int8_models_path": model_dir if use_int8 else None,
            "lip_transfer": {
                "enable": False,
            },
        }

    def _resolve_model(self, model_name: str, models_root: str) -> str:
        candidates = [
            Path(models_root) / "avatar" / "fluxrt" / model_name,
            Path(models_root) / "avatar" / "fluxrt" / "FLUX.2-klein-4B-int8",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"FluxRT model not found: {model_name}")

    def _load_reference(self, ref: str | np.ndarray):
        if isinstance(ref, np.ndarray):
            return ref
        if os.path.isfile(ref):
            import cv2
            img = cv2.imread(ref)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ref
