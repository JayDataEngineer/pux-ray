"""NVIDIA Upscale — GPU-accelerated video/image upscaling via torchvision.

Uses torchvision.transforms.functional.resize with Lanczos interpolation
on GPU. Handles both image (PNG/JPEG) and video (MP4) inputs.

Available as MCP tool via the Wan2GP `run` endpoint.
"""
from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)

from models.base_handler import BaseFamilyHandler, _make_handler_cls


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["nvidia_upscale"]
    FAMILY = "upscale"
    FAMILY_INFOS = {"nvidia_upscale": (400, "GPU Upscale (Lanczos)")}
    MODEL_DEF = {"audio_only": False, "image_outputs": False}
    DEFAULTS = {"target_resolution": "1920x1080"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        pipeline = _UpscalePipeline()
        pipe_dict = {}
        return pipeline, pipe_dict


class _UpscalePipeline:
    """GPU-accelerated upscale pipeline using Lanczos interpolation."""

    def generate(self, *, video: str | None = None,
                 target_resolution: str = "1920x1080",
                 image_b64: str = "", **kw) -> dict:
        try:
            tw, th = map(int, target_resolution.split("x"))
        except (ValueError, AttributeError):
            tw, th = 1920, 1080

        if video and Path(video).exists():
            return self._upscale_video(video, tw, th)
        elif image_b64:
            return self._upscale_image_b64(image_b64, tw, th)
        else:
            raise ValueError("Provide 'video' path or 'image_b64' data")

    def _upscale_image_b64(self, image_b64: str, tw: int, th: int) -> dict:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        tensor = F.to_tensor(img).unsqueeze(0).cuda()
        upscaled = F.resize(tensor, [th, tw],
                            interpolation=F.InterpolationMode.LANCZOS,
                            antialias=True)
        result = F.to_pil_image(upscaled.squeeze(0).cpu())
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return {"status": "success",
                "data": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "image/png"}

    def _upscale_video(self, video_path: str, tw: int, th: int) -> dict:
        import subprocess
        import os

        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out.close()

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"scale={tw}:{th}:flags=lanczos",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            out.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        data = Path(out.name).read_bytes()
        os.unlink(out.name)

        return {"status": "success",
                "data": base64.b64encode(data).decode(),
                "media_type": "video/mp4"}
