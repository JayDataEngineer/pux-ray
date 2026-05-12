"""Wan2GP handler — wraps vendor family_handler pattern into Model Engine.

Dynamically imports Wan2GP's per-family handlers (wan, hunyuan, flux, TTS),
calls their load_model(), and adapts the output to LoadResult for the shared
mmgp pool.

The Wan2GP model object becomes the "pipeline" in LoadResult — it already
has a .generate() method that works with mmgp-managed modules.

Reference: vendor/wan2gp/models/*/handler.py
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)

WAN2GP_VENDOR = Path(__file__).parents[4] / "vendor" / "wan2gp"

# ── Model registry ──────────────────────────────────────────────────────────

VARIANTS = {
    # WAN video
    "wan/t2v-14B": ModelVariant(
        name="wan/t2v-14B", family="wan2gp", display_name="WAN 2.1 Text-to-Video 14B",
        vram_estimate_gb=14,
        defaults={"width": 1280, "height": 720, "frames": 81, "steps": 50, "guidance": 5.0},
    ),
    "wan/i2v-14B": ModelVariant(
        name="wan/i2v-14B", family="wan2gp", display_name="WAN 2.1 Image-to-Video 14B",
        vram_estimate_gb=14,
        defaults={"width": 832, "height": 480, "frames": 81, "steps": 50, "guidance": 5.0},
    ),
    # Hunyuan video
    "hunyuan/t2v": ModelVariant(
        name="hunyuan/t2v", family="wan2gp", display_name="Hunyuan Text-to-Video",
        vram_estimate_gb=12,
        defaults={"width": 848, "height": 480, "frames": 125, "steps": 30, "guidance": 6.0},
    ),
    # Flux image
    "flux/t2i": ModelVariant(
        name="flux/t2i", family="wan2gp", display_name="Flux Text-to-Image",
        vram_estimate_gb=8,
        defaults={"width": 1024, "height": 1024, "steps": 28, "guidance": 3.5},
    ),
}

# Maps variant name → Wan2GP handler config
_HANDLER_MAP = {
    "wan/t2v-14B": {
        "handler": "models.wan.wan_handler",
        "model_type": "t2v",
        "base_model_type": "t2v",
    },
    "wan/i2v-14B": {
        "handler": "models.wan.wan_handler",
        "model_type": "i2v",
        "base_model_type": "i2v",
    },
    "hunyuan/t2v": {
        "handler": "models.hyvideo.hunyuan_handler",
        "model_type": "hunyuan",
        "base_model_type": "hunyuan",
    },
    "flux/t2i": {
        "handler": "models.flux.flux_handler",
        "model_type": "flux",
        "base_model_type": "flux",
    },
}

MMGP_PROFILES = {
    "max_speed": 1,
    "balanced": 2,
    "low_vram": 4,
    "minimum": 5,
}


class Wan2GPHandler(BaseHandler):
    """Wraps Wan2GP's family_handler pattern into Model Engine.

    Each Wan2GP model variant (wan, hunyuan, flux) is loaded via its
    vendor handler. The returned (model, pipe) tuple is adapted to
    LoadResult — model becomes the pipeline, pipe becomes the mmgp dict.
    """

    def __init__(self, mmgp_profile: int = 2):
        self._mmgp_profile = mmgp_profile
        self._vendor_loaded = False

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(
                f"Unknown Wan2GP model: {model_type}. Available: {list(VARIANTS.keys())}"
            )
        return VARIANTS[model_type]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        quantize_transformer: bool = True,
        mmgp_profile: int | None = None,
        **kwargs,
    ) -> LoadResult:
        """Load a Wan2GP model via its vendor handler.

        Args:
            model_type: variant name (e.g., "wan/t2v-14B")
            model_path: directory containing model weights
            dtype: target dtype for model weights
            quantize_transformer: quantize the main transformer via mmgp
            mmgp_profile: mmgp profile number (default: balanced=2)
        """
        config = _HANDLER_MAP.get(model_type)
        if config is None:
            raise ValueError(f"No handler config for {model_type}")

        self._ensure_vendor()
        handler = self._import_handler(config["handler"])
        profile = mmgp_profile or self._mmgp_profile

        # Wan2GP's file locator needs the parent dir for shared deps
        from shared.utils import files_locator as fl
        checkpoint_root = model_path.parent if model_path.parent.name != "wan2gp" else model_path
        fl.set_checkpoints_paths([str(checkpoint_root)])

        # Find model files
        model_files = list(model_path.rglob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensors found for {model_type} in {model_path}")

        text_encoder_filename = None
        te_dir = model_path / "text_encoder"
        if te_dir.is_dir():
            te_files = list(te_dir.rglob("*.safetensors"))
            if te_files:
                text_encoder_filename = str(te_files[0])

        # Build model definition
        model_def = self._build_model_def(handler, config["base_model_type"], model_path)

        logger.info(
            "Loading Wan2GP %s from %s (profile=%d)",
            model_type, model_path, profile,
        )

        torch.set_default_device("cpu")

        model_obj, pipe = handler.load_model(
            [str(f) for f in model_files],
            config["model_type"],
            config["base_model_type"],
            model_def,
            quantizeTransformer=quantize_transformer,
            text_encoder_quantization="int8",
            dtype=dtype,
            VAE_dtype=torch.float32,
            text_encoder_filename=text_encoder_filename,
            profile=profile,
        )

        # Wrap the model object + pipe in a Wan2GPPipeline adapter
        pipeline = Wan2GPPipeline(
            model=model_obj,
            model_type=model_type,
            config=config,
            defaults=dict(VARIANTS[model_type].defaults),
        )

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info(
            "Wan2GP %s loaded: pipe_keys=%s VRAM=%.0fMB",
            model_type, list(pipe.keys()), vram,
        )

        return LoadResult(
            pipeline=pipeline,
            pipe=pipe,
            co_tenants={},
        )

    # ── Vendor setup ──────────────────────────────────────────────────────────

    def _ensure_vendor(self):
        if self._vendor_loaded:
            return
        vendor = str(WAN2GP_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        os.environ["WAN2GP_ROOT"] = vendor
        # Shim: Wan2GP vendor imports QuantizedCacheConfig which was removed in
        # transformers >= 4.50. Provide a simple stand-in.
        import transformers.cache_utils as _tcu
        if not hasattr(_tcu, "QuantizedCacheConfig"):
            _tcu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {
                "__init__": lambda self, **kw: None,
                "__getattr__": lambda self, name: None,
            })
        self._vendor_loaded = True

    def _import_handler(self, handler_path: str):
        import importlib
        mod = importlib.import_module(handler_path)
        return mod.family_handler

    def _build_model_def(self, handler, base_model_type: str, model_path: Path) -> dict:
        text_encoder_folder = None
        te_dir = model_path / "text_encoder"
        if te_dir.is_dir():
            text_encoder_folder = str(te_dir)

        text_encoder_urls = None
        te_urls_file = model_path / "text_encoder_urls.json"
        if te_urls_file.exists():
            text_encoder_urls = json.loads(te_urls_file.read_text())

        base = {
            "text_encoder_folder": text_encoder_folder,
            "text_encoder_URLs": text_encoder_urls,
            "profiles_dir": [base_model_type],
            "group": base_model_type,
        }

        enriched = handler.query_model_def(base_model_type, base)
        return enriched


class Wan2GPPipeline:
    """Adapts a Wan2GP model object into a callable pipeline for ModelExecutor.

    The Wan2GP model's .generate() method already handles mmgp integration
    internally (it calls mmgp to load modules on demand). We just translate
    the Model Engine payload format into Wan2GP's generate() kwargs.
    """

    def __init__(self, model, model_type: str, config: dict, defaults: dict):
        self.model = model
        self.model_type = model_type
        self.config = config
        self.defaults = defaults

    def __call__(self, payload: dict) -> dict:
        """Generate from the model. dict in, dict out."""
        return self.generate(payload)

    def generate(self, payload: dict) -> dict:
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required")

        seed = int(payload.get("seed", -1))
        steps = int(payload.get("steps", self.defaults.get("steps", 50)))
        guidance = float(payload.get("guidance", self.defaults.get("guidance", 5.0)))
        width = int(payload.get("width", self.defaults.get("width", 1280)))
        height = int(payload.get("height", self.defaults.get("height", 720)))
        frames = int(payload.get("frames", self.defaults.get("frames", 81)))

        gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
        if seed >= 0:
            gen.manual_seed(seed)

        logger.info(
            "Wan2GP generate: model=%s prompt=%r steps=%d frames=%d",
            self.model_type, prompt[:80], steps, frames,
        )

        kwargs = {
            "input_prompt": prompt,
            "width": width,
            "height": height,
            "frame_num": frames,
            "sampling_steps": steps,
            "guide_scale": guidance,
            "seed": gen.initial_seed() if seed >= 0 else -1,
        }

        base_model_type = self.config["base_model_type"]

        if base_model_type in ("i2v", "i2v_2_2"):
            import io
            from PIL import Image
            image_b64 = payload.get("image_b64", "")
            if not image_b64:
                raise ValueError("image_b64 is required for i2v models")
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            kwargs["image_start"] = img

        output = self.model.generate(**kwargs)

        from mmgp import offload
        offload.clear_caches()

        # Process output
        if isinstance(output, dict):
            frames_tensor = output.get("x")
            audio = output.get("audio")
        elif isinstance(output, torch.Tensor):
            frames_tensor = output
            audio = None
        else:
            frames_tensor = output
            audio = None

        if frames_tensor is None:
            raise RuntimeError("Model returned no frames")

        import numpy as np
        frames_np = frames_tensor.cpu().numpy() if isinstance(frames_tensor, torch.Tensor) else frames_tensor
        if frames_np.dtype != np.uint8:
            frames_np = ((frames_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        if len(frames_np.shape) == 4 and frames_np.shape[0] > 1:
            import imageio
            fps = payload.get("fps", self.defaults.get("fps", 16))
            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            for f in frames_np:
                writer.append_data(f)
            writer.close()
        else:
            from PIL import Image as PILImage
            img = frames_np[0] if len(frames_np.shape) == 4 else frames_np
            PILImage.fromarray(img).save(tmp_path, format="PNG")

        with open(tmp_path, "rb") as f:
            media_bytes = f.read()
        os.unlink(tmp_path)

        media_type = "video/mp4" if frames_np.ndim == 4 and frames_np.shape[0] > 1 else "image/png"

        result = {
            "media_type": media_type,
            "data": media_bytes,
            "model": self.model_type,
        }

        if audio is not None:
            import soundfile as sf
            import io as audio_io
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            audio_buf = audio_io.BytesIO()
            sf.write(audio_buf, audio_np, 24000, format="WAV")
            result["audio_b64"] = base64.b64encode(audio_buf.getvalue()).decode()

        return result
