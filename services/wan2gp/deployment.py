"""Wan2GP Pool — unified adapter for all Wan2GP model families.

Runs Wan2GP's model implementations in-process with mmgp VRAM management.
Single deployment handles 90+ model variants via model routing.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig
from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

WAN2GP_VENDOR = Path(__file__).parents[2] / "vendor" / "wan2gp"

V2V_MODELS = {
    "wan/t2v-14B": {
        "handler": "models.wan.wan_handler",
        "model_type": "t2v",
        "base_model_type": "t2v",
        "vram_gb": 14,
        "defaults": {"width": 1280, "height": 720, "frames": 81, "steps": 50, "guidance": 5.0},
    },
    "wan/i2v-14B": {
        "handler": "models.wan.wan_handler",
        "model_type": "i2v",
        "base_model_type": "i2v",
        "vram_gb": 14,
        "defaults": {"width": 832, "height": 480, "frames": 81, "steps": 50, "guidance": 5.0},
    },
    "hunyuan/t2v": {
        "handler": "models.hyvideo.hunyuan_handler",
        "model_type": "hunyuan",
        "base_model_type": "hunyuan",
        "vram_gb": 12,
        "defaults": {"width": 848, "height": 480, "frames": 125, "steps": 30, "guidance": 6.0},
    },
    "flux/t2i": {
        "handler": "models.flux.flux_handler",
        "model_type": "flux",
        "base_model_type": "flux",
        "vram_gb": 8,
        "defaults": {"width": 1024, "height": 1024, "steps": 28, "guidance": 3.5},
    },
    "ace_step/v1_5": {
        "handler": "models.TTS.ace_step_handler",
        "model_type": "ace_step_v1_5",
        "base_model_type": "ace_step_v1_5",
        "vram_gb": 8,
        "defaults": {"duration": 30, "steps": 8, "guidance": 2.5},
    },
    "index_tts/v2": {
        "handler": "models.TTS.index_tts2_handler",
        "model_type": "index_tts2",
        "base_model_type": "index_tts2",
        "vram_gb": 6,
        "defaults": {},
    },
}

MMGP_PROFILES = {
    "max_speed": 1,    # 24GB+ VRAM, keep everything resident
    "balanced": 2,     # 12GB+ VRAM, swap when needed
    "low_vram": 4,     # 12GB VRAM, aggressive offload
    "minimum": 5,      # 10GB VRAM, maximum offload
}


class Wan2GPPool(BaseGPUDeployment):
    """Wan2GP model pool — 90+ variants, mmgp-managed VRAM.

    Runs inside the MasterRouter process (single CUDA context).
    Each variant is loaded on-demand and stays in mmgp's offload pool.

    Standard lifecycle via BaseGPUDeployment:
        pool._load("wan/t2v-14B")      # load + mmgp profile
        pool._unload()                  # evict all models, flush caches
        pool(model_key, payload)        # generate
    """
    vram_mb = 0  # Pool manages VRAM mmgp nternally — no Governor lease needed
    _service_name = "wan2gp"

    def __init__(self):
        super().__init__()
        self._models: dict[str, dict] = {}
        self._vendor_loaded = False

    def _ensure_vendor(self):
        if self._vendor_loaded:
            return
        vendor = str(WAN2GP_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        os.environ["WAN2GP_ROOT"] = vendor
        self._vendor_loaded = True

    def _get_handler(self, model_key: str):
        info = V2V_MODELS.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(V2V_MODELS.keys())}")
        self._ensure_vendor()
        import importlib
        mod = importlib.import_module(info["handler"])
        handler = mod.family_handler
        return handler, info

    def _build_model_def(self, handler, base_model_type: str, model_path: Path) -> dict:
        """Build a proper model_def dict using the handler's query_model_def().

        This mirrors how wgp.py enriches the raw model registry entry with
        per-variant configuration flags (i2v_class, vace_class, profiles_dir, etc.).
        """
        text_encoder_folder = None
        te_dir = model_path / "text_encoder"
        if te_dir.is_dir():
            text_encoder_folder = str(te_dir)

        text_encoder_urls = None
        te_urls_file = model_path / "text_encoder_urls.json"
        if te_urls_file.exists():
            import json
            text_encoder_urls = json.loads(te_urls_file.read_text())

        base = {
            "text_encoder_folder": text_encoder_folder,
            "text_encoder_URLs": text_encoder_urls,
            "profiles_dir": [model_type],
            "group": base_model_type,
        }

        enriched = handler.query_model_def(base_model_type, base)
        return enriched

    def _load(self, model_name: str) -> None:
        """Load a model variant into the pool. mmgp manages VRAM.

        If another model is already loaded, mmgp handles eviction
        at the subcomponent level — no explicit unload needed.
        """
        import gc
        import torch

        handler, info = self._get_handler(model_name)
        model_type = info["model_type"]
        base_model_type = info["base_model_type"]

        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()

        model_key_safe = model_name.replace("/", "-")
        model_path = registry.get_path("wan2gp", model_key_safe) if "wan2gp" in registry.data else None
        if not model_path or not model_path.is_dir():
            logger.warning("Model path not found in registry for %s, using default ckpts", model_name)
            model_path = Path(cfg.models_root) / "wan2gp" / model_type

        model_filename = list(model_path.rglob("*.safetensors"))
        if not model_filename:
            raise FileNotFoundError(f"No safetensors found for {model_name} in {model_path}")

        text_encoder_path = model_path / "text_encoder"
        text_encoder_filename = None
        if text_encoder_path.is_dir():
            te_files = list(text_encoder_path.rglob("*.safetensors"))
            if te_files:
                text_encoder_filename = str(te_files[0])

        logger.info("Loading %s from %s (mmgp profile=%s)", model_name, model_path, "balanced")
        torch.set_default_device("cpu")

        model_def = self._build_model_def(handler, base_model_type, model_path)

        wan_model, pipe = handler.load_model(
            [str(f) for f in model_filename],
            model_type,
            base_model_type,
            model_def,
            quantizeTransformer=True,
            text_encoder_quantization="int8",
            dtype=torch.bfloat16,
            VAE_dtype=torch.float32,
            text_encoder_filename=text_encoder_filename,
            profile=MMGP_PROFILES["balanced"],
        )

        from mmgp import offload

        profile_no = MMGP_PROFILES["balanced"]
        budgets = {}
        if profile_no in (2, 4, 5):
            budgets["transformer"] = 250
            budgets["text_encoder"] = 250
            budgets["*"] = 3000

        offload.profile(
            pipe,
            profile_no=profile_no,
            quantizeTransformer=False,
            budgets=budgets,
            loras=[],
            perc_reserved_mem_max=0.5,
            vram_safety_coefficient=0.9,
            coTenantsMap={},
        )

        self._models[model_name] = {
            "model": wan_model,
            "pipe": pipe,
            "info": info,
            "loaded_at": time.time(),
        }
        self.model = True
        self.model_name = model_name

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (VRAM=%.0fMB, active models=%d)", model_name, vram, len(self._models))
        torch.cuda.empty_cache()
        gc.collect()

    def unload_model(self, model_key: str | None = None) -> None:
        """Unload a specific model or all models from the pool."""
        from mmgp import offload

        if model_key and model_key in self._models:
            entry = self._models.pop(model_key)
            del entry["model"]
            del entry["pipe"]
            logger.info("Unloaded %s", model_key)
        elif model_key is None:
            for k in list(self._models):
                self.unload_model(k)

        if not self._models:
            self.model = None
            self.model_name = None

        offload.flush_torch_caches()
        gc.collect()
        torch.cuda.empty_cache()

    def _unload(self) -> None:
        self.unload_model()
        super()._unload()

    async def __call__(self, request):
        if request.method == "GET":
            return {
                "status": "ok",
                "active_models": list(self._models.keys()),
                "loaded": bool(self._models),
            }

        start = time.perf_counter()

        try:
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" in content_type:
                form = await request.form()
                body = dict(form)
                extracted = {}
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)
                body["input"] = tnap_req.input

            import asyncio

            model_key = body.get("model", body.get("model_name", "wan/t2v-14B"))
            if model_key not in self._models:
                await asyncio.to_thread(self._load, model_key)

            payload = {**body, **extracted}
            result = await asyncio.to_thread(self._infer, model_key, payload)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                    extra_metrics=result.get("extra", {}),
                )
            )
        except Exception as e:
            logger.error("Wan2GP inference failed: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    def _infer(self, model_key: str, payload: dict) -> dict:
        entry = self._models.get(model_key)
        if entry is None:
            raise RuntimeError(f"Model {model_key} not loaded")

        model = entry["model"]
        info = entry["info"]
        defaults = info.get("defaults", {})
        base_model_type = info.get("base_model_type", "")

        from mmgp import offload

        prompt = payload.get("prompt", payload.get("input", {}).get("prompt", ""))
        if not prompt:
            raise ValueError("prompt is required")

        seed = int(payload.get("seed", payload.get("input", {}).get("seed", -1)))
        steps = int(payload.get("steps", payload.get("input", {}).get("steps", defaults.get("steps", 50))))
        guidance = float(payload.get("guidance", payload.get("input", {}).get("guidance", defaults.get("guidance", 5.0))))
        width = int(defaults.get("width", 1280))
        height = int(defaults.get("height", 720))
        frames = int(defaults.get("frames", 81))

        gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
        if seed >= 0:
            gen.manual_seed(seed)

        logger.info("Wan2GP %s generate: prompt=%r seed=%d steps=%d", model_key, prompt[:80], seed, steps)

        if base_model_type in ("t2v", "t2v_2_2"):
            output = model.generate(
                input_prompt=prompt,
                width=width,
                height=height,
                frame_num=frames,
                sampling_steps=steps,
                guide_scale=guidance,
                seed=gen.initial_seed() if seed >= 0 else -1,
            )
        elif base_model_type in ("i2v", "i2v_2_2"):
            import base64
            import io
            from PIL import Image

            image_b64 = payload.get("image_b64", payload.get("input", {}).get("image_b64", ""))
            if not image_b64:
                raise ValueError("image_b64 is required for i2v models")

            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            output = model.generate(
                input_prompt=prompt,
                image_start=img,
                width=width,
                height=height,
                frame_num=frames,
                sampling_steps=steps,
                guide_scale=guidance,
                seed=gen.initial_seed() if seed >= 0 else -1,
            )
        else:
            raise ValueError(f"Unsupported base_model_type: {base_model_type}")

        offload.clear_caches()

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

        frames_np = frames_tensor.cpu().numpy() if isinstance(frames_tensor, torch.Tensor) else frames_tensor

        if frames_np.dtype != np.uint8:
            frames_np = ((frames_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        if len(frames_np.shape) == 4 and frames_np.shape[0] > 1:
            import imageio

            fps = payload.get("fps", defaults.get("fps", 16))
            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            for f in frames_np:
                writer.append_data(f)
            writer.close()
        else:
            from PIL import Image as PILImage
            img = frames_np[0] if len(frames_np.shape) == 4 else frames_np
            PILImage.fromarray(img).save(tmp_path, format="PNG")

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        extra = {}
        if audio is not None:
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            import io as audio_io
            import soundfile as sf
            audio_buf = audio_io.BytesIO()
            sf.write(audio_buf, audio_np, 24000, format="WAV")
            extra["audio_b64"] = base64.b64encode(audio_buf.getvalue()).decode()

        media_type = "video/mp4" if frames_np.ndim == 4 and frames_np.shape[0] > 1 else "image/png"
        import base64 as b64
        return {
            "data": video_bytes,
            "media_type": media_type,
            "extra": {"model": model_key, **extra},
        }


class Wan2GPService(ForgeService):
    """Forge adapter for Wan2GP — self-managed VRAM via mmgp.

    The Forge trusts this service to handle its own VRAM budget (vram_mb=0)
    because mmgp offloading manages multiple model variants in a shared pool.
    """

    vram_mb = 0
    service_name = "wan2gp"
    default_model = "wan/t2v-14B"

    def __init__(self):
        super().__init__()
        self._models: dict[str, dict] = {}
        self._vendor_loaded = False

    def load(self, model_name: str | None = None) -> None:
        model_name = model_name or self.default_model
        handler, info = self._get_handler(model_name)
        model_type = info["model_type"]
        base_model_type = info["base_model_type"]

        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()

        model_key_safe = model_name.replace("/", "-")
        model_path = registry.get_path("wan2gp", model_key_safe) if "wan2gp" in registry.data else None
        if not model_path or not model_path.is_dir():
            logger.warning("Model path not found in registry for %s, using default ckpts", model_name)
            model_path = Path(cfg.models_root) / "wan2gp" / model_type

        model_filename = list(model_path.rglob("*.safetensors"))
        if not model_filename:
            raise FileNotFoundError(f"No safetensors found for {model_name} in {model_path}")

        text_encoder_path = model_path / "text_encoder"
        text_encoder_filename = None
        if text_encoder_path.is_dir():
            te_files = list(text_encoder_path.rglob("*.safetensors"))
            if te_files:
                text_encoder_filename = str(te_files[0])

        logger.info("Loading %s from %s (mmgp profile=%s)", model_name, model_path, "balanced")
        torch.set_default_device("cpu")

        model_def = self._build_model_def(handler, base_model_type, model_path)

        wan_model, pipe = handler.load_model(
            [str(f) for f in model_filename],
            model_type,
            base_model_type,
            model_def,
            quantizeTransformer=True,
            text_encoder_quantization="int8",
            dtype=torch.bfloat16,
            VAE_dtype=torch.float32,
            text_encoder_filename=text_encoder_filename,
            profile=MMGP_PROFILES["balanced"],
        )

        from mmgp import offload

        profile_no = MMGP_PROFILES["balanced"]
        budgets = {}
        if profile_no in (2, 4, 5):
            budgets["transformer"] = 250
            budgets["text_encoder"] = 250
            budgets["*"] = 3000

        offload.profile(
            pipe,
            profile_no=profile_no,
            quantizeTransformer=False,
            budgets=budgets,
            loras=[],
            perc_reserved_mem_max=0.5,
            vram_safety_coefficient=0.9,
            coTenantsMap={},
        )

        self._models[model_name] = {
            "model": wan_model,
            "pipe": pipe,
            "info": info,
            "loaded_at": time.time(),
        }
        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (VRAM=%.0fMB, active models=%d)", model_name, vram, len(self._models))
        torch.cuda.empty_cache()
        gc.collect()

    def unload(self) -> None:
        self._models.clear()
        self._loaded = False
        super().unload()

    def infer(self, payload: dict) -> dict:
        model_key = payload.get("model", self.default_model)
        if model_key not in self._models:
            self.load(model_key)
        result = self._infer(model_key, payload)
        import base64
        return {
            "status": "ok",
            "data": base64.b64encode(result["data"]).decode(),
            "media_type": result["media_type"],
        }

    # Re-use helper methods from Wan2GPPool
    def _ensure_vendor(self):
        if self._vendor_loaded:
            return
        vendor = str(WAN2GP_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        os.environ["WAN2GP_ROOT"] = vendor
        self._vendor_loaded = True

    def _get_handler(self, model_key: str):
        info = V2V_MODELS.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(V2V_MODELS.keys())}")
        self._ensure_vendor()
        import importlib
        mod = importlib.import_module(info["handler"])
        handler = mod.family_handler
        return handler, info

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
