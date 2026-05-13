"""Wan2GP Service — unified multi-model pool, mmgp-managed VRAM.

ForgeService-compatible: runs inside the Forge (num_gpus: 1.0).
ALL models (vendor + custom nn.Module) load as variants through
a single service. mmgp manages VRAM via per-subcomponent offloading.

Supports two engine types:
  vendor       — upstream Wan2GP family_handler (wan t2v/i2v, hunyuan, flux)
  model_engine — our custom nn.Module decomposition (anigen, trellis,
                 hy_motion, moss, see_through, faster_qwen3_tts,
                 vibevoice_asr, vibevoice_tts)

Lifecycle via ForgeService:
    load(model_name)   — load variant + configure mmgp profile
    unload()           — evict all models, flush mmgp + CUDA caches
    infer(payload)     — generate, return dict with b64 data
"""
from __future__ import annotations

import base64
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

WAN2GP_VENDOR = Path(__file__).parents[2] / "vendor" / "wan2gp"

# ─── Variant Registry ─────────────────────────────────────────────────────────

V2V_MODELS = {
    # ── Vendor: upstream Wan2GP family_handlers ────────────────────────────
    "wan/t2v-14B": {
        "engine": "vendor",
        "handler": "models.wan.wan_handler",
        "model_type": "t2v",
        "base_model_type": "t2v",
        "vram_gb": 14,
        "defaults": {"width": 1280, "height": 720, "frames": 81, "steps": 50, "guidance": 5.0},
    },
    "wan/i2v-14B": {
        "engine": "vendor",
        "handler": "models.wan.wan_handler",
        "model_type": "i2v",
        "base_model_type": "i2v",
        "vram_gb": 14,
        "defaults": {"width": 832, "height": 480, "frames": 81, "steps": 50, "guidance": 5.0},
    },
    "hunyuan/t2v": {
        "engine": "vendor",
        "handler": "models.hyvideo.hunyuan_handler",
        "model_type": "hunyuan",
        "base_model_type": "hunyuan",
        "vram_gb": 12,
        "defaults": {"width": 848, "height": 480, "frames": 125, "steps": 30, "guidance": 6.0},
    },
    "flux/t2i": {
        "engine": "vendor",
        "handler": "models.flux.flux_handler",
        "model_type": "flux",
        "base_model_type": "flux",
        "vram_gb": 8,
        "defaults": {"width": 1024, "height": 1024, "steps": 28, "guidance": 3.5},
    },
    "ace_step/v1_5": {
        "engine": "vendor",
        "handler": "models.TTS.ace_step_handler",
        "model_type": "ace_step_v1_5",
        "base_model_type": "ace_step_v1_5",
        "vram_gb": 8,
        "defaults": {"duration": 30, "steps": 8, "guidance": 2.5},
    },
    "index_tts/v2": {
        "engine": "vendor",
        "handler": "models.TTS.index_tts2_handler",
        "model_type": "index_tts2",
        "base_model_type": "index_tts2",
        "vram_gb": 6,
        "defaults": {},
        "blocked": True,
        "blocked_reason": "Vendored transformers_generation_utils.py incompatible with transformers>=4.55",
    },

    # ── Model Engine: custom nn.Module decomposition ────────────────────────
    "anigen": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.anigen",
        "handler_cls": "AniGenHandler",
        "registry_path": ("3d", "anigen"),
        "vram_gb": 12,
        "defaults": {"ss_steps": 25, "slat_steps": 25, "cfg": 3.5},
    },
    "trellis": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.trellis",
        "handler_cls": "TrellisHandler",
        "registry_path": ("3d", "trellis"),
        "vram_gb": 10,
        "defaults": {"steps": 50, "guidance": 3.0},
    },
    "hy_motion": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.hy_motion",
        "handler_cls": "HYMotionHandler",
        "registry_path": ("motion", "hy-motion-1.0"),
        "vram_gb": 6,
        "defaults": {"steps": 50, "cfg": 2.0},
    },
    "moss_soundeffect": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.moss",
        "handler_cls": "MossHandler",
        "registry_path": ("audio", "moss-soundeffect"),
        "vram_gb": 16,
        "defaults": {"max_tokens": 4096},
    },
    "see_through": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.see_through",
        "handler_cls": "SeeThroughHandler",
        "registry_path": None,  # uses vendored code at /opt/seethrough
        "vram_gb": 6,
        "defaults": {"resolution": 1280, "steps": 30},
    },
    "faster_qwen3_tts": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.faster_qwen3_tts",
        "handler_cls": "FasterQwen3TTSHandler",
        "registry_path": ("tts", "qwen3-tts-12hz-1.7b-customvoice"),
        "vram_gb": 6,
        "defaults": {"voice": "Aiden", "language": "English"},
    },
    "vibevoice_asr": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.vibevoice_asr",
        "handler_cls": "VibeVoiceASRHandler",
        "registry_path": ("asr", "vibevoice-asr"),
        "vram_gb": 16,
        "defaults": {"language": "english", "max_tokens": 512},
    },
    "vibevoice_tts": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.vibevoice_tts",
        "handler_cls": "VibeVoiceTTSHandler",
        "registry_path": ("tts", "vibevoice"),
        "vram_gb": 18,
        "defaults": {"language": "English", "max_tokens": 4096},
    },

    # ── CPU services (no mmgp offloading needed) ───────────────────────────
    "kokoro": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.kokoro",
        "handler_cls": "KokoroHandler",
        "registry_path": ("tts", "kokoro"),
        "vram_gb": 0,
        "defaults": {"voice": "af_bella", "speed": 1.0},
    },
    "espeak": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.espeak",
        "handler_cls": "EspeakHandler",
        "registry_path": None,  # no model files — subprocess binary
        "vram_gb": 0,
        "defaults": {"voice": "en", "speed": 175, "pitch": 50},
    },
    "faster_whisper": {
        "engine": "model_engine",
        "handler": "services.model_engine.handlers.faster_whisper",
        "handler_cls": "FasterWhisperHandler",
        "registry_path": ("asr", "faster-whisper"),
        "vram_gb": 0,
        "defaults": {"language": None, "beam_size": 5},
    },

}

MMGP_PROFILES = {
    "max_speed": 1,
    "balanced": 2,
    "low_vram": 4,
    "minimum": 5,
}


class Wan2GPService(ForgeService):
    """Forge adapter for Wan2GP — self-managed VRAM via mmgp.

    Supports both vendor (upstream Wan2GP family_handler) and model_engine
    (custom nn.Module decomposition) variants.
    """

    vram_mb = 0
    service_name = "wan2gp"
    default_model = "wan/t2v-14B"

    def __init__(self):
        super().__init__()
        self._models: dict[str, dict] = {}
        self._vendor_loaded = False

    # ── ForgeService lifecycle ────────────────────────────────────────────────

    def load(self, model_name: str | None = None) -> None:
        model_name = model_name or self.default_model
        if model_name in self._models:
            logger.debug("Wan2GP: %s already loaded", model_name)
            return

        handler, info = self._get_handler(model_name)
        if info.get("blocked"):
            raise RuntimeError(
                f"Wan2GP model '{model_name}' is blocked: {info.get('blocked_reason', 'unknown')}"
            )

        engine = info.get("engine", "vendor")

        if engine == "model_engine":
            self._load_model_engine(model_name, handler, info)
        else:
            self._load_vendor(model_name, handler, info)

        self.model_name = model_name
        self._loaded = True

    def _load_vendor(self, model_name, handler, info):
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()
        model_type = info["model_type"]
        base_model_type = info["base_model_type"]

        model_key_safe = model_name.replace("/", "-")
        model_path = registry.get_path("wan2gp", model_key_safe) if "wan2gp" in registry.data else None
        if not model_path or not model_path.is_dir():
            logger.warning("Model path not found in registry for %s, using default ckpts", model_name)
            model_path = Path(cfg.models_root) / "wan2gp" / model_type

        checkpoint_root = model_path.parent if model_path.parent.name == "wan2gp" else model_path
        from shared.utils import files_locator as fl
        fl.set_checkpoints_paths([str(checkpoint_root)])

        model_files = list(model_path.rglob("*.safetensors"))
        if not model_files and model_type.endswith("tts"):
            parent_files = list(model_path.parent.glob("*.safetensors"))
            model_files = [f for f in parent_files if model_type in f.name.lower() or "index_tts2" in f.name.lower()]
        if not model_files:
            raise FileNotFoundError(f"No safetensors found for {model_name} in {model_path}")

        text_encoder_filename = None
        te_dir = model_path / "text_encoder"
        if te_dir.is_dir():
            te_files = list(te_dir.rglob("*.safetensors"))
            if te_files:
                text_encoder_filename = str(te_files[0])

        logger.info("Loading %s from %s (vendor, mmgp=balanced)", model_name, model_path)
        torch.set_default_device("cpu")

        model_def = self._build_model_def(handler, base_model_type, model_path)

        wan_model, pipe = handler.load_model(
            [str(f) for f in model_files],
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

        budgets = {"transformer": 250, "text_encoder": 250, "*": 3000}
        offload.profile(
            pipe,
            profile_no=MMGP_PROFILES["balanced"],
            quantizeTransformer=False,
            budgets=budgets,
            loras=[],
            perc_reserved_mem_max=0.5,
            vram_safety_coefficient=0.9,
            coTenantsMap={},
        )

        self._models[model_name] = {
            "engine": "vendor",
            "model": wan_model,
            "pipe": pipe,
            "info": info,
            "loaded_at": time.time(),
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (VRAM=%.0fMB, active=%d)", model_name, vram, len(self._models))
        torch.cuda.empty_cache()
        gc.collect()

    def _load_model_engine(self, model_name, handler, info):
        from registry.config import Config

        cfg = Config()
        models_root = Path(cfg.models_root)

        # Resolve model path: registry → handler fallback → see_through fixed path
        registry_path = info.get("registry_path")
        model_path = None

        if registry_path:
            try:
                from registry.models import ModelRegistry
                registry = ModelRegistry()
                model_path = registry.get_path(*registry_path)
            except (KeyError, FileNotFoundError):
                model_path = None

        if model_path is None or not (model_path := Path(model_path)).is_dir():
            model_path = handler.resolve_path(model_name, models_root)

        if not model_path.is_dir():
            # Last resort: see_through uses vendored code at fixed location
            if registry_path is None:
                model_path = Path("/opt/seethrough")
            if not model_path.is_dir():
                raise FileNotFoundError(f"Model path not found for {model_name}: {model_path}")

        model_type = info.get("model_type", model_name)

        logger.info("Loading %s (model_engine, nn.Module) from %s", model_name, model_path)
        torch.set_default_device("cpu")

        load_result = handler.load_model(model_type, model_path=model_path, dtype=torch.bfloat16)

        orchestrator = load_result.pipeline
        pipe = load_result.pipe
        co_tenants = load_result.co_tenants or {}

        # Only run mmgp offload if there are nn.Modules to manage
        if pipe:
            from mmgp import offload

            budgets = {"transformer": 250, "text_encoder": 250, "*": 3000}
            offload.profile(
                pipe,
                profile_no=MMGP_PROFILES["balanced"],
                quantizeTransformer=False,
                budgets=budgets,
                loras=[],
                perc_reserved_mem_max=0.5,
                vram_safety_coefficient=0.9,
                coTenantsMap=co_tenants,
            )

        self._models[model_name] = {
            "engine": "model_engine",
            "orchestrator": orchestrator,
            "pipe": pipe,
            "info": info,
            "loaded_at": time.time(),
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("Loaded %s (VRAM=%.0fMB, active=%d)", model_name, vram, len(self._models))
        torch.cuda.empty_cache()
        gc.collect()

    def unload(self) -> None:
        self._models.clear()
        self._loaded = False
        try:
            from mmgp import offload
            offload.flush_torch_caches()
        except ImportError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    def infer(self, payload: dict) -> dict:
        model_key = payload.get("model", self.default_model)
        if model_key not in self._models:
            self.load(model_key)

        entry = self._models.get(model_key)
        if entry is None:
            return {"status": "error", "error": f"Model {model_key} not loaded"}

        try:
            if entry.get("engine") == "model_engine":
                result = entry["orchestrator"](payload)
            else:
                gen = self._do_generate(entry, payload)
                result = {
                    "status": "ok",
                    "data": base64.b64encode(gen["data"]).decode(),
                    "media_type": gen["media_type"],
                }

            result["model"] = model_key
            return result
        except Exception as e:
            logger.error("Wan2GP %s inference failed: %s", model_key, e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def _do_generate(self, entry: dict, payload: dict) -> dict:
        """Vendor path: core generation logic. Returns dict with 'data' (bytes) and 'media_type'."""
        model = entry["model"]
        info = entry["info"]
        defaults = info.get("defaults", {})
        base_model_type = info.get("base_model_type", "")

        from mmgp import offload

        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required")

        seed = int(payload.get("seed", -1))
        steps = int(payload.get("steps", defaults.get("steps", 50)))
        guidance = float(payload.get("guidance", defaults.get("guidance", 5.0)))
        width = int(defaults.get("width", 1280))
        height = int(defaults.get("height", 720))
        frames = int(defaults.get("frames", 81))

        gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
        if seed >= 0:
            gen.manual_seed(seed)

        logger.info("Wan2GP generate: prompt=%r seed=%d steps=%d", prompt[:80], seed, steps)

        kwargs = {
            "input_prompt": prompt,
            "width": width,
            "height": height,
            "frame_num": frames,
            "sampling_steps": steps,
            "guide_scale": guidance,
            "seed": gen.initial_seed() if seed >= 0 else -1,
        }

        if base_model_type in ("t2v", "t2v_2_2"):
            output = model.generate(**kwargs)
        elif base_model_type in ("i2v", "i2v_2_2"):
            image_b64 = payload.get("image_b64", "")
            if not image_b64:
                raise ValueError("image_b64 is required for i2v models")
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            kwargs["image_start"] = img
            output = model.generate(**kwargs)
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
            import soundfile as sf
            import io as audio_io
            audio_buf = audio_io.BytesIO()
            sf.write(audio_buf, audio_np, 24000, format="WAV")
            extra["audio_b64"] = base64.b64encode(audio_buf.getvalue()).decode()

        media_type = "video/mp4" if frames_np.ndim == 4 and frames_np.shape[0] > 1 else "image/png"
        return {"data": video_bytes, "media_type": media_type}

    # ── Vendor loading helpers ─────────────────────────────────────────────────

    def _ensure_vendor(self):
        if self._vendor_loaded:
            return
        vendor = str(WAN2GP_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        os.environ["WAN2GP_ROOT"] = vendor
        import transformers.cache_utils as _tcu
        if not hasattr(_tcu, "QuantizedCacheConfig"):
            _tcu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {
                "__init__": lambda self, **kw: None,
                "__getattr__": lambda self, name: None,
            })
        self._vendor_loaded = True

    def _get_handler(self, model_key: str):
        info = V2V_MODELS.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model: {model_key}. Available: {sorted(V2V_MODELS.keys())}")

        engine = info.get("engine", "vendor")
        if engine == "model_engine":
            import importlib
            mod = importlib.import_module(info["handler"])
            handler_cls = getattr(mod, info["handler_cls"])
            handler = handler_cls()
            return handler, info

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
