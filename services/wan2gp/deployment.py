"""Wan2GP Service — dynamic model registry, mmgp-managed VRAM.

All GPU models (vendor + model_engine) coexist under one mmgp profile.
Payload passthrough: security-allowlisted kwargs forwarded to model.generate().

Architecture:
    - 19 vendor handlers auto-discovered via query_supported_types()
    - 12 model_engine handlers statically registered (our code)
    - mmgp wraps all nn.Modules for unified GPU/CPU/RAM management
    - CPU models (espeak, kokoro, faster_whisper) use empty pipe dict
    - No ForgeService dependency — standalone deployment
"""
from __future__ import annotations

import base64
import gc
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

WAN2GP_VENDOR = Path(__file__).parents[2] / "vendor" / "wan2gp"

# ─── Vendor Handler Registry ──────────────────────────────────────────────────

VENDOR_HANDLERS = [
    "models.wan.wan_handler",
    "models.wan.ovi_handler",
    "models.wan.df_handler",
    "models.hyvideo.hunyuan_handler",
    "models.ltx_video.ltxv_handler",
    "models.ltx2.ltx2_handler",
    "models.longcat.longcat_handler",
    "models.flux.flux_handler",
    "models.qwen.qwen_handler",
    "models.kandinsky5.kandinsky_handler",
    "models.z_image.z_image_handler",
    "models.magi_human.magi_human_handler",
    "models.TTS.chatterbox_handler",
    "models.TTS.qwen3_handler",
    "models.TTS.yue_handler",
    "models.TTS.heartmula_handler",
    "models.TTS.kugelaudio_handler",
    "models.TTS.index_tts2_handler",
]

# ─── Model Engine Entries (our code) ──────────────────────────────────────────

MODEL_ENGINE_ENTRIES = [
    {
        "name": "ace_step",
        "handler": "services.model_engine.handlers.ace_step",
        "handler_cls": "AceStepHandler",
        "registry_path": ("audio", "acestep"),
        "vram_gb": 7,
        "defaults": {"num_inference_steps": 8, "alt_guidance_scale": 2.5,
                     "duration_seconds": 30, "temperature": 0.85, "top_p": 0.9},
        "key_map": {
            "steps": "num_inference_steps",
            "cover_strength": "audio_cover_strength",
            "ref_audio": "reference_audio",
        },
    },
    {
        "name": "anigen",
        "handler": "services.model_engine.handlers.anigen",
        "handler_cls": "AniGenHandler",
        "registry_path": ("3d", "anigen"),
        "vram_gb": 12,
        "defaults": {"ss_steps": 25, "slat_steps": 25, "cfg": 3.5},
    },
    {
        "name": "trellis",
        "handler": "services.model_engine.handlers.trellis",
        "handler_cls": "TrellisHandler",
        "registry_path": ("3d", "trellis"),
        "vram_gb": 10,
        "defaults": {"steps": 50, "guidance": 3.0},
    },
    {
        "name": "hy_motion",
        "handler": "services.model_engine.handlers.hy_motion",
        "handler_cls": "HYMotionHandler",
        "registry_path": ("motion", "hy-motion-1.0"),
        "vram_gb": 6,
        "defaults": {"steps": 50, "cfg": 2.0},
    },
    {
        "name": "moss_soundeffect",
        "handler": "services.model_engine.handlers.moss",
        "handler_cls": "MossHandler",
        "registry_path": ("audio", "moss-soundeffect"),
        "vram_gb": 16,
        "defaults": {"max_tokens": 4096},
    },
    {
        "name": "see_through",
        "handler": "services.model_engine.handlers.see_through",
        "handler_cls": "SeeThroughHandler",
        "registry_path": None,
        "vram_gb": 6,
        "defaults": {"resolution": 1280, "steps": 30},
    },
    {
        "name": "faster_qwen3_tts",
        "handler": "services.model_engine.handlers.faster_qwen3_tts",
        "handler_cls": "FasterQwen3TTSHandler",
        "registry_path": ("tts", "qwen3-tts-12hz-1.7b-customvoice"),
        "vram_gb": 6,
        "defaults": {"voice": "Aiden", "language": "English"},
    },
    {
        "name": "vibevoice_asr",
        "handler": "services.model_engine.handlers.vibevoice_asr",
        "handler_cls": "VibeVoiceASRHandler",
        "registry_path": ("asr", "vibevoice-asr"),
        "vram_gb": 16,
        "defaults": {"language": "english", "max_tokens": 512},
    },
    {
        "name": "vibevoice_tts",
        "handler": "services.model_engine.handlers.vibevoice_tts",
        "handler_cls": "VibeVoiceTTSHandler",
        "registry_path": ("tts", "vibevoice"),
        "vram_gb": 18,
        "defaults": {"language": "English", "max_tokens": 4096},
    },
    {
        "name": "kokoro",
        "handler": "services.model_engine.handlers.kokoro",
        "handler_cls": "KokoroHandler",
        "registry_path": ("tts", "kokoro"),
        "vram_gb": 0,
        "defaults": {"voice": "af_bella", "speed": 1.0},
    },
    {
        "name": "espeak",
        "handler": "services.model_engine.handlers.espeak",
        "handler_cls": "EspeakHandler",
        "registry_path": None,
        "vram_gb": 0,
        "defaults": {"voice": "en", "speed": 175, "pitch": 50},
    },
    {
        "name": "faster_whisper",
        "handler": "services.model_engine.handlers.faster_whisper",
        "handler_cls": "FasterWhisperHandler",
        "registry_path": ("asr", "faster-whisper"),
        "vram_gb": 0,
        "defaults": {"language": None, "beam_size": 5},
    },
]

# ─── mmgp Profiles ────────────────────────────────────────────────────────────

MMGP_PROFILES = {
    "max_speed": 1,
    "balanced": 2,
    "low_vram": 4,
    "minimum": 5,
}

# ─── Payload Passthrough — Security Allowlist ─────────────────────────────────

# API payload key → upstream generate() kwarg name (naming differences only)
_KEY_MAP = {
    "prompt": "input_prompt",
    "negative_prompt": "n_prompt",
    "steps": "sampling_steps",
    "guidance": "guide_scale",
    "frames": "frame_num",
}

# Keys safe to pass through directly (no mapping needed)
_SAFE_PASSTHROUGH = {
    "seed", "width", "height", "fps", "batch_size", "shift",
    "sample_solver", "temperature", "top_p", "top_k",
    "fit_into_canvas", "joint_pass", "enable_RIFLEx",
    "cfg_star_switch", "cfg_zero_step", "embedded_guidance_scale",
    "alt_guide_scale", "guide2_scale", "guide3_scale",
    "switch_threshold", "switch2_threshold", "guide_phases",
    "model_switch_phase", "NAG_scale", "NAG_tau", "NAG_alpha",
    "apg_switch", "overlapped_latents", "return_latent_slice",
    "overlap_noise", "overlap_size", "conditioning_latents_size",
    "window_no", "window_start_frame_no",
    "denoising_strength", "masking_strength", "motion_amplitude",
    "audio_scale", "audio_cfg_scale", "audio_context_lens",
    "audio_prompt_type", "audio_proj",
    "perturbation_switch", "perturbation_layers",
    "perturbation_start", "perturbation_end",
    "self_refiner_setting", "self_refiner_plan",
    "self_refiner_f_uncertainty", "self_refiner_certain_percentage",
    "color_correction_strength", "prefix_frames_count",
    "video_prompt_type", "image_mode", "model_mode",
    "loras_selected", "control_scale_alt",
    "speakers_bboxes", "pre_video_frame", "prefix_video",
    "image_refs_relative_size", "image_prompt_type",
    "duration_seconds", "pause_seconds",
    # ACE-Step / TTS specific
    "alt_prompt", "language",
}

# Blocked for security (filesystem paths, internal config)
_BLOCKED_KEYS = {
    "input_custom", "audio_guide", "audio_guide2",
    "custom_settings", "model_filename", "lora_dir",
    "output_dir", "input_frames", "input_masks",
    "input_ref_images", "input_video", "input_faces",
    "image_start", "image_end",
}

# ─── Dynamic Model Discovery ──────────────────────────────────────────────────

def discover_models(models_root: Path | None = None) -> dict:
    """Auto-discover all available models.

    Scans vendor handlers via query_supported_types() and matches against
    weight files on disk. Model_engine handlers are always registered.
    Returns a dict of {model_name: entry} compatible with the service registry.
    """
    from registry.config import Config
    from registry.models import ModelRegistry

    cfg = Config()
    models_root = models_root or Path(cfg.models_root)
    registry = ModelRegistry()

    discovered = {}

    # ── 1. Vendor handlers (upstream Wan2GP) ──────────────────────────────
    _ensure_vendor_path()

    for handler_path in VENDOR_HANDLERS:
        try:
            handler_mod = importlib.import_module(handler_path)
            family = handler_mod.family_handler
            supported = family.query_supported_types()

            for model_type in sorted(supported):
                model_key = _vendor_key(model_type, handler_path)
                weight_path = _find_vendor_weights(model_type, registry, models_root)

                discovered[model_key] = {
                    "engine": "vendor",
                    "handler_path": handler_path,
                    "model_type": model_type,
                    "base_model_type": model_type,
                    **({"blocked": True, "blocked_reason": "No safetensors found"}
                       if not weight_path else {}),
                    "weight_path": str(weight_path) if weight_path else None,
                    "defaults": {},
                }
        except ImportError as e:
            logger.debug("Vendor handler unavailable: %s (%s)", handler_path, e)
        except Exception as e:
            logger.debug("Vendor handler discovery failed: %s (%s)", handler_path, e)

    # ── 2. Model_engine handlers (our code) ───────────────────────────────
    for entry in MODEL_ENGINE_ENTRIES:
        handler_mod = importlib.import_module(entry["handler"])
        handler_cls = getattr(handler_mod, entry["handler_cls"])
        handler = handler_cls()

        base_name = entry["name"]

        for model_type in handler.supported_types():
            # Derive user-friendly registry key from model_type
            # e.g., "ace_step_v1_5_turbo" → "ace_step/v1_5_turbo"
            # For single-type handlers (kokoro, espeak) → "kokoro", "espeak"
            if model_type == base_name:
                reg_key = base_name
            elif model_type.startswith(base_name + "_"):
                reg_key = base_name + "/" + model_type[len(base_name) + 1:]
            else:
                reg_key = model_type

            me_info = {
                "engine": "model_engine",
                "handler": entry["handler"],
                "handler_cls": entry["handler_cls"],
                "model_type": model_type,
                "registry_path": entry.get("registry_path"),
                "vram_gb": entry.get("vram_gb", 0),
                "defaults": dict(entry.get("defaults", {})),
                "key_map": dict(entry.get("key_map", {})),
            }

            # Check if model weights exist
            if entry.get("registry_path"):
                try:
                    model_path = registry.get_path(*entry["registry_path"])
                    if Path(model_path).is_dir():
                        me_info["weight_path"] = str(model_path)
                except (KeyError, FileNotFoundError):
                    pass
            elif entry.get("vram_gb", 0) == 0:
                # CPU services — no weight files needed
                me_info["weight_path"] = None

            discovered[reg_key] = me_info

    return discovered


def _vendor_key(model_type: str, handler_path: str) -> str:
    """Derive a stable model key from the model type and handler family."""
    family = handler_path.split(".")[1]  # e.g., "wan", "hyvideo", "flux"
    family_map = {
        "wan": "wan", "hyvideo": "hunyuan", "TTS": "tts",
        "ltx_video": "ltxv", "ltx2": "ltx2", "longcat": "longcat",
        "qwen": "qwen", "kandinsky5": "kandinsky",
        "z_image": "z_image", "magi_human": "magi_human",
        "flux": "flux",
    }
    prefix = family_map.get(family, family)
    return f"{prefix}/{model_type}"


def _find_vendor_weights(model_type: str, registry, models_root: Path) -> Path | None:
    """Find weight files for a vendor model type."""
    # Try ModelRegistry first
    model_key = model_type.replace("/", "-").replace(".", "-")
    try:
        path = registry.get_path("wan2gp", model_key)
        if Path(path).is_dir() and list(Path(path).rglob("*.safetensors")):
            return Path(path)
    except (KeyError, FileNotFoundError):
        pass

    # Fall back to models_root/wan2gp/<model_type>/
    fallback = models_root / "wan2gp" / model_type
    if fallback.is_dir() and list(fallback.rglob("*.safetensors")):
        return fallback

    return None


_ven_loaded = False


def _ensure_vendor_path():
    """Add Wan2GP vendor to sys.path (idempotent)."""
    global _ven_loaded
    if _ven_loaded:
        return
    vendor = str(WAN2GP_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    os.environ.setdefault("WAN2GP_ROOT", vendor)
    _ven_loaded = True


def _ensure_quantized_cache():
    """Monkey-patch QuantizedCacheConfig if missing (transformers compat)."""
    import transformers.cache_utils as _tcu
    if not hasattr(_tcu, "QuantizedCacheConfig"):
        _tcu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {
            "__init__": lambda self, **kw: None,
            "__getattr__": lambda self, name: None,
        })


# ─── Wan2GP Service ───────────────────────────────────────────────────────────

class Wan2GPService:
    """Standalone Wan2GP service — mmgp-managed VRAM for ALL GPU models.

    Not a ForgeService. Runs as its own Ray Serve deployment with num_gpus: 1.0.
    mmgp handles all memory management; no external VRAM accounting needed.
    """

    service_name = "wan2gp"
    default_model = "wan/t2v-14B"

    def __init__(self, models_root: Path | None = None):
        self._registry = discover_models(models_root)
        self._offload = None
        self._loaded_model: str | None = None
        self._models: dict[str, dict] = {}
        self._vendor_ready = False

    # ── Discovery API ─────────────────────────────────────────────────────

    @property
    def registry(self) -> dict:
        return dict(self._registry)

    def available_models(self) -> list[str]:
        """Models that can be loaded (weights exist, not blocked)."""
        return sorted(k for k, v in self._registry.items()
                      if not v.get("blocked") or v.get("vram_gb", 0) == 0)

    def blocked_models(self) -> dict[str, str]:
        """Models that are blocked (missing weights, deps, etc.)."""
        return {k: v.get("blocked_reason", "unknown")
                for k, v in self._registry.items() if v.get("blocked")}

    def status(self) -> dict:
        """Full status: registry, loaded model, VRAM."""
        gpu = {}
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu = {
                    "device": props.name,
                    "total_mb": int(props.total_memory / (1024 * 1024)),
                    "allocated_mb": int(torch.cuda.memory_allocated(0) / (1024 * 1024)),
                    "reserved_mb": int(torch.cuda.memory_reserved(0) / (1024 * 1024)),
                }
        except Exception:
            pass

        return {
            "loaded": self._loaded_model,
            "available": self.available_models(),
            "blocked": self.blocked_models(),
            "total_models": len(self._registry),
            "gpu": gpu,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self, model_name: str | None = None) -> None:
        model_name = model_name or self.default_model

        if model_name == self._loaded_model:
            return

        # Unload current model
        self.unload()

        entry = self._registry.get(model_name)
        if entry is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {self.available_models()}"
            )
        if entry.get("blocked"):
            raise RuntimeError(
                f"Model '{model_name}' is blocked: {entry.get('blocked_reason', 'unknown')}"
            )

        engine = entry.get("engine", "vendor")
        if engine == "model_engine":
            self._load_model_engine(model_name, entry)
        else:
            self._load_vendor(model_name, entry)

        self._loaded_model = model_name

        vram = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
        logger.info("Wan2GP: loaded %s (VRAM=%.0fMB)", model_name, vram)

    def unload(self) -> None:
        if self._offload is not None:
            try:
                self._offload.unload_all()
            except Exception:
                pass
            try:
                self._offload.release()
            except Exception:
                pass
            self._offload = None

        self._models.clear()
        self._loaded_model = None

        try:
            from mmgp import offload
            offload.flush_torch_caches()
        except ImportError:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, payload: dict) -> dict:
        model_key = payload.get("model", self._loaded_model or self.default_model)

        if model_key != self._loaded_model:
            self.load(model_key)

        entry = self._registry.get(self._loaded_model)
        if entry is None:
            return {"status": "error", "error": f"No model loaded"}

        try:
            if entry.get("engine") == "model_engine":
                m = self._models.get(self._loaded_model)
                if m is None:
                    return {"status": "error", "error": "Model entry not found"}
                model = m["model"]
                # If the model has __call__ (old pattern: dispatch payload → generate),
                # use it directly. Otherwise use generate(**kwargs) (new Wan2GP pattern).
                if hasattr(model, "__call__"):
                    return model(payload)
                defaults = entry.get("defaults", {})
                key_map = entry.get("key_map", {})
                kwargs = _build_generate_kwargs(payload, defaults, key_map)
                return model.generate(**kwargs)

            # Vendor path: build kwargs, call generate
            gen = self._do_generate(self._models[self._loaded_model], payload)
            result = {
                "status": "ok",
                "data": base64.b64encode(gen["data"]).decode(),
                "media_type": gen["media_type"],
            }
            result["model"] = self._loaded_model
            return result

        except Exception as e:
            logger.error("Wan2GP inference failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def _do_generate(self, entry: dict, payload: dict) -> dict:
        """Vendor generate with security-allowlisted payload passthrough."""
        model = entry["model"]
        info = entry["info"]
        defaults = info.get("defaults", {})
        base_model_type = info.get("base_model_type", "")

        # Build kwargs from payload with allowlist
        kwargs = _build_generate_kwargs(payload, defaults)

        # Handle image for i2v models
        if base_model_type in ("i2v", "i2v_2_2"):
            image_b64 = payload.get("image_b64", "")
            if not image_b64:
                raise ValueError("image_b64 is required for i2v models")
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            kwargs["image_start"] = img

        logger.info("Wan2GP generate: prompt=%r steps=%d",
                     kwargs.get("input_prompt", "")[:80],
                     kwargs.get("sampling_steps", 50))

        output = model.generate(**kwargs)

        from mmgp import offload
        offload.clear_caches()

        # Extract frames/audio from output
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
            fps = int(payload.get("fps", defaults.get("fps", 16)))
            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            for f in frames_np:
                writer.append_data(f)
            writer.close()
        else:
            from PIL import Image as PILImage
            img = frames_np[0] if len(frames_np.shape) == 4 else frames_np
            PILImage.fromarray(img).save(tmp_path, format="PNG")

        with open(tmp_path, "rb") as f:
            data_bytes = f.read()
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
        return {"data": data_bytes, "media_type": media_type, **extra}

    # ── Vendor Loading ────────────────────────────────────────────────────

    def _load_vendor(self, model_name: str, entry: dict) -> None:
        _ensure_vendor_path()
        _ensure_quantized_cache()

        handler_path = entry["handler_path"]
        model_type = entry["model_type"]
        base_model_type = entry["base_model_type"]

        handler_mod = importlib.import_module(handler_path)
        handler = handler_mod.family_handler

        from registry.config import Config
        from registry.models import ModelRegistry
        cfg = Config()
        model_registry = ModelRegistry()

        model_key_safe = model_name.replace("/", "-")
        model_path = None
        try:
            path = model_registry.get_path("wan2gp", model_key_safe)
            model_path = Path(path) if Path(path).is_dir() else None
        except (KeyError, FileNotFoundError):
            pass

        if model_path is None:
            model_path = Path(cfg.models_root) / "wan2gp" / model_type

        checkpoint_root = model_path.parent if model_path.parent.name == "wan2gp" else model_path
        from shared.utils import files_locator as fl
        fl.set_checkpoints_paths([str(checkpoint_root)])

        model_files = list(model_path.rglob("*.safetensors"))
        if not model_files and model_type.endswith("tts"):
            parent_files = list(model_path.parent.glob("*.safetensors"))
            model_files = [f for f in parent_files
                          if model_type in f.name.lower() or "index_tts2" in f.name.lower()]
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
            "info": entry,
            "loaded_at": time.time(),
        }

    def _load_model_engine(self, model_name: str, entry: dict) -> None:
        handler_mod = importlib.import_module(entry["handler"])
        handler_cls = getattr(handler_mod, entry["handler_cls"])
        handler = handler_cls()

        from registry.config import Config
        cfg = Config()
        models_root = Path(cfg.models_root)

        registry_path = entry.get("registry_path")
        model_path = None

        if registry_path:
            try:
                from registry.models import ModelRegistry
                registry = ModelRegistry()
                model_path = registry.get_path(*registry_path)
                model_path = Path(model_path)
            except (KeyError, FileNotFoundError):
                model_path = None

        if model_path is None or not model_path.is_dir():
            model_path = handler.resolve_path(model_name, models_root)

        if not model_path.is_dir():
            if entry.get("vram_gb", 0) == 0:
                model_path = models_root
            elif registry_path is None:
                model_path = Path("/opt/seethrough")
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"Model path not found for {model_name}: {model_path}"
                )

        model_type = entry.get("model_type", model_name)
        logger.info("Loading %s (model_engine, nn.Module) from %s", model_type, model_path)
        torch.set_default_device("cpu")

        load_result = handler.load_model(
            model_type, model_path=model_path, dtype=torch.bfloat16
        )

        # Handle both old (LoadResult) and new ((model, pipe) tuple) return types
        if isinstance(load_result, tuple):
            model, pipe = load_result
        else:
            model = load_result.pipeline
            pipe = {"pipe": load_result.pipe, "coTenantsMap": load_result.co_tenants or {}}

        # Unwrap double-nested pipe following upstream convention
        pipe_kwargs = {}
        if isinstance(pipe, dict) and "pipe" in pipe:
            pipe_kwargs = pipe
            pipe = pipe_kwargs.pop("pipe", {})

        co_tenants = pipe_kwargs.pop("coTenantsMap", {})

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
            "model": model,
            "pipe": pipe,
            "info": entry,
            "loaded_at": time.time(),
        }

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


# ─── Payload Passthrough Helpers ───────────────────────────────────────────────

def _build_generate_kwargs(payload: dict, defaults: dict, key_map: dict | None = None) -> dict:
    """Build kwargs for model.generate() with security allowlist.

    Forwards all safe params, maps known key names, blocks dangerous ones.
    Accepts optional per-model key_map that is merged with the global _KEY_MAP.
    """
    merged_map = dict(_KEY_MAP)
    if key_map:
        merged_map.update(key_map)

    kwargs = {}

    # Apply defaults first (can be overridden by payload)
    for k, v in defaults.items():
        if k in _SAFE_PASSTHROUGH or k in merged_map.values():
            kwargs[k] = v

    # Map known keys from payload
    for src, dst in merged_map.items():
        if src in payload:
            kwargs[dst] = payload[src]
        elif src not in payload and dst not in kwargs:
            kwargs[dst] = defaults.get(src)

    # Safe passthrough from payload
    for key in _SAFE_PASSTHROUGH:
        if key in payload:
            kwargs[key] = payload[key]

    # Handle seed specially
    if "seed" not in kwargs:
        kwargs["seed"] = -1

    # Log blocked keys for audit
    for key in _BLOCKED_KEYS:
        if key in payload:
            logger.debug("Blocked key in payload: %s", key)

    return kwargs

