"""Wan2GP Service — Wan2GP-native model discovery, mmgp-managed VRAM.

All models (vendor + custom families) discovered via Wan2GP fork's
refresh_model_defs() and map_family_handlers(). Custom model families live
as first-class citizens under opt/wan2gp/models/.

Architecture:
    - Vendor handlers from the fork's wgp.py family_handlers list
    - Custom handlers appended via CUSTOM_HANDLERS below
    - Wan2GP scans defaults/*.json → discovers all models
    - handler.load_model() → (pipeline, pipe_dict) → offload.profile() for mmgp
    - CPU models use empty pipe dict (no mmgp)
    - Payload passthrough with security allowlist
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

WAN2GP_VENDOR = Path(os.environ.get("WAN2GP_ROOT", "/opt/wan2gp"))

# ─── mmgp Profiles ────────────────────────────────────────────────────────────

MMGP_PROFILES = {
    "max_speed": 1,
    "balanced": 2,
    "low_vram": 4,
    "minimum": 5,
}

# ─── Payload Passthrough — Security Allowlist ─────────────────────────────────

_KEY_MAP = {
    "prompt": "input_prompt",
    "negative_prompt": "n_prompt",
    "steps": "sampling_steps",
    "guidance": "guide_scale",
    "frames": "frame_num",
    "reference_images": "input_ref_images",
}

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
    "alt_prompt", "language",
    "workflow", "character_name", "existing_character", "sex", "age", "race",
    "eyes", "hair", "face_attrs", "body_attrs", "skin_color",
    "additional_details", "background_color", "aesthetics", "nsfw",
    "instruction", "costumes_data", "emotions_data", "prompt_style",
    "game_name", "additional_caption", "min_size", "target_height",
    "reference_images", "reference_weights", "cfg", "timeout",
    "text", "voice", "audio_b64", "image_b64",
    "image", "resolution", "steps", "ss_steps", "slat_steps",
}

_BLOCKED_KEYS = {
    "input_custom", "audio_guide", "audio_guide2",
    "custom_settings", "model_filename", "lora_dir",
    "output_dir", "input_frames", "input_masks",
    "input_ref_images", "input_video", "input_faces",
    "image_start", "image_end",
}

# Dummy offloadobj for WanAny2V.generate() which calls offloadobj.unload_all()
_DEFAULT_OFFLOADOBJ = type("_DummyOffload", (), {
    "unload_all": lambda self: None,
    "release": lambda self: None,
})()

_logged_msgs: set = set()
def log_once(msg: str) -> None:
    if msg not in _logged_msgs:
        _logged_msgs.add(msg)
        logger.warning("Wan2GP: %s", msg)
        print(f"[Wan2GP-debug] {msg}", flush=True)

# ─── Vendor Path Setup ─────────────────────────────────────────────────────────

_ven_loaded = False


def _ensure_vendor_path():
    global _ven_loaded
    if _ven_loaded:
        return
    vendor = str(WAN2GP_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    os.environ.setdefault("WAN2GP_ROOT", vendor)
    _ven_loaded = True


def _ensure_quantized_cache():
    import transformers.cache_utils as _tcu
    if not hasattr(_tcu, "QuantizedCacheConfig"):
        _tcu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {
            "__init__": lambda self, **kw: None,
            "__getattr__": lambda self, name: None,
        })


def _ensure_transformers_compat():
    """Patch renamed/removed transformers symbols that vendor handlers depend on."""
    import transformers.generation.configuration_utils as _gcu
    if not hasattr(_gcu, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
        _gcu.NEED_SETUP_CACHE_CLASSES_MAPPING = getattr(
            _gcu, "ALL_CACHE_IMPLEMENTATIONS", ()
        )
    if not hasattr(_gcu, "QUANT_BACKEND_CLASSES_MAPPING"):
        _gcu.QUANT_BACKEND_CLASSES_MAPPING = {}


# ─── Dynamic Model Discovery ──────────────────────────────────────────────────

def discover_models(models_root: Path | None = None) -> dict:
    """Discover all models via Wan2GP's native family_handlers system.

    Imports each handler from the family_handlers list, calls
    query_supported_types() to get model types, checks for weight files.
    """
    from registry.config import Config
    from registry.models import ModelRegistry

    cfg = Config()
    models_root = models_root or Path(cfg.models_root)
    registry = ModelRegistry()
    _ensure_vendor_path()

    discovered = {}

    # Wan2GP's family_handlers list (from wgp.py, includes our additions)
    family_handlers_list = _get_family_handlers()

    for handler_path in family_handlers_list:
        try:
            handler_mod = importlib.import_module(handler_path)
            handler = handler_mod.family_handler
            supported = handler.query_supported_types()

            for model_type in sorted(supported):
                model_key = _derive_key(model_type, handler_path)
                weight_path = _find_weights(model_type, handler_path, registry, models_root)

                entry = {
                    "handler_path": handler_path,
                    "model_type": model_type,
                    "base_model_type": model_type,
                    "defaults": {},
                }

                if weight_path:
                    entry["weight_path"] = str(weight_path)
                elif model_type not in _CPU_ONLY_TYPES:
                    # Vendor handlers (models.*) download weights on first use
                    is_vendor = handler_path.startswith("models.")
                    if not is_vendor:
                        entry["blocked"] = True
                        entry["blocked_reason"] = "No weights found"

                discovered[model_key] = entry

        except ImportError as e:
            logger.debug("Handler unavailable: %s (%s)", handler_path, e)
        except Exception as e:
            logger.debug("Handler discovery failed: %s (%s)", handler_path, e)

    return discovered


# Types that don't need local weight files (CPU, proxy, or HuggingFace auto-download)
_CPU_ONLY_TYPES = {
    "kokoro", "espeak", "faster_whisper",
}

_HF_AUTO_DOWNLOAD = {
    "qwen_image_edit_vnccs_20B",
}


CUSTOM_HANDLERS = [
    "models.kokoro.kokoro_handler",
    "models.moss.moss_handler",
    "models.espeak.espeak_handler",
    "models.faster_whisper.faster_whisper_handler",
    "models.faster_qwen3_tts.faster_qwen3_tts_handler",
    "models.vibevoice_asr.vibevoice_asr_handler",
    "models.vibevoice_tts.vibevoice_tts_handler",
    "models.anigen.anigen_handler",
    "models.see_through.see_through_handler",
    "models.hy_motion.hy_motion_handler",
    "models.pixal3d.pixal3d_handler",
]

def _get_family_handlers() -> list[str]:
    """Discover family handlers by scanning models/ directory + custom list.

    Scans opt/wan2gp/models/ for ``*_handler.py`` files, converting each
    to a dotted import path (e.g. ``models.kokoro.kokoro_handler``).
    Custom handlers are appended if not already found by the scan.
    """
    import importlib

    handlers: list[str] = []
    models_dir = WAN2GP_VENDOR / "models"

    if models_dir.is_dir():
        for family_dir in sorted(models_dir.iterdir()):
            if not family_dir.is_dir() or family_dir.name.startswith("_"):
                continue
            for handler_file in sorted(family_dir.glob("*_handler.py")):
                handler_path = f"models.{family_dir.name}.{handler_file.stem}"
                try:
                    # Quick check that it actually has family_handler
                    mod = importlib.import_module(handler_path)
                    if hasattr(mod, "family_handler"):
                        handlers.append(handler_path)
                except (ImportError, ModuleNotFoundError):
                    # Handler may have import deps not available outside Docker
                    # — include it anyway, it'll fail at load_model() time
                    handlers.append(handler_path)

    # Append custom handlers not found by the scan
    for h in CUSTOM_HANDLERS:
        if h not in handlers:
            handlers.append(h)
    return handlers


def _derive_key(model_type: str, handler_path: str) -> str:
    """Derive a stable model key from the model type and handler path."""
    parts = handler_path.split(".")
    if len(parts) >= 2:
        family = parts[1]
    else:
        family = model_type

    family_map = {
        "wan": "wan", "hyvideo": "hunyuan", "TTS": "tts",
        "ltx_video": "ltxv", "ltx2": "ltx2", "longcat": "longcat",
        "qwen": "qwen", "kandinsky5": "kandinsky",
        "z_image": "z_image", "magi_human": "magi_human",
        "flux": "flux",
    }
    prefix = family_map.get(family, family)
    return f"{prefix}/{model_type}"


_WEIGHT_SEARCH = {
    "faster-qwen3-tts": [("tts", "qwen3-tts")],
    "anigen": [("3d", "anigen")],
    "moss-soundeffect": [("audio", "moss-soundeffect")],
    "moss-tts": [("audio", "moss-tts")],
    "moss-ttsd": [("audio", "moss-ttsd")],
    "moss-voicegenerator": [("audio", "moss-voicegenerator")],
    "see-through": [("image", "see-through-layerdiff"), ("image", "see-through-marigold")],
    "hy-motion-1.0": [("motion", "hy-motion-1.0")],
    "hy-motion-1.0-lite": [("motion", "hy-motion-1.0-lite")],
    "vibevoice-asr": [("asr", "vibevoice-asr")],
    "vibevoice-tts": [("tts", "vibevoice-tts")],
    # Wan2GP vendor models — registry keys use versioned names
    "t2v":           [("wan2gp", "wan-t2v-14B")],
    "i2v":           [("wan2gp", "wan-i2v-14B")],
    "t2v_2_2":       [("wan2gp", "wan-t2v-14B")],
    "i2v_2_2":       [("wan2gp", "wan-i2v-14B")],
    "trellis":       [("3d", "trellis")],
    "index_tts2":    [("tts", "index-tts")],
    "kokoro":        [("tts", "kokoro")],
    "faster_whisper": [("asr", "faster-whisper")],
    # Flux models
    "flux":          [("wan2gp", "flux")],
    "flux_schnell":  [("wan2gp", "flux-schnell")],
    "flux2_dev":     [("wan2gp", "flux2-dev")],
    "flux2_klein_4b": [("wan2gp", "flux2-klein-4b")],
    "flux_chroma":   [("wan2gp", "flux-chroma")],
}


def _find_weights(model_type: str, handler_path: str, registry, models_root: Path) -> Path | None:
    """Find weight files for a model."""
    if model_type in _HF_AUTO_DOWNLOAD:
        return Path("/dev/null")  # sentinel: not blocked, downloads at load time

    model_key_safe = model_type.replace("/", "-").replace(".", "-")
    weight_exts = ("*.safetensors", "*.pt", "*.pth", "*.ckpt", "*.bin")

    def _has_weights(p: Path) -> bool:
        return p.is_dir() and any(list(p.rglob(ext)) for ext in weight_exts)

    # Search wan2gp section first (vendor models)
    try:
        path = registry.get_path("wan2gp", model_key_safe)
        if _has_weights(Path(path)):
            return Path(path)
    except (KeyError, FileNotFoundError):
        pass

    # Search model-specific registry paths
    for svc_type, reg_name in _WEIGHT_SEARCH.get(model_type, []):
        try:
            path = registry.get_path(svc_type, reg_name)
            if _has_weights(Path(path)):
                return Path(path)
        except (KeyError, FileNotFoundError):
            pass

    # Fallback: models_root/wan2gp/<model_type>
    fallback = models_root / "wan2gp" / model_type
    if _has_weights(fallback):
        return fallback

    # Fallback: ckpts/ (writable, used for auto-downloaded vendor models)
    ckpts = Path("ckpts").resolve()
    if (ckpts / f"{model_type}.safetensors").is_file() or \
       (ckpts / f"{model_type}.pth").is_file():
        return ckpts

    return None


# ─── Wan2GP Service ───────────────────────────────────────────────────────────

class Wan2GPService:
    """Standalone Wan2GP service — unified model discovery via family_handlers."""

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
        return sorted(k for k, v in self._registry.items()
                      if not v.get("blocked") or v.get("model_type") in _CPU_ONLY_TYPES)

    def blocked_models(self) -> dict[str, str]:
        return {k: v.get("blocked_reason", "unknown")
                for k, v in self._registry.items() if v.get("blocked")}

    def status(self) -> dict:
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

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        model_name = model_name or self.default_model

        if model_name == self._loaded_model:
            return

        self.unload()

        entry = self._registry.get(model_name)
        if entry is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {self.available_models()}"
            )
        if entry.get("blocked") or ("weight_path" not in entry
                                     and entry.get("model_type") not in _CPU_ONLY_TYPES):
            # Try auto-download from registry source
            model_type = entry.get("model_type", model_name)
            if self._try_download(model_type):
                self._registry = discover_models()
                entry = self._registry.get(model_name)
                if entry is None or entry.get("blocked"):
                    raise RuntimeError(
                        f"Model '{model_name}' still blocked after download: "
                        f"{entry.get('blocked_reason', 'unknown') if entry else 'not found'}"
                    )
            elif entry.get("blocked"):
                raise RuntimeError(
                    f"Model '{model_name}' is blocked: {entry.get('blocked_reason', 'unknown')}"
                )

        try:
            self._load_model(model_name, entry, quant=quant)
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, e)
            self._models.pop(model_name, None)
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e
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
            try:
                self.load(model_key)
            except Exception as e:
                self._loaded_model = None
                return {"status": "error", "error": str(e)}

        entry = self._registry.get(self._loaded_model)
        if entry is None:
            return {"status": "error", "error": "No model loaded"}

        try:
            m = self._models.get(self._loaded_model)
            if m is None:
                return {"status": "error", "error": "Model entry not found"}

            model = m["model"]
            info = m.get("info", entry)
            defaults = info.get("defaults", {})

            kwargs = _build_generate_kwargs(payload, defaults)

            # Handle image for i2v / image-input models
            base_model_type = info.get("base_model_type", "")
            if base_model_type in ("i2v", "i2v_2_2") or payload.get("image_b64"):
                image_b64 = payload.get("image_b64", "")
                if image_b64:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                    kwargs["image_start"] = img

            # Handle second image for last-frame conditioning (WDC FFLF)
            if payload.get("image_end_b64"):
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(base64.b64decode(payload["image_end_b64"]))).convert("RGB")
                kwargs["image_end"] = img

            # Handle reference_images for VNCCS (base64 strings → PIL images)
            if "reference_images" in kwargs and isinstance(kwargs["reference_images"], list):
                from PIL import Image
                import io
                decoded = []
                for img_b64 in kwargs["reference_images"]:
                    if isinstance(img_b64, str):
                        decoded.append(
                            Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                        )
                if decoded:
                    kwargs["reference_images"] = decoded

            logger.info("Wan2GP generate: model=%s kwargs=%s",
                        self._loaded_model, list(kwargs.keys()))

            # Some Wan2GP models need _interrupt flag and offloadobj parameter
            if not getattr(model, "_interrupt", False):
                model._interrupt = False

            # WanAny2V.generate() has an offloadobj=None parameter that's
            # called as offloadobj.unload_all() — provide a dummy no-op.
            kwargs.setdefault("offloadobj", _DEFAULT_OFFLOADOBJ)

            # mmgp shared_state needs _attention key for Wan models
            from mmgp import offload as _moff
            if "_attention" not in _moff.shared_state:
                _moff.shared_state["_attention"] = "sdpa"

            # Load LoRAs if requested (loras_selected was silently ignored before)
            loras_selected = payload.get("loras_selected", [])
            if loras_selected and isinstance(loras_selected, list):
                kwargs["loras_selected"] = loras_selected
                pipe_dict = m.get("pipe", {})
                transformer = self._find_transformer(pipe_dict)
                if transformer is not None:
                    resolved = self._resolve_lora_paths(loras_selected)
                    if resolved:
                        from mmgp import offload as _moff
                        _moff.load_loras_into_model(
                            transformer, resolved,
                            [1.0] * len(resolved),
                            activate_all_loras=True,
                        )
                        # Build slists so update_loras_slists activates them
                        kwargs["loras_slists"] = {
                            "phase1": [[1.0]] * len(resolved),
                            "phase2": [],
                            "phase3": [],
                        }
            kwargs.setdefault("loras_slists", {"phase1": [], "phase2": [], "phase3": []})
            # Wan models call callback() for progress — provide a no-op
            kwargs.setdefault("callback", lambda *a, **kw: None)

            result = model.generate(**kwargs)

            # If pipeline returns our custom format (status + data), pass through
            if isinstance(result, dict) and "status" in result:
                result["model"] = self._loaded_model
                return result

            # Vendor format: tensor output → encode to video/image
            gen = self._encode_output(result, payload, defaults)
            gen["model"] = self._loaded_model
            return gen

        except Exception as e:
            logger.error("Wan2GP inference failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    # ── LoRA Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_transformer(pipe: dict) -> Any | None:
        """Find the transformer nn.Module in a pipe dict.

        Tries common key names. Returns None if not found.
        """
        for key in ("transformer", "model", "unet", "dit"):
            if key in pipe:
                return pipe[key]
        if "motion_transformer" in pipe:
            return pipe["motion_transformer"]
        return None

    @staticmethod
    def _resolve_lora_paths(loras_selected: list[str]) -> list[str]:
        """Resolve LoRA filenames to full paths.

        Searches models_root/loras/ and Wan2GP's files_locator paths.
        """
        from pathlib import Path
        from registry.config import Config

        models_root = Path(Config().models_root)
        lora_dirs = [
            models_root / "loras",
            models_root / "wan2gp" / "loras",
            models_root / "image-gen" / "comfyui" / "loras" / "qwen",
            models_root / "image-gen" / "comfyui" / "loras",
        ]

        resolved = []
        for name in loras_selected:
            found = None
            for ld in lora_dirs:
                candidate = ld / name
                if candidate.exists():
                    found = str(candidate)
                    break
            if found is None:
                try:
                    from shared.utils import files_locator as fl
                    found = fl.locate_file(name)
                except Exception:
                    pass
            if found:
                resolved.append(found)
            else:
                logger.warning("LoRA not found: %s (searched %s)", name, lora_dirs)
        return resolved

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_model(self, model_name: str, entry: dict, *,
                    quant: str | None = None) -> None:
        _ensure_vendor_path()
        _ensure_quantized_cache()
        _ensure_transformers_compat()

        handler_path = entry["handler_path"]
        model_type = entry["model_type"]
        base_model_type = entry.get("base_model_type", model_type)

        handler_mod = importlib.import_module(handler_path)
        handler = handler_mod.family_handler

        from registry.config import Config
        from registry.models import ModelRegistry
        cfg = Config()
        model_registry = ModelRegistry()

        model_path = self._resolve_model_path(model_name, entry, model_registry, cfg)

        # If no local weights found, try downloading via handler's query_model_files
        if model_path is None:
            self._ensure_vendor_files(handler, base_model_type, model_def={})
            # Re-resolve after download
            model_path = self._resolve_model_path(model_name, entry, model_registry, cfg)

        extra_paths = self._resolve_handler_paths(
            model_type, model_registry, cfg, quant=quant)
        model_def = self._build_model_def(handler, base_model_type, model_path)
        if extra_paths:
            model_def.update(extra_paths)

        is_cpu = model_type in _CPU_ONLY_TYPES
        model_filename = self._resolve_model_filename(
            model_type, base_model_type, model_path, model_def)
        logger.info("Loading %s from %s (family_handler, quant=%s)", model_name, model_path or "N/A", quant)
        torch.set_default_device("cpu")

        # Add model path to Wan2GP's files_locator search paths so vendor
        # handlers can find tokenizer configs, VAE weights, etc.
        if model_path and model_path.is_dir():
            from shared.utils import files_locator as fl
            p = str(model_path)
            if p not in fl._checkpoints_paths:
                fl._checkpoints_paths.append(p)

        # Resolve text encoder filename for vendor models that need it
        # (e.g., Wan's T5 encoder, Flux's text encoder)
        text_encoder_path = None
        if model_path and model_path.is_dir():
            for f in sorted(model_path.iterdir()):
                if f.suffix in (".safetensors", ".pth", ".pt"):
                    name_lower = f.name.lower()
                    if ("t5" in name_lower or "umt5" in name_lower
                            or "text_encoder" in name_lower):
                        # Prefer safetensors over pth to avoid format mismatch
                        if f.suffix == ".safetensors":
                            text_encoder_path = str(f)
                        elif text_encoder_path is None:
                            text_encoder_path = str(f)

        pipeline, pipe_wrapper = handler.load_model(
            model_filename, model_type, base_model_type, model_def,
            quantizeTransformer=not is_cpu,
            text_encoder_quantization="int8" if not is_cpu else None,
            dtype=None if is_cpu else torch.bfloat16,
            VAE_dtype=None if is_cpu else torch.float32,
            profile=0 if is_cpu else MMGP_PROFILES["balanced"],
            quant=quant,
            text_encoder_filename=text_encoder_path,
        )

        pipe, co_tenants = self._unwrap_pipe(pipe_wrapper)

        self._apply_mmgp_profile(pipe, co_tenants, is_cpu, model_type)

        self._models[model_name] = {
            "model": pipeline,
            "pipe": pipe,
            "info": entry,
            "loaded_at": time.time(),
        }

    def _ensure_vendor_files(self, handler, base_model_type: str,
                              model_def: dict) -> None:
        """Download missing model files via handler's query_model_files."""
        ckpts_base = Path("ckpts").resolve()
        ckpts_base.mkdir(parents=True, exist_ok=True)

        # Step 1: query_model_files for supporting files (VAE, text encoders)
        qmf = getattr(handler, "query_model_files", None)
        if qmf is not None:
            try:
                download_defs = qmf([], base_model_type, model_def)
            except Exception:
                download_defs = []
            if not isinstance(download_defs, list):
                download_defs = [download_defs]

            from huggingface_hub import hf_hub_download
            for entry in (download_defs or []):
                repo_id = entry.get("repoId")
                source_folders = entry.get("sourceFolderList", [])
                file_lists = entry.get("fileList", [])
                if not repo_id:
                    continue
                for folder, files in zip(source_folders, file_lists):
                    for fname in files:
                        rel = Path(folder) / fname if folder else Path(fname)
                        local = ckpts_base / rel
                        if local.exists():
                            continue
                        try:
                            hf_hub_download(
                                repo_id, str(rel),
                                local_dir=str(ckpts_base / folder) if folder else str(ckpts_base),
                                local_dir_use_symlinks=False,
                            )
                            logger.info("Downloaded %s/%s → %s", repo_id, rel, local)
                        except Exception as e:
                            logger.debug("Download failed %s/%s: %s", repo_id, rel, e)

        # Step 2: check defaults URL for the main model file, download to ckpts
        self._ensure_main_model(handler, base_model_type, ckpts_base)

    def _ensure_main_model(self, handler, base_model_type: str,
                            ckpts_base: Path) -> None:
        """Download the main model file from the defaults URL if missing."""
        defaults_file = Path.home() / ".wan2gp" / "defaults" / f"{base_model_type}.json"
        alt = Path("/opt/wan2gp/defaults") / f"{base_model_type}.json"
        if not defaults_file.exists() and alt.exists():
            defaults_file = alt
        if not defaults_file.exists():
            return
        try:
            import json
            with open(defaults_file) as f:
                defaults = json.load(f)
            urls = defaults.get("model", {}).get("URLs", [])
        except Exception:
            return
        if not urls:
            return

        from huggingface_hub import hf_hub_download
        import re

        # Determine model path from registry
        from registry.models import ModelRegistry
        try:
            reg = ModelRegistry()
            model_dir = reg.get_path("wan2gp", base_model_type.replace("_", "-"))
            model_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            model_dir = ckpts_base

        # Prefer quantized variants (smaller downloads) — prefer *quanto* variants
        sorted_urls = sorted(urls, key=lambda u: 0 if "quanto" in u else 1)

        for url in sorted_urls:
            m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/main/(.+)", url)
            if not m:
                continue
            repo_id = m.group(1)
            filename = m.group(2)

            # Check both ckpts and model_dir
            if (ckpts_base / filename).exists() or (model_dir / filename).exists():
                continue

            try:
                hf_hub_download(
                    repo_id, filename,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                logger.info("Downloaded main model %s → %s", url, model_dir / filename)
                # Symlink to ckpts so files_locator finds it
                if not (ckpts_base / filename).exists():
                    (ckpts_base / filename).symlink_to(model_dir / filename)
            except Exception as e:
                logger.debug("Main model download failed %s: %s", url, e)

    @staticmethod
    def _unwrap_pipe(pipe_wrapper) -> tuple[dict, dict]:
        if isinstance(pipe_wrapper, dict):
            return pipe_wrapper.get("pipe", pipe_wrapper), pipe_wrapper.get("coTenantsMap", {})
        return {}, {}

    # Models whose handlers manage GPU memory internally (no mmgp needed)
    _NO_MMGP_MODELS = {"pixal3d", "anigen"}

    @staticmethod
    def _apply_mmgp_profile(pipe: dict, co_tenants: dict, is_cpu: bool,
                            model_type: str) -> None:
        if not pipe or is_cpu:
            return
        if model_type in Wan2GPService._NO_MMGP_MODELS:
            logger.info("Skipping mmgp for %s (self-managed GPU memory)", model_type)
            return
        from mmgp import offload

        # Normalize dtypes — mmgp asserts all pipe modules share one dtype
        target_dtype = torch.bfloat16
        for k, v in pipe.items():
            if not isinstance(v, torch.nn.Module):
                continue
            dtypes = {p.dtype for p in v.parameters() if p.dtype != target_dtype}
            if dtypes:
                v.to(target_dtype)

        offload.profile(
            pipe,
            profile_no=MMGP_PROFILES["balanced"],
            quantizeTransformer=False,
            budgets={"transformer": 250, "text_encoder": 250, "*": 3000},
            loras=[],
            perc_reserved_mem_max=0.5,
            vram_safety_coefficient=0.9,
            coTenantsMap=co_tenants,
        )

    def _resolve_model_path(self, model_name: str, entry: dict,
                             registry, cfg) -> Path | None:
        model_type = entry.get("model_type", model_name)

        # Try wan2gp registry section first
        model_key_safe = model_name.replace("/", "-")
        try:
            path = registry.get_path("wan2gp", model_key_safe)
            if Path(path).is_dir():
                return Path(path)
        except (KeyError, FileNotFoundError):
            pass

        # Try model-specific registry paths
        for svc_type, reg_name in _WEIGHT_SEARCH.get(model_type, []):
            try:
                path = registry.get_path(svc_type, reg_name)
                if Path(path).is_dir():
                    return Path(path)
            except (KeyError, FileNotFoundError):
                pass

        # Try models_root/wan2gp/<model_type>
        fallback = Path(cfg.models_root) / "wan2gp" / model_type
        if fallback.is_dir():
            return fallback

        # Fallback: ckpts/ (writable, for auto-downloaded vendor models)
        ckpts = Path("ckpts").resolve()
        if any(ckpts.glob("*.safetensors")) or any(ckpts.glob("*.pth")):
            return ckpts

        # CPU models may not have weight files
        if model_type in _CPU_ONLY_TYPES:
            return None

        return None

    def _try_download(self, model_type: str) -> bool:
        """Try to auto-download model weights from registry source URL."""
        from registry.config import Config
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        cfg = Config()

        # Find the registry entry for this model type
        search_pairs = [("wan2gp", model_type.replace("/", "-"))]
        for svc_type, reg_name in _WEIGHT_SEARCH.get(model_type, []):
            search_pairs.append((svc_type, reg_name))

        for svc_type, reg_name in search_pairs:
            try:
                meta = registry.get_metadata(svc_type, reg_name)
                source = meta.get("source", "")
                download = meta.get("download", "skip")
                if not source or download == "skip" or download == "manual":
                    continue

                logger.info("Auto-downloading %s/%s from %s", svc_type, reg_name, source)

                if source.startswith("hf://"):
                    repo_id = source[5:]
                    target_path = registry.get_path(svc_type, reg_name)
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    if download == "snapshot":
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id, local_dir=str(target_path))
                        logger.info("Downloaded %s → %s", repo_id, target_path)
                        return True
                    elif download == "file":
                        from huggingface_hub import hf_hub_download
                        filename = Path(meta.get("path", "")).name
                        hf_hub_download(repo_id, filename, local_dir=str(target_path.parent))
                        logger.info("Downloaded %s/%s → %s", repo_id, filename, target_path)
                        return True
                elif source.startswith("modelscope://"):
                    repo_id = source[13:]
                    target_path = registry.get_path(svc_type, reg_name)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    from modelscope import snapshot_download as ms_download
                    ms_download(repo_id, cache_dir=str(target_path.parent))
                    logger.info("Downloaded from ModelScope: %s", repo_id)
                    return True

            except (KeyError, FileNotFoundError):
                continue
            except Exception as e:
                logger.warning("Auto-download failed for %s/%s: %s", svc_type, reg_name, e)
                return False

        return False

    # Spec-aware model type → spec name mapping
    _SPEC_AWARE = {
        "moss-soundeffect": "moss",
        "moss-tts": "moss_tts",
        "moss-ttsd": "moss_ttsd",
        "moss-voicegenerator": "moss_voicegenerator",
        "trellis": "trellis",
        "kokoro": "kokoro",
        "faster_whisper": "faster_whisper",
        "vibevoice-asr": "vibevoice_asr",
        "vibevoice-tts": "vibevoice_tts",
        "anigen": "anigen",
        "see-through": "see_through",
        "hy-motion-1.0": "hy_motion",
        "hy-motion-1.0-lite": "hy_motion_lite",
        "pixal3d": "pixal3d",
    }

    # model_type → [(output_key, spec_module_key, registry_category, registry_name), ...]
    _PATH_MAP = {
        "moss-soundeffect": [
            ("moss_soundeffect_path", "language_model", "audio", "moss-soundeffect"),
            ("moss_audio_tokenizer_path", "audio_tokenizer", "audio", "moss-audio-tokenizer"),
        ],
        "moss-tts": [
            ("moss_tts_path", "language_model", "audio", "moss-tts"),
            ("moss_audio_tokenizer_path", "audio_tokenizer", "audio", "moss-audio-tokenizer"),
        ],
        "moss-ttsd": [
            ("moss_ttsd_path", "language_model", "audio", "moss-ttsd"),
            ("moss_audio_tokenizer_path", "audio_tokenizer", "audio", "moss-audio-tokenizer"),
        ],
        "moss-voicegenerator": [
            ("moss_voicegenerator_path", "language_model", "audio", "moss-voicegenerator"),
            ("moss_audio_tokenizer_path", "audio_tokenizer", "audio", "moss-audio-tokenizer"),
        ],
        "kokoro": [("kokoro_path", "model", "tts", "kokoro")],
        "faster_whisper": [("faster_whisper_path", "model", "asr", "faster-whisper")],
        "vibevoice-asr": [("vibevoice_asr_path", "language_model", "asr", "vibevoice-asr")],
        "vibevoice-tts": [("vibevoice_tts_path", "language_model", "tts", "vibevoice-tts")],
        "anigen": [("anigen_path", "pipeline_root", "3d", "anigen")],
        "faster-qwen3-tts": [("faster_qwen3_tts_path", None, "tts", "qwen3-tts")],
    }

    # Models with multiple paths sharing the same pattern
    _MULTI_PATH_MAP = {
        "see-through": [
            ("see_through_layerdiff_path", "layerdiff", "image", "see-through-layerdiff"),
            ("see_through_marigold_path", "marigold", "image", "see-through-marigold"),
            ("see_through_scheduler_path", "scheduler", "image", "see-through-scheduler"),
        ],
    }

    def _resolve_handler_paths(self, model_type: str, registry, cfg, *,
                               quant: str | None = None) -> dict:
        """Resolve extra paths for custom handlers and return as a flat dict."""
        paths = {}

        # Try spec resolver for spec-aware models
        spec_name = self._SPEC_AWARE.get(model_type)
        spec = None
        if spec_name:
            try:
                from registry.specs import resolve
                spec = resolve(spec_name, quant=quant)
            except Exception:
                pass

        # Data-driven path resolution
        path_entries = self._PATH_MAP.get(model_type) or self._MULTI_PATH_MAP.get(model_type)
        if path_entries:
            for out_key, spec_key, reg_cat, reg_name in path_entries:
                if spec and spec_key:
                    paths[out_key] = spec["modules"].get(spec_key, "")
                else:
                    try:
                        paths[out_key] = registry.get_path(reg_cat, reg_name)
                    except Exception:
                        pass

        # Special cases needing custom logic
        elif model_type.startswith("hy-motion"):
            self._resolve_hymotion_paths(paths, model_type, cfg, spec)

        # Espeak binary path (always resolved)
        try:
            paths["espeak_bin"] = cfg.get("binaries.espeak_ng", "espeak-ng")
        except Exception:
            pass

        return paths

    def _resolve_hymotion_paths(self, paths: dict, model_type: str,
                                cfg, spec: dict | None) -> None:
        if spec:
            paths["hy_motion_path"] = spec["modules"].get("pipeline_root", "")
            text_enc = spec["modules"].get("text_encoder", "")
            if text_enc:
                paths["hy_motion_Qwen3_8B_path"] = text_enc
        else:
            mp = Path(cfg.models_root) / "motion" / model_type
            if mp.is_dir():
                paths["hy_motion_path"] = str(mp)
                for sub in ["Qwen3-8B", "clip-vit-large-patch14"]:
                    p = mp / "ckpts" / sub
                    if p.is_dir():
                        paths[f"hy_motion_{sub.replace('-','_').replace('.','_')}_path"] = str(p)

    def _build_model_def(self, handler, base_model_type: str,
                          model_path: Path | None) -> dict:
        base = {}
        if model_path:
            base["text_encoder_folder"] = str(model_path / "text_encoder") if (model_path / "text_encoder").is_dir() else None
            te_urls_file = model_path / "text_encoder_urls.json"
            if te_urls_file.exists():
                base["text_encoder_URLs"] = json.loads(te_urls_file.read_text())
            base["profiles_dir"] = [base_model_type]
            base["group"] = base_model_type

        return handler.query_model_def(base_model_type, base)

    # model_type → (variant_dir_name, default_filename) for transformer weights
    _TRANSFORMER_WEIGHTS = {
        "ace_step_v1_5": ("acestep-v15-turbo", "model.safetensors"),
        "ace_step_v1_5_xl": ("acestep-v15-xl-turbo", "model.safetensors"),
        "ace_step_v1": ("ace_step", None),
    }

    def _resolve_model_filename(self, model_type: str, base_model_type: str,
                                 model_path: Path | None, model_def: dict):
        """Resolve transformer weights file path for models that need it via model_filename."""
        if model_path is None:
            return []

        # Check model_def for explicit path first
        explicit = model_def.get("transformer_weights_path")
        if explicit and Path(explicit).is_file():
            return [explicit]

        # Check transformer weights mapping
        tw_info = self._TRANSFORMER_WEIGHTS.get(base_model_type)
        if tw_info is not None:
            variant_dir, default_file = tw_info
            for candidate in [
                model_path / variant_dir / (default_file or "model.safetensors"),
                model_path / "transformer" / "model.safetensors",
            ]:
                if candidate.is_file():
                    return [str(candidate)]
            if (model_path / variant_dir).is_dir():
                for f in (model_path / variant_dir).glob("*.safetensors"):
                    return [str(f)]

        # Fallback for general vendor models: find the main model file
        # (largest safetensors that isn't VAE, T5, or encoder)
        exclude_patterns = ("VAE", "vae", "t5", "T5", "umt5", "clip", "CLIP",
                           "text_encoder", "encoder")
        candidates = sorted(model_path.glob("*.safetensors"),
                           key=lambda p: p.stat().st_size, reverse=True)
        for f in candidates:
            name = f.name
            if any(p in name for p in exclude_patterns):
                continue
            if f.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                return [str(f)]
        return []

    def _encode_output(self, output, payload: dict, defaults: dict) -> dict:
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
            raise RuntimeError("Model returned no output")

        frames_np = frames_tensor.cpu().numpy() if isinstance(frames_tensor, torch.Tensor) else frames_tensor
        msg = f"encode_input: shape={frames_np.shape} dtype={frames_np.dtype} min={frames_np.min():.2f} max={frames_np.max():.2f}"
        logger.info("Wan2GP: %s", msg)
        print(f"[Wan2GP] {msg}", flush=True)

        if frames_np.dtype != np.uint8:
            frames_np = ((frames_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

        # Wan models output (C, F, H, W) — transpose to (F, H, W, C)
        if frames_np.ndim == 4:
            if frames_np.shape[0] in (1, 3, 4) and frames_np.shape[1] > 4:
                logger.info("Wan2GP: transpose CFHW→FHWC")
                frames_np = frames_np.transpose(1, 2, 3, 0)  # (F, H, W, C)
            elif frames_np.shape[3] not in (1, 3, 4):
                logger.info("Wan2GP: transpose FCHW→FHWC")
                frames_np = frames_np.transpose(0, 2, 3, 1)  # (F, H, W, C)
        elif frames_np.ndim == 3 and frames_np.shape[0] in (1, 3, 4):
            logger.info("Wan2GP: transpose CHW→HWC")
            frames_np = frames_np.transpose(1, 2, 0)     # (H, W, C)
        logger.info("Wan2GP: final shape=%s dtype=%s", frames_np.shape, frames_np.dtype)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        if frames_np.ndim == 4 and frames_np.shape[0] > 1:
            import imageio
            fps = int(payload.get("fps", defaults.get("fps", 16)))
            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            for f in frames_np:
                writer.append_data(f)
            writer.close()
        else:
            from PIL import Image as PILImage
            img = frames_np[0] if frames_np.ndim == 4 else frames_np
            PILImage.fromarray(img).save(tmp_path, format="PNG")

        with open(tmp_path, "rb") as f:
            data_bytes = f.read()
        os.unlink(tmp_path)

        result = {
            "status": "ok",
            "data": base64.b64encode(data_bytes).decode(),
            "media_type": "video/mp4" if frames_np.ndim == 4 and frames_np.shape[0] > 1 else "image/png",
        }

        if audio is not None:
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            import soundfile as sf
            import io as audio_io
            audio_buf = audio_io.BytesIO()
            sf.write(audio_buf, audio_np, 24000, format="WAV")
            result["audio_b64"] = base64.b64encode(audio_buf.getvalue()).decode()

        return result


# ─── Payload Passthrough Helpers ───────────────────────────────────────────────

def _build_generate_kwargs(payload: dict, defaults: dict, key_map: dict | None = None) -> dict:
    merged_map = dict(_KEY_MAP)
    if key_map:
        merged_map.update(key_map)

    kwargs = {}

    for k, v in defaults.items():
        kwargs[k] = v

    for src, dst in merged_map.items():
        if src in payload:
            kwargs[dst] = payload[src]

    for key in _SAFE_PASSTHROUGH:
        if key in payload:
            kwargs[key] = payload[key]

    for k in defaults:
        if k in payload:
            kwargs[k] = payload[k]

    if "seed" not in kwargs:
        kwargs["seed"] = -1

    for key in _BLOCKED_KEYS:
        if key in payload:
            logger.debug("Blocked key in payload: %s", key)

    return kwargs
