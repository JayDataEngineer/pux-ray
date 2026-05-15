"""Wan2GP Service — Wan2GP-native model discovery, mmgp-managed VRAM.

All models (vendor + custom families) discovered via Wan2GP's refresh_model_defs()
and map_family_handlers(). No parallel model_engine system.

Architecture:
    - 19 vendor handlers + 12 custom family handlers in family_handlers list
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

WAN2GP_VENDOR = Path(__file__).parents[2] / "vendor" / "wan2gp"

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

# ─── Vendor Path Setup ─────────────────────────────────────────────────────────

_ven_loaded = False


def _ensure_vendor_path():
    global _ven_loaded
    if _ven_loaded:
        return
    vendor = str(WAN2GP_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    custom_models = str(Path(__file__).parent / "custom_models")
    if os.path.isdir(custom_models) and custom_models not in sys.path:
        sys.path.insert(0, custom_models)
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
    "trellis.trellis_handler",
    "anigen_handler.anigen_handler",
    "see_through.see_through_handler",
    "hy_motion.hy_motion_handler",
    "kokoro.kokoro_handler",
    "moss.moss_handler",
    "espeak.espeak_handler",
    "faster_whisper.faster_whisper_handler",
    "vibevoice_asr.vibevoice_asr_handler",
    "vibevoice_tts.vibevoice_tts_handler",
    "faster_qwen3_tts.faster_qwen3_tts_handler",
]

def _get_family_handlers() -> list[str]:
    """Get the family_handlers list from Wan2GP's wgp.py, plus our custom handlers."""
    try:
        import wgp
        base = wgp.family_handlers
    except (ImportError, AttributeError):
        base = [
            "models.wan.wan_handler", "models.wan.ovi_handler", "models.wan.df_handler",
            "models.hyvideo.hunyuan_handler", "models.ltx_video.ltxv_handler",
            "models.ltx2.ltx2_handler", "models.longcat.longcat_handler",
            "models.flux.flux_handler", "models.qwen.qwen_handler",
            "models.kandinsky5.kandinsky_handler", "models.z_image.z_image_handler",
            "models.magi_human.magi_human_handler",
            "models.TTS.ace_step_handler", "models.TTS.chatterbox_handler",
            "models.TTS.qwen3_handler", "models.TTS.yue_handler",
            "models.TTS.heartmula_handler", "models.TTS.kugelaudio_handler",
            "models.TTS.index_tts2_handler",
            "models.vnccs.vnccs_handler",
        ]
    # Append our custom handlers living in services/wan2gp/custom_models/
    for h in CUSTOM_HANDLERS:
        if h not in base:
            base.append(h)
    return base


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
    "trellis": [("3d", "trellis")],
    "anigen": [("3d", "anigen")],
    "moss-soundeffect": [("audio", "moss-soundeffect")],
    "see-through": [("image", "see-through-layerdiff"), ("image", "see-through-marigold")],
    "hy-motion-1.0": [("motion", "hy-motion-1.0")],
    "hy-motion-1.0-lite": [("motion", "hy-motion-1.0-lite")],
    "vibevoice-asr": [("asr", "vibevoice-asr")],
    "vibevoice-tts": [("tts", "vibevoice-tts")],
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

    def load(self, model_name: str | None = None) -> None:
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
        if entry.get("blocked"):
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
            else:
                raise RuntimeError(
                    f"Model '{model_name}' is blocked: {entry.get('blocked_reason', 'unknown')}"
                )

        try:
            self._load_model(model_name, entry)
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

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_model(self, model_name: str, entry: dict) -> None:
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

        # Resolve model path
        model_path = self._resolve_model_path(model_name, entry, model_registry, cfg)

        # Build model_def
        model_def = self._build_model_def(handler, base_model_type, model_path)

        # Determine if this is a CPU-only model
        is_cpu = model_type in _CPU_ONLY_TYPES
        dtype = None if is_cpu else torch.bfloat16
        vae_dtype = None if is_cpu else torch.float32
        profile = 0 if is_cpu else MMGP_PROFILES["balanced"]

        logger.info("Loading %s from %s (family_handler)", model_name, model_path or "N/A")
        torch.set_default_device("cpu")

        pipeline, pipe_wrapper = handler.load_model(
            [] if model_path is None else [],
            model_type,
            base_model_type,
            model_def,
            quantizeTransformer=not is_cpu,
            text_encoder_quantization="int8" if not is_cpu else None,
            dtype=dtype,
            VAE_dtype=vae_dtype,
            profile=profile,
        )

        # Unwrap pipe dict
        if isinstance(pipe_wrapper, dict):
            pipe = pipe_wrapper.get("pipe", pipe_wrapper)
            co_tenants = pipe_wrapper.get("coTenantsMap", {})
        else:
            pipe = {}
            co_tenants = {}

        # mmgp profile for GPU models with nn.Modules
        if pipe and not is_cpu:
            from mmgp import offload
            budgets = {"transformer": 250, "text_encoder": 250, "*": 3000}
            offload.profile(
                pipe,
                profile_no=profile,
                quantizeTransformer=False,
                budgets=budgets,
                loras=[],
                perc_reserved_mem_max=0.5,
                vram_safety_coefficient=0.9,
                coTenantsMap=co_tenants,
            )

        self._models[model_name] = {
            "model": pipeline,
            "pipe": pipe,
            "info": entry,
            "loaded_at": time.time(),
        }

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

    def _build_model_def(self, handler, base_model_type: str, model_path: Path | None) -> dict:
        base = {}
        if model_path:
            base["text_encoder_folder"] = str(model_path / "text_encoder") if (model_path / "text_encoder").is_dir() else None
            te_urls_file = model_path / "text_encoder_urls.json"
            if te_urls_file.exists():
                base["text_encoder_URLs"] = json.loads(te_urls_file.read_text())
            base["profiles_dir"] = [base_model_type]
            base["group"] = base_model_type

        return handler.query_model_def(base_model_type, base)

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
