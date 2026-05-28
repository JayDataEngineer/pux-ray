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
    # Universal text input: any of these → input_prompt (and aliased to 'text' below)
    "prompt": "input_prompt",
    "input": "input_prompt",
    "text": "input_prompt",
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
    "input_prompt", "max_tokens", "reference", "tokens",
    "prompts", "num_frames", "num_denoising_steps", "post_processing",
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

    # Handlers that bundle vendor packages as subdirectories need their
    # parent on sys.path so the inner package is importable at top level.
    for subdir in (WAN2GP_VENDOR / "models").iterdir():
        if not subdir.is_dir():
            continue
        for nested in subdir.iterdir():
            if nested.is_dir() and (nested / "__init__.py").exists():
                inner_name = nested.name
                try:
                    importlib.import_module(inner_name)
                except ImportError:
                    if str(subdir) not in sys.path:
                        sys.path.insert(0, str(subdir))

    _ven_loaded = True


def _ensure_quantized_cache():
    import transformers.cache_utils as _tcu
    if not hasattr(_tcu, "QuantizedCacheConfig"):
        _tcu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {
            "__init__": lambda self, **kw: None,
            "__getattr__": lambda self, name: None,
        })


def _patch_quanto_compat():
    """Patch optimum-quanto WeightQBytesTensor to handle old serialization format.

    optimum-quanto <= 0.2.2 stored fields with leading underscores (_qtype, _axis,
    _data, _size, _stride, _scale, _is_param, etc.) in safetensors metadata.
    v0.2.3+ expects them without underscores. Old quantized files fail with
    "unexpected keyword argument '_...'" when loaded in a process where
    WeightQBytesTensor is a safetensors tensor subclass.
    """
    try:
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
    except ImportError:
        return

    _orig_new = WeightQBytesTensor.__new__

    @staticmethod
    def _compat_new(cls, *args, **kwargs):
        # Remap any _underscore kwargs to their non-underscore equivalents
        cleaned = {}
        for k, v in kwargs.items():
            if k.startswith('_') and len(k) > 1 and k[1:] not in kwargs:
                cleaned[k[1:]] = v
            else:
                cleaned[k] = v
        return _orig_new(cls, *args, **cleaned)

    if not getattr(WeightQBytesTensor, "_qtype_compat_patched", False):
        WeightQBytesTensor.__new__ = _compat_new
        WeightQBytesTensor._qtype_compat_patched = True


def _ensure_transformers_compat():
    """Patch renamed/removed transformers symbols that vendor handlers depend on."""
    import transformers.generation.configuration_utils as _gcu
    if not hasattr(_gcu, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
        _gcu.NEED_SETUP_CACHE_CLASSES_MAPPING = getattr(
            _gcu, "ALL_CACHE_IMPLEMENTATIONS", ()
        )
    if not hasattr(_gcu, "QUANT_BACKEND_CLASSES_MAPPING"):
        _gcu.QUANT_BACKEND_CLASSES_MAPPING = {}

    # QuantizedCacheConfig was removed in transformers 4.57+
    import transformers.cache_utils as _cu
    if not hasattr(_cu, "QuantizedCacheConfig"):
        _cu.QuantizedCacheConfig = type("QuantizedCacheConfig", (), {})


def _ensure_writable_hf_cache():
    """Redirect HF cache env vars and torch hub to /tmp if the PVC is read-only."""
    import tempfile
    writable = Path(tempfile.gettempdir()) / "hf_cache"
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        val = os.environ.get(var, "")
        if val and not os.access(val, os.W_OK):
            writable.mkdir(parents=True, exist_ok=True)
            os.environ[var] = str(writable)
            logger.info("%s redirected to writable %s (PVC read-only)", var, writable)
    # torch.hub writes trusted_list to its hub dir — redirect to /tmp when PVC is read-only
    try:
        import torch
        hub_dir = torch.hub.get_dir()
        if not os.access(hub_dir, os.W_OK):
            writable_hub = Path(tempfile.gettempdir()) / "torch_hub"
            writable_hub.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(writable_hub))
            logger.info("torch.hub redirected to %s (PVC read-only)", writable_hub)
    except Exception:
        pass


def _patch_anigen_decoder():
    """Patch anigen_decoder.py to force float32 in skeleton_grouping scatter ops.

    spconv bf16→fp32 hooks create mixed dtypes between skin_feats_skl (bf16)
    and scatter target tensors. Forces float32 on skin_feats and conf_skin
    before scatter_add_ operations.

    Uses path-based file discovery (no import) so the patched file is available
    when the module is first imported during pipeline construction.
    """
    import pathlib
    # Find the vendored anigen_decoder.py relative to /opt/vendor
    for base in ("/opt/vendor", "/app/vendor"):
        vendor_anigen = pathlib.Path(base) / "anigen" / "models" / "structured_latent_vae"
        if vendor_anigen.is_dir():
            break
    else:
        return
    ad_path = vendor_anigen / "anigen_decoder.py"
    if not ad_path.is_file():
        return
    try:
        _src = ad_path.read_text()
        marker = "skin_feats_skl = rep_skl.skin_feats if skin_feats_skl_list is None else skin_feats_skl_list[i]\n"
        if marker not in _src:
            return
        _src = _src.replace(
            marker,
            "skin_feats_skl = (rep_skl.skin_feats if skin_feats_skl_list is None else skin_feats_skl_list[i]).float()\n",
        )
        _src = _src.replace(
            "conf_skin = torch.sigmoid(rep_skl.conf_skin) if rep_skl.conf_skin is not None else torch.ones_like(skin_feats_skl[:, :1])\n",
            "conf_skin = (torch.sigmoid(rep_skl.conf_skin) if rep_skl.conf_skin is not None else torch.ones_like(skin_feats_skl[:, :1])).float()\n",
        )
        ad_path.write_text(_src)
        # Invalidate cached pyc
        for pyc in vendor_anigen.glob("__pycache__/anigen_decoder*.pyc"):
            pyc.unlink(missing_ok=True)
        logger.info("Patched anigen_decoder for float32 scatter ops")
    except Exception:
        logger.debug("Could not patch anigen_decoder", exc_info=True)


def _patch_anigen_grouping():
    """Patch grouping.py to match scatter_add zero-tensor dtype with source dtype.

    torch.zeros((Nj, 3), device=joints.device) creates float32 by default,
    but joints/parents/conf are bf16 from mmgp. scatter_add requires matching
    dtypes between self and src.
    """
    import pathlib
    for base in ("/opt/vendor", "/app/vendor"):
        grouping_path = pathlib.Path(base) / "anigen" / "representations" / "skeleton" / "grouping.py"
        if grouping_path.is_file():
            break
    else:
        return
    try:
        _src = grouping_path.read_text()
        if "dtype=joints.dtype" in _src:
            return  # already patched
        _src = _src.replace(
            "torch.zeros((Nj, 3), device=joints.device)",
            "torch.zeros((Nj, 3), device=joints.device, dtype=joints.dtype)",
        )
        _src = _src.replace(
            "torch.zeros((Nj, 1), device=joints.device)",
            "torch.zeros((Nj, 1), device=joints.device, dtype=joints.dtype)",
        )
        grouping_path.write_text(_src)
        pyc_dir = grouping_path.parent / "__pycache__"
        if pyc_dir.is_dir():
            for pyc in pyc_dir.glob("grouping*.pyc"):
                pyc.unlink(missing_ok=True)
        logger.info("Patched grouping.py for dtype-matched scatter ops")
    except Exception:
        logger.debug("Could not patch grouping.py", exc_info=True)


def _patch_anigen_cube2mesh():
    """Patch cube2mesh_skeleton.py to convert vertices to float32 before
    creating AniGenMeshExtractResult. mmgp converts params to bf16, and the
    decoder output vertices inherit bf16, which causes scatter() dtype
    mismatches in flexicubes and scatter_add_ in comput_v_normals.
    """
    import pathlib
    for base in ("/opt/vendor", "/app/vendor"):
        vendor_dir = pathlib.Path(base) / "anigen" / "representations" / "mesh"
        if vendor_dir.is_dir():
            break
    else:
        return
    cube_path = vendor_dir / "cube2mesh_skeleton.py"
    if not cube_path.is_file():
        return
    try:
        _src = cube_path.read_text()
        # In the decode_to_mesh method, convert vertices to float32 before
        # constructing AniGenMeshExtractResult.
        old_line = "mesh = AniGenMeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=rgbnormal_colors, vertex_skin_feats=vertex_skin_feats, grid_positions=v_pos_normalized, grid_skin_feats=grid_skin_feats, res=self.res)\n"
        new_line = "mesh = AniGenMeshExtractResult(vertices=vertices.float(), faces=faces, vertex_attrs=rgbnormal_colors, vertex_skin_feats=vertex_skin_feats, grid_positions=v_pos_normalized, grid_skin_feats=grid_skin_feats, res=self.res)\n"
        if old_line not in _src:
            return
        _src = _src.replace(old_line, new_line)
        cube_path.write_text(_src)
        for pyc in vendor_dir.glob("__pycache__/cube2mesh_skeleton*.pyc"):
            pyc.unlink(missing_ok=True)
        logger.info("Patched cube2mesh_skeleton for float32 vertices")
    except Exception:
        logger.debug("Could not patch cube2mesh_skeleton", exc_info=True)


# ─── Dynamic Model Discovery ──────────────────────────────────────────────────

def discover_models(models_root: Path | None = None) -> dict:
    """Discover all models via Wan2GP's native family_handlers system.

    Imports each handler from the family_handlers list, calls
    query_supported_types() to get model types, checks for weight files.
    """
    from registry.config import Config
    from registry.models import ModelRegistry
    from services.wan2gp.custom_models.base_handler import register_handler_meta

    cfg = Config()
    models_root = models_root or Path(cfg.models_root)
    registry = ModelRegistry()
    _ensure_vendor_path()

    # Initialize wgp globals so handlers that reference wgp.get_lora_root()
    # during query_supported_types() don't crash on None server_config.
    Wan2GPService._init_wan2gp()

    discovered = {}

    # Wan2GP's family_handlers list (from wgp.py, includes our additions)
    family_handlers_list = _get_family_handlers()

    for handler_path in family_handlers_list:
        try:
            handler_mod = importlib.import_module(handler_path)
            handler = handler_mod.family_handler

            handler_meta = getattr(handler_mod, "HANDLER_META", None)
            supported = handler.query_supported_types()

            for model_type in sorted(supported):
                model_key = _derive_key(model_type, handler_path)

                # Register HANDLER_META for this type
                if handler_meta is not None:
                    register_handler_meta(model_type, handler_meta)
                    logger.debug("Registered HANDLER_META for %s", model_type)

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
            logger.warning("Handler unavailable: %s (%s)", handler_path, e)
        except Exception as e:
            logger.warning("Handler discovery failed: %s (%s: %s)", handler_path, type(e).__name__, e)

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
    "models.moss.moss_v2_handler",
    "models.espeak.espeak_handler",
    "models.faster_whisper.faster_whisper_handler",
    "models.faster_qwen3_tts.faster_qwen3_tts_handler",
    "models.vibevoice_asr.vibevoice_asr_handler",
    "models.trellis.trellis_handler",
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
    "moss_soundeffect_v2": [("audio", "moss-soundeffect-v2")],
    "moss-tts": [("audio", "moss-tts")],
    "moss-ttsd": [("audio", "moss-ttsd")],
    "moss-voicegenerator": [("audio", "moss-voicegenerator")],
    "moss-tts-local-transformer": [("audio", "moss-tts-local-transformer")],
    "moss-tts-realtime": [("audio", "moss-tts-realtime")],
    "moss-tts-nano": [("audio", "moss-tts-nano")],
    "see-through": [("image", "see-through-layerdiff"), ("image", "see-through-marigold")],
    "hy-motion-1.0": [("motion", "hy-motion-1.0")],
    "hy-motion-1.0-lite": [("motion", "hy-motion-1.0-lite")],
    "vibevoice-asr": [("asr", "vibevoice-asr")],
    # Wan2GP vendor models — registry keys use versioned names
    "t2v":           [("wan2gp", "wan-t2v-14B")],
    "t2v_1.3B":      [("wan2gp", "wan-t2v-1.3B")],
    "i2v":           [("wan2gp", "wan-i2v-14B")],
    "t2v_2_2":       [("wan2gp", "wan-t2v-14B")],
    "i2v_2_2":       [("wan2gp", "wan-i2v-14B")],
    "trellis":       [("3d", "trellis")],
    "ace_step_v1_5": [("audio", "acestep")],
    "ace_step_v1_5_xl": [("audio", "acestep")],
    "ace_step_v1":  [("audio", "acestep")],
    "index_tts2":    [("tts", "index-tts")],
    "kokoro":        [("tts", "kokoro")],
    "faster_whisper": [("asr", "faster-whisper")],
    # Flux models
    "flux":          [("wan2gp", "flux")],
    "flux_schnell":  [("wan2gp", "flux-schnell")],
    "flux2_dev":     [("wan2gp", "flux2-dev")],
    "flux2_klein_4b": [("wan2gp", "flux2-klein-4b")],
    "flux_chroma":   [("wan2gp", "flux-chroma")],
    # Lance models
    "lance-image":     [("lance", "lance-image")],
    "lance-video":     [("lance", "lance-video")],
    "lance-image-awq": [("lance", "lance-image-awq")],
    "lance-video-awq": [("lance", "lance-video-awq")],
    # Kimodo models (auto-download from HuggingFace, needs HF_TOKEN for gated Llama)
    "kimodo-soma-rp": [("motion", "kimodo-soma-rp")],
    "kimodo-soma-seed": [("motion", "kimodo-soma-seed")],
    "kimodo-g1-rp": [("motion", "kimodo-g1-rp")],
    "kimodo-smplx-rp": [("motion", "kimodo-smplx-rp")],
    # Pixal3D
    "pixal3d": [("3d", "pixal3d")],
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

    # Fallback: models_root/wan2gp/wan/<model_type> (wan family models
    # stored under wan/ subdirectory with hyphenated names)
    for wan_name in [model_type.replace("_", "-"), model_key_safe]:
        wan_fallback = models_root / "wan2gp" / "wan" / wan_name
        if _has_weights(wan_fallback):
            return wan_fallback

    # Fallback: ckpts/ (writable, used for auto-downloaded vendor models)
    ckpts = Path("ckpts").resolve()
    if (ckpts / f"{model_type}.safetensors").is_file() or \
       (ckpts / f"{model_type}.pth").is_file():
        return ckpts

    return None


# ─── Wan2GP Service ───────────────────────────────────────────────────────────


def _set_param_by_path(module: torch.nn.Module, name: str, new_param: torch.nn.Parameter):
    """Replace a parameter in a (possibly nested) module by dotted name."""
    parts = name.split(".")
    for part in parts[:-1]:
        child = getattr(module, part, None)
        if child is None:
            return
        module = child
    module._parameters[parts[-1]] = new_param


class Wan2GPService:
    """Standalone Wan2GP service — unified model discovery via family_handlers."""

    service_name = "wan2gp"
    default_model = "wan/t2v_1.3B"

    # Aliases that _resolve_model can't figure out on its own:
    # - Cross-family names (ace_step lives under tts/, not ace_step/)
    # - Version downgrades (heavy → lite variants)
    # - "wan/t2v" resolves to 14B (exact registry match), use "wan/t2v-lite"
    #   or "wan/t2v_1.3B" for the distilled 1.3B model.
    _ALIASES = {
        "wan/i2v-14B": "wan/i2v",
        "wan/t2v-lite": "wan/t2v_1.3B",
        "hy_motion/hy-motion-1.0": "hy_motion/hy-motion-1.0-lite",
        "ace_step": "tts/ace_step_v1_5",
        "index_tts2": "tts/index_tts2",
        "see-through": "see_through/see-through",
    }

    def __init__(self, models_root: Path | None = None):
        self._registry = discover_models(models_root)
        self._offload = None
        self._loaded_model: str | None = None
        self._models: dict[str, dict] = {}
        self._vendor_ready = False
        self._native_loaded = False  # True if model loaded via wgp.load_models()

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

    def _resolve_model(self, name: str) -> str:
        """Resolve a model name to a registry key.

        Tries: exact match → alias → family/model exact → family prefix →
        substring match on model_type part.
        """
        if name in self._registry:
            return name
        if name in self._ALIASES:
            return self._ALIASES[name]
        # "z_image" → try "z_image/z_image", "trellis" → try "trellis/trellis"
        family_match = f"{name}/{name}"
        if family_match in self._registry:
            return family_match
        # "z_image" → find first key starting with "z_image/"
        for key in sorted(self._registry):
            if key.startswith(f"{name}/"):
                return key
        # Last resort: substring match on model_type after "/"
        for key in sorted(self._registry):
            parts = key.split("/", 1)
            if len(parts) == 2 and parts[1] == name:
                return key
        return name

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        logger.info("Wan2GP load() model=%s loaded=%s", model_name, self._loaded_model)

        # Resolve before comparing so short names match registry keys
        model_name = self._resolve_model(model_name)

        if model_name == self._loaded_model:
            return

        self.unload()

        entry = self._registry.get(model_name)
        if entry is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {self.available_models()}"
            )
        if entry.get("blocked") or (not entry.get("weight_path")
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
            model_type = entry.get("model_type", model_name)
            handler_path = entry.get("handler_path", "")
            is_custom = handler_path in CUSTOM_HANDLERS

            if is_custom:
                # Custom handlers have their own load_model(), mmgp profiles,
                # and inference logic. Never route through native pipeline.
                self._load_model(model_name, entry, quant=quant)
            else:
                # Vendor models: use Wan2GP's native load_models() pipeline.
                self._init_wan2gp()
                _prev_cwd = os.getcwd()
                os.chdir("/opt/wan2gp")
                try:
                    import wgp
                    is_native = model_type in wgp.models_def
                finally:
                    os.chdir(_prev_cwd)

                if is_native:
                    self._load_native(model_name, model_type)
                else:
                    self._load_model(model_name, entry, quant=quant)
        except Exception as e:
            err_msg = str(e) or repr(e) or type(e).__name__
            logger.error("Failed to load model %s: %s", model_name, err_msg, exc_info=True)
            # Clean up partial load — mmgp may have allocated GPU memory
            # that won't be freed if we just pop the model entry.
            self.unload()
            raise RuntimeError(f"Failed to load model '{model_name}': {err_msg}") from e
        self._loaded_model = model_name

        vram = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
        logger.info("Wan2GP: loaded %s (VRAM=%.0fMB)", model_name, vram)

    def unload(self) -> None:
        import gc as _gc

        # Release mmgp offloadobj — this is the critical step that unpins
        # all tensors from mmgp's internal tracking. Without this, mmgp
        # holds references to model weights even after we delete them.
        if self._offload is not None:
            try:
                self._offload.release()
            except Exception:
                pass
            self._offload = None

        # Release model references so GC can reclaim tensors.
        # Move modules to CPU first to release GPU allocations that
        # mmgp may not have fully unpinned.
        for m in self._models.values():
            model = m.get("model")
            if model is not None:
                for attr_name in list(vars(model)):
                    attr = getattr(model, attr_name, None)
                    if isinstance(attr, torch.nn.Module):
                        try:
                            attr.cpu()
                        except Exception:
                            pass
                    try:
                        delattr(model, attr_name)
                    except Exception:
                        pass
            pipe = m.get("pipe", {})
            if isinstance(pipe, dict):
                for mod in pipe.values():
                    if isinstance(mod, torch.nn.Module):
                        try:
                            mod.cpu()
                        except Exception:
                            pass
        self._models.clear()
        self._loaded_model = None
        self._native_loaded = False

        # Clear mmgp shared state caches + flush torch caches
        # (mirrors wgp.py release_model exactly)
        try:
            from mmgp import offload
            if "_cache" in offload.shared_state:
                del offload.shared_state["_cache"]
            offload.flush_torch_caches()
        except (ImportError, KeyError):
            pass

        # Force Python GC to reclaim unreferenced tensors, then free
        # CUDA cached blocks. Without gc.collect(), Python may hold
        # references in cyclic garbage that prevents CUDA memory release.
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Native Wan2GP Pipeline ────────────────────────────────────────────

    @staticmethod
    def _ensure_wgp_path():
        if "/opt/wan2gp" not in sys.path:
            sys.path.insert(0, "/opt/wan2gp")

    _wgp_initialized = False

    @staticmethod
    def _init_wan2gp():
        """Initialize Wan2GP's module-level globals for native load_models().

        Idempotent — safe to call multiple times. Sets up:
        - server_config (checkpoints_paths, quantization, profiles)
        - files_locator (model search paths)
        - model_types_handlers + models_def (handler ↔ model mapping)
        """
        if Wan2GPService._wgp_initialized:
            return
        _ensure_transformers_compat()
        _patch_quanto_compat()
        Wan2GPService._ensure_wgp_path()

        # wgp.py opens models/_settings.json at import time, so CWD must
        # be /opt/wan2gp. Change CWD, import, then restore.
        _prev_cwd = os.getcwd()
        os.chdir("/opt/wan2gp")
        _saved_argv = list(sys.argv)
        sys.argv = ["wan2gp.py"]
        try:
            import wgp
        finally:
            sys.argv[:] = _saved_argv
            os.chdir(_prev_cwd)

        from shared.utils import files_locator as fl
        from registry.config import Config

        models_root = Path(Config().models_root)
        checkpoints = [str(models_root / "wan2gp"), str(models_root)]

        if not wgp.server_config:
            wgp.server_config = {
                "attention_mode": "auto",
                "transformer_types": [],
                "transformer_quantization": "int8",
                "text_encoder_quantization": "int8",
                "lm_decoder_engine": "",
                "save_path": "outputs",
                "image_save_path": "outputs",
                "compile": "",
                "boost": 1,
                "enable_int8_kernels": 1,
                "clear_file_list": 5,
                "keep_intermediate_sliding_windows": 1,
                "enable_4k_resolutions": 0,
                "max_reserved_loras": -1,
                "vae_config": 0,
                "profile": 2,
                "video_profile": 2,
                "image_profile": 2,
                "audio_profile": 2,
                "preload_model_policy": [],
                "UI_theme": "default",
                "checkpoints_paths": checkpoints,
                "loras_root": "loras",
                "save_queue_if_crash": 1,
                "queue_color_scheme": "pastel",
                "process_queues_when_browser_unfocused": 1,
                "model_hierarchy_type": 1,
                "mmaudio_mode": 0,
                "mmaudio_persistence": 1,
                "flashvsr_mode": 0,
                "flashvsr_persistence": 1,
                "flashvsr_topk_ratio": 0.0,
                "rife_version": "v4",
                "metadata_type": "metadata",
                "mixed_precision": "0",
            }
        else:
            existing = wgp.server_config.get("checkpoints_paths", [])
            for cp in checkpoints:
                if cp not in existing:
                    existing.append(cp)
            wgp.server_config["checkpoints_paths"] = existing

        fl.set_checkpoints_paths(wgp.server_config["checkpoints_paths"])

        if not wgp.model_types_handlers:
            wgp.refresh_model_defs()
            wgp.map_family_handlers()

        Wan2GPService._wgp_initialized = True

    def _load_native(self, model_name: str, model_type: str):
        """Load via Wan2GP's native load_models() pipeline.

        Handles file resolution, T5/VAE/text encoder loading, mmgp profiling,
        and dtype management automatically — no manual path resolution needed.
        """
        self._init_wan2gp()

        # Redirect HF cache to writable /tmp if the models PVC is read-only.
        # Wan2GP's download_models() uses hf_hub_download with local_dir
        # pointing under /models, which is read-only on K3s.
        _ensure_writable_hf_cache()

        # Some handlers write runtime configs to the model directory.
        # If the PVC is read-only, create a writable overlay in /tmp.
        self._ensure_writable_overlay(model_type)

        # Patch Wan2GP's download root to /tmp when the models dir is read-only.
        # hf_hub_download creates .cache/huggingface in local_dir which fails
        # on read-only PVCs.
        from shared.utils import files_locator as _fl
        _orig_get_smart_download_root = _fl.get_smart_download_root
        if _fl._checkpoints_paths and not os.access(_fl._checkpoints_paths[0], os.W_OK):
            def _patched_get_smart_download_root(force_path=None, _orig=_orig_get_smart_download_root):
                result = _orig(force_path)
                if result and not os.access(result, os.W_OK):
                    return "/tmp"
                return result
            _fl.get_smart_download_root = _patched_get_smart_download_root

        # wgp.py expects CWD to be /opt/wan2gp for relative paths like
        # models/_settings.json and ckpts/
        _prev_cwd = os.getcwd()
        os.chdir("/opt/wan2gp")
        try:
            import wgp
            from mmgp import offload as _moff

            # Remove mmgp hooks from the previous native model's modules.
            # release_model() nullifies the offloadobj but leaves forward hooks
            # attached to nn.Modules. When the next model loads, these stale
            # hooks still reference the released offloadobj (self.models=None),
            # causing TypeError on any forward pass.
            if self._offload is not None:
                try:
                    self._offload.unload_all()
                except Exception:
                    pass
            for m_entry in self._models.values():
                _model = m_entry.get("model")
                if _model is not None:
                    for _mod in _model.modules():
                        if hasattr(_mod, '_mm_lora_old_forward'):
                            _mod.forward = _mod._mm_lora_old_forward
                            del _mod._mm_lora_old_forward
                        if hasattr(_mod, '_mm_forward'):
                            del _mod._mm_forward

            wgp.release_model()

            # Purge all mmgp state between model switches.
            _moff.shared_state.pop("_cache", None)
            _moff.last_offload_obj = None
            gc.collect()

            # Some Wan2GP modules (e.g. openvoice_app.py in index_tts2) call
            # argparse.ArgumentParser.parse_args() at module import time
            # without args=, so they read sys.argv. In the Ray worker process,
            # sys.argv contains Ray args (--node-ip-address etc.) which argparse
            # rejects with sys.exit(2). Temporarily replace sys.argv with a
            # safe value during model loading to prevent this crash.
            _saved_argv = list(sys.argv)
            sys.argv = ["wan2gp.py"]
            try:
                wan_model, offloadobj = wgp.load_models(model_type)
            finally:
                sys.argv[:] = _saved_argv
            self._offload = offloadobj
            self._models[model_name] = {
                "model": wan_model,
                "info": {
                    "base_model_type": wgp.get_base_model_type(model_type),
                    "defaults": self._native_defaults(model_type),
                },
            }
            self._native_loaded = True
            vram = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
            logger.info("Wan2GP native: loaded %s (VRAM=%.0fMB)", model_type, vram)
        finally:
            os.chdir(_prev_cwd)

    def _ensure_writable_overlay(self, model_type: str):
        """Create writable /tmp overlay if model dir is on read-only PVC."""
        from shared.utils import files_locator as fl
        import functools, shutil, tempfile

        # Check all possible folder names Wan2GP might use
        base_model_type = model_type.split("_")[0] if "_" in model_type else model_type
        for folder_name in [model_type, base_model_type]:
            model_dir = fl.locate_folder(folder_name, error_if_none=False)
            if not model_dir or not os.path.isdir(model_dir):
                continue
            if os.access(model_dir, os.W_OK):
                continue

            overlay = Path(tempfile.gettempdir()) / "wan2gp_overlay" / folder_name
            if not overlay.exists():
                shutil.copytree(
                    model_dir, overlay, symlinks=True,
                    copy_function=lambda src, dst: Path(dst).symlink_to(src)
                    if Path(src).is_file() else shutil.copy2(src, dst),
                    dirs_exist_ok=True,
                )
            # Ensure configs/ dir is writable (handlers write runtime configs)
            (overlay / "configs").mkdir(parents=True, exist_ok=True)

            # Patch locate_folder to redirect to overlay
            _orig = fl.locate_folder
            @functools.wraps(_orig)
            def _patched(fn, _name=folder_name, _ov=str(overlay), **kw):
                if fn == _name:
                    return _ov
                return _orig(fn, **kw)
            fl.locate_folder = _patched

    @staticmethod
    def _native_defaults(model_type: str) -> dict:
        """Default inference params for native Wan models."""
        return {
            "sampling_steps": 30 if "_1.3B" not in model_type else 20,
            "guide_scale": 5.0,
            "shift": 5.0,
            "frame_num": 81,
            "width": 1280,
            "height": 720,
            "fps": 16,
            "sample_solver": "unipc",
        }

    # Aliases: map model.generate() parameter names to our API payload keys
    _PARAM_ALIASES = {
        "input_prompt": ["prompt", "text"],
        "text": "prompt",
        "sampling_steps": "steps",
        "guide_scale": "guidance_scale",
        "frame_num": "num_frames",
    }

    # Universal defaults for common generate() parameters
    _GENERATE_DEFAULTS = {
        "offloadobj": None,
        "loras_slists": {"phase1": [], "phase2": [], "phase3": []},
        "callback": lambda *a, **kw: None,
        "model_mode": 0,
        "audio_guide": None,
    }

    def _infer_native(self, model, payload: dict, defaults: dict) -> dict:
        """Inference via model.generate() with automatic parameter matching.

        Inspects the model's generate() signature and maps our payload
        to its expected parameters. Works for ALL Wan2GP model families
        without per-model code.
        """
        import inspect
        from mmgp import offload as _moff
        if "_attention" not in _moff.shared_state:
            _moff.shared_state["_attention"] = "sdpa"

        sig = inspect.signature(model.generate)
        param_names = set(sig.parameters)
        kwargs = {}

        # Base64 → file/object conversions for native models.
        # Wan2GP generate() expects file paths or PIL images, not base64.
        if "audio_guide" in param_names and "audio_b64" in payload:
            kwargs["audio_guide"] = self._decode_audio_b64(payload["audio_b64"])
        if "image_start" in param_names and "image_b64" in payload:
            kwargs["image_start"] = self._decode_image_b64(payload["image_b64"])
        if "image_end" in param_names and "image_end_b64" in payload:
            kwargs["image_end"] = self._decode_image_b64(payload["image_end_b64"])

        for name, param in sig.parameters.items():
            if name in kwargs:
                continue
            # 1. Direct match in payload
            if name in payload:
                kwargs[name] = payload[name]
                continue
            # 2. Alias match (single key or list of alternatives)
            alias = self._PARAM_ALIASES.get(name)
            if alias:
                if isinstance(alias, str) and alias in payload:
                    kwargs[name] = payload[alias]
                    continue
                elif isinstance(alias, list):
                    for alt in alias:
                        if alt in payload:
                            kwargs[name] = payload[alt]
                            break
                    if name in kwargs:
                        continue
            # 3. Defaults
            if name in defaults:
                kwargs[name] = defaults[name]
                continue
            # 4. Universal defaults
            if name in self._GENERATE_DEFAULTS:
                kwargs[name] = self._GENERATE_DEFAULTS[name]
                if name == "offloadobj":
                    kwargs[name] = self._offload
                continue
            # 5. Parameter's own default
            if param.default != inspect.Parameter.empty:
                continue

        if not hasattr(model, '_interrupt'):
            model._interrupt = False
        result = model.generate(**kwargs)
        return self._encode_output(result, payload, defaults)

    @staticmethod
    def _decode_audio_b64(audio_b64: str) -> str:
        import tempfile
        raw = base64.b64decode(audio_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name

    @staticmethod
    def _decode_image_b64(image_b64: str):
        from PIL import Image
        import io
        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, payload: dict) -> dict:
        model_key = payload.get("model") or payload.get("model_type") or self._loaded_model or self.default_model

        if model_key != self._loaded_model:
            try:
                self.load(model_key)
            except Exception as e:
                self._loaded_model = None
                return {"status": "error", "error": str(e)}

        # Custom (non-native) models need set_default_device("cuda") because
        # they bypass Wan2GP's mmgp device management. Native models handle
        # device placement internally via the offloadobj — setting this would
        # cause "weights on cpu, input on cuda" mismatches.
        # CPU-only models (kokoro, espeak, faster_whisper) stay on CPU.
        is_cpu_model = self._loaded_model in _CPU_ONLY_TYPES or (
            self._registry.get(self._loaded_model, {}).get("model_type") in _CPU_ONLY_TYPES
        )
        if is_cpu_model:
            torch.set_default_device("cpu")
        elif torch.cuda.is_available() and not self._native_loaded:
            torch.set_default_device("cuda")

        entry = self._registry.get(self._loaded_model)
        if entry is None:
            return {"status": "error", "error": "No model loaded"}

        try:
            from services.wan2gp.custom_models.base_handler import get_handler_meta
            m = self._models.get(self._loaded_model)
            if m is None:
                return {"status": "error", "error": "Model entry not found"}

            model = m["model"]
            info = m.get("info", entry)
            defaults = info.get("defaults", {})

            # Native Wan2GP pipeline: call model.generate() directly.
            # Wan2GP handles device placement (via offloadobj) and dtype
            # internally — do NOT wrap with autocast or set_default_device.
            if self._native_loaded:
                gen = self._infer_native(model, payload, defaults)
                gen["model"] = self._loaded_model
                return gen

            kwargs = _build_generate_kwargs(payload, defaults)

            # Handle image for i2v / image-input models
            base_model_type = info.get("base_model_type", "")

            image_b64 = payload.get("image_b64", "")
            if image_b64 and base_model_type in ("i2v", "i2v_2_2"):
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                kwargs["image_start"] = img
            elif image_b64 and base_model_type in ("see-through",):
                # Decode and resize to handler's default resolution (1280).
                # The handler's _stage_marigold blends layer_images (at
                # resolution) with the input, so sizes must match.
                # Pass as bytes (PNG) since handler expects bytes or base64.
                from PIL import Image
                import io as _io
                _img = Image.open(_io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")
                _default_res = defaults.get("resolution", 1280)
                if _img.size[0] != _default_res or _img.size[1] != _default_res:
                    _img = _img.resize((_default_res, _default_res), Image.LANCZOS)
                _buf = _io.BytesIO()
                _img.save(_buf, format="PNG")
                kwargs["image"] = _buf.getvalue()
            elif image_b64 and base_model_type == "anigen":
                # AniGen expects raw image bytes — it opens with PIL internally
                kwargs["image"] = base64.b64decode(image_b64)

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

            # Wan2GP pipelines call offloadobj.unload_all() during generate()
            # to swap modules between GPU/CPU. Pass the real mmgp offloadobj
            # that was created during _apply_mmgp_profile(). Fall back to
            # dummy only if mmgp isn't managing this model.
            if self._offload is not None:
                kwargs.setdefault("offloadobj", self._offload)
            else:
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

            # Handler-specific kwarg remapping and device patching via
            # HANDLER_META hooks. Falls back to no-op if no meta registered.
            meta = get_handler_meta(base_model_type)

            # Apply device patch before generate()
            if meta and getattr(meta.get("hooks"), "needs_device_patch", False):
                type(model).device = property(lambda self: torch.device("cuda"))

            # Apply before_generate hooks (kwarg remapping, inner device patches)
            if meta and meta.get("hooks"):
                kwargs = meta["hooks"].before_generate(model, kwargs)

            # bf16 autocast wraps model.generate() to handle mixed float32/bfloat16
            # ops (FSQ quantizer, attention layers). Default: ON.
            # Handlers can opt out by setting needs_bf16_autocast = False.
            skip_autocast = meta and getattr(meta.get("hooks"), "needs_bf16_autocast", True) is False
            hooks = meta.get("hooks") if meta else None
            if skip_autocast and hasattr(hooks, "wrap_generate"):
                result = hooks.wrap_generate(model, kwargs, model.generate)
            elif skip_autocast:
                result = model.generate(**kwargs)
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
        logger.info("Wan2GP _load_model() model=%s", model_name)
        _ensure_vendor_path()
        _ensure_quantized_cache()
        _ensure_transformers_compat()
        _patch_quanto_compat()
        _ensure_writable_hf_cache()

        from services.wan2gp.custom_models.base_handler import get_handler_meta

        handler_path = entry["handler_path"]
        model_type = entry["model_type"]
        base_model_type = entry.get("base_model_type", model_type)

        # Run pre_import hooks (source file patching before Python imports)
        meta = get_handler_meta(base_model_type)
        if meta and meta.get("hooks"):
            meta["hooks"].pre_import()

        handler_mod = importlib.import_module(handler_path)
        handler = handler_mod.family_handler

        from registry.config import Config
        from registry.models import ModelRegistry
        cfg = Config()
        model_registry = ModelRegistry()

        model_path = self._resolve_model_path(model_name, entry, model_registry, cfg)

        # Download supporting files (VAE, text encoders, vocoders, etc.) via
        # handler's query_model_files. Also run when model_path exists because
        # some handlers need auxiliary files (e.g. index_tts2 semantic_codec)
        # that aren't in the local weights directory.
        # If model_path is None (weights not on disk), create the target
        # directory so the download has somewhere to go.
        if model_path is None:
            model_path = self._ensure_download_dir(model_type, model_registry, cfg)
        self._ensure_vendor_files(handler, base_model_type, model_def={}, model_path=model_path)
        if model_path is None:
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
        # Also add parent dir and wan2gp dir for shared dependencies like BigVGAN.
        if model_path and model_path.is_dir():
            from shared.utils import files_locator as fl
            from registry.config import Config
            models_root = Path(Config().models_root)

            for search_path in [model_path, model_path.parent,
                                models_root / "wan2gp", models_root]:
                sp = str(search_path)
                if search_path.is_dir() and sp not in fl._checkpoints_paths:
                    fl._checkpoints_paths.append(sp)

            # Handlers expect specific folder names — bind alternate names.
            # Some handlers also write to model_dir (e.g. index_tts2 caches a
            # runtime config). If the PVC is read-only, create a writable overlay.
            _FOLDER_ALIASES = {
                "index_tts2": "index-tts",
            }
            alias = _FOLDER_ALIASES.get(base_model_type)
            if alias and model_path.name == alias:
                _orig_locate = fl.locate_folder
                _alias_name = base_model_type

                # Try creating a writable overlay in /tmp
                import tempfile, shutil
                overlay = Path(tempfile.gettempdir()) / "wan2gp_overlay" / base_model_type
                if not overlay.exists():
                    # Copy structure but symlink large files to avoid duplication
                    shutil.copytree(
                        model_path, overlay,
                        symlinks=True,
                        copy_function=lambda src, dst: Path(dst).symlink_to(src)
                        if Path(src).is_file() else shutil.copy2(src, dst),
                        dirs_exist_ok=True,
                    )
                    # Ensure configs/ dir is writable (handler writes runtime config)
                    configs_dir = overlay / "configs"
                    configs_dir.mkdir(parents=True, exist_ok=True)

                _alias_target = str(overlay)

                # Merge files downloaded by _ensure_vendor_files into the
                # overlay so the handler finds them in its expected model_dir.
                ckpts_dir = Path("ckpts").resolve() / base_model_type
                if ckpts_dir.is_dir():
                    for f in ckpts_dir.iterdir():
                        dst = overlay / f.name
                        if not dst.exists():
                            if f.is_dir():
                                shutil.copytree(f, dst, symlinks=True)
                            else:
                                shutil.copy2(f, dst)

                import functools
                @functools.wraps(_orig_locate)
                def _patched_locate(folder_name, **kw):
                    if folder_name == _alias_name:
                        return _alias_target
                    return _orig_locate(folder_name, **kw)
                fl.locate_folder = _patched_locate

        # Resolve text encoder filename for vendor models that need it
        # (e.g., Wan's T5 encoder, Flux's text encoder)
        text_encoder_path = None
        # Search paths: model dir first, then sibling dirs for shared files
        # (e.g., wan/t2v-1.3B shares T5/VAE with wan/t2v-14B)
        _search_dirs = []
        if model_path and model_path.is_dir():
            _search_dirs.append(model_path)
            if model_path.parent.is_dir():
                _search_dirs.append(model_path.parent)
                for sibling in sorted(model_path.parent.iterdir()):
                    if sibling.is_dir() and sibling != model_path:
                        _search_dirs.append(sibling)
        for search_dir in _search_dirs:
            if text_encoder_path and text_encoder_path.endswith(".safetensors"):
                break
            for f in sorted(search_dir.iterdir()):
                if f.suffix in (".safetensors", ".pth", ".pt"):
                    name_lower = f.name.lower()
                    if ("t5" in name_lower or "umt5" in name_lower
                            or "text_encoder" in name_lower):
                        if f.suffix == ".safetensors":
                            text_encoder_path = str(f)
                            break
                        elif text_encoder_path is None:
                            text_encoder_path = str(f)

        # AniGen's DSINE hub module does `from models import dsine` inside
        # torch.hub.load(). By that point wan2gp's 'models' package is cached
        # in sys.modules. DSINE's models/ is a namespace package (no __init__.py)
        # so Python's regular-package-first resolution skips it in favor of
        # wan2gp's. Fix: create a temp dir with a proper models/ package (with
        # __init__.py), remove /opt/wan2gp from sys.path, and add both the temp
        # dir and DSINE hub dir so Python finds DSINE's models first.
        _torch_hub_original = None
        _anigen_cleanup = None
        if base_model_type == "anigen":
            # AniGen's attention modules default to flash_attn, which only
            # works on CUDA tensors. mmgp offloads modules to CPU during
            # generate(), causing flash_attn to fail. Patch the attention
            # backend to sdpa which works on both CPU and CUDA.
            try:
                import anigen.modules.attention.full_attn as _anigen_full_attn
                _anigen_full_attn.BACKEND = "sdpa"
                # Need sdpa function available for the sdpa branch
                from torch.nn.functional import scaled_dot_product_attention as _sdpa
                if not hasattr(_anigen_full_attn, 'sdpa'):
                    _anigen_full_attn.sdpa = _sdpa
            except (ImportError, AttributeError):
                pass
            import torch.hub as _torch_hub
            _torch_hub_original = _torch_hub._load_local
            _dsine_hub_dir = None
            for hub_dir in (Path(cfg.models_root) / "3d" / "anigen" / "hub").glob("hugoycj_DSINE*"):
                _dsine_hub_dir = str(hub_dir)
                break
            if _dsine_hub_dir:
                import tempfile, shutil
                _tmp_dir = tempfile.mkdtemp()
                for _pkg in ("models", "utils"):
                    _pkg_src = os.path.join(_dsine_hub_dir, _pkg)
                    if not os.path.isdir(_pkg_src):
                        continue
                    _pkg_dst = os.path.join(_tmp_dir, _pkg)
                    os.makedirs(_pkg_dst)
                    with open(os.path.join(_pkg_dst, "__init__.py"), "w") as _f:
                        pass
                    for _fname in os.listdir(_pkg_src):
                        _src = os.path.join(_pkg_src, _fname)
                        if os.path.isfile(_src):
                            shutil.copy2(_src, os.path.join(_pkg_dst, _fname))
                _wan2gp_path_idx = None
                try:
                    _wan2gp_path_idx = sys.path.index("/opt/wan2gp")
                except ValueError:
                    pass
                def _patched_load_local(repo_or_dir, model, *args, **kwargs):
                    _prev_models = sys.modules.pop("models", None)
                    _prev_utils = sys.modules.pop("utils", None)
                    _removed_wan2gp = None
                    if _wan2gp_path_idx is not None:
                        _removed_wan2gp = sys.path.pop(_wan2gp_path_idx)
                    sys.path.insert(0, _dsine_hub_dir)
                    sys.path.insert(0, _tmp_dir)
                    try:
                        return _torch_hub_original(repo_or_dir, model, *args, **kwargs)
                    finally:
                        for _p in (_tmp_dir, _dsine_hub_dir):
                            if _p in sys.path:
                                sys.path.remove(_p)
                        if _removed_wan2gp is not None:
                            sys.path.insert(_wan2gp_path_idx, _removed_wan2gp)
                        if _prev_models is not None:
                            sys.modules["models"] = _prev_models
                        elif "models" in sys.modules:
                            del sys.modules["models"]
                        if _prev_utils is not None:
                            sys.modules["utils"] = _prev_utils
                        elif "utils" in sys.modules:
                            del sys.modules["utils"]
                _torch_hub._load_local = _patched_load_local
                _anigen_cleanup = _tmp_dir

        # See-Through's handler does `from modules.layerdiffuse...` but
        # _ensure_vendor_path() added /opt/wan2gp/models/wan/ to sys.path
        # which shadows the dist-packages modules/ that has layerdiffuse/.
        # Also, layerdiffuse code does `from utils.cv import ...` which needs
        # /opt/seethrough/common on sys.path (the handler's vendor path
        # points to /app/vendor which doesn't exist).
        # Fix: pre-import correct modules, add seethrough/common to sys.path,
        # and clear stale utils from sys.modules.
        _prev_modules = None
        _prev_utils = None
        _seethrough_common = None
        _seethrough_wan_paths = None
        if base_model_type == "see-through":
            _prev_modules = sys.modules.pop("modules", None)
            _prev_utils = sys.modules.pop("utils", None)
            _seethrough_wan_paths = [p for p in sys.path
                                if p.startswith("/opt/wan2gp/models/")]
            for _wp in _seethrough_wan_paths:
                sys.path.remove(_wp)
            # Add seethrough/common so layerdiffuse's utils.cv import works
            _seethrough_common = "/opt/seethrough/common"
            if _seethrough_common not in sys.path:
                sys.path.insert(0, _seethrough_common)
            try:
                _dist_modules = importlib.import_module("modules")
                sys.modules["modules"] = _dist_modules
                # Also pre-import utils so the module is cached correctly
                importlib.import_module("utils")
            except ImportError:
                pass

        try:
            pipeline, pipe_wrapper = handler.load_model(
            model_filename, model_type, base_model_type, model_def,
            quantizeTransformer=not is_cpu,
            text_encoder_quantization="int8" if not is_cpu else None,
            dtype=None if is_cpu else torch.bfloat16,
            VAE_dtype=None if is_cpu else torch.float32,
            profile=0 if is_cpu else MMGP_PROFILES["low_vram"],
            quant=quant,
            text_encoder_filename=text_encoder_path,
        )
        finally:
            if _torch_hub_original is not None:
                import torch.hub as _torch_hub
                _torch_hub._load_local = _torch_hub_original
            if _anigen_cleanup is not None:
                import shutil
                shutil.rmtree(_anigen_cleanup, ignore_errors=True)
            if _prev_modules is not None:
                sys.modules["modules"] = _prev_modules
            if _prev_utils is not None:
                sys.modules["utils"] = _prev_utils
            # Reset default device — set_default_device("cpu") at the top
            # of _load_model must be cleared before mmgp's offload.profile()
            # runs. mmgp uses init_empty_weights (meta tensors) and .to_empty()
            # internally; setting cuda as default device breaks meta tensor
            # handling. Set to "cpu" here; infer() resets to "cuda" after
            # the entire load (including mmgp) completes.
            torch.set_default_device("cpu")
            # Restore wan2gp model paths that were removed for see-through
            if _seethrough_wan_paths:
                for _wp in _seethrough_wan_paths:
                    if _wp not in sys.path:
                        sys.path.insert(0, _wp)

        pipe, co_tenants = self._unwrap_pipe(pipe_wrapper)

        offloadobj = self._apply_mmgp_profile(pipe, co_tenants, is_cpu, model_type)
        if offloadobj is not None:
            self._offload = offloadobj

        # Trellis: move image_cond (DinoV3FeatureExtractor) to CUDA.
        # The handler passes it to _Pipeline separately (not via mmgp pipe),
        # so we need to ensure GPU placement here.
        if base_model_type == "trellis" and hasattr(pipeline, "m"):
            ic = pipeline.m.get("image_cond")
            if ic is not None and torch.cuda.is_available():
                ic.to("cuda")
                logger.info("TRELLIS: moved image_cond (DINOv3) to CUDA")

        # Apply handler on_loaded hooks (attention patching, rembg setup, etc.)
        if meta and meta.get("hooks"):
            meta["hooks"].on_loaded(pipeline, pipe, base_model_type)

        # ACE-Step 1.5: the FSQ quantizer produces Float32 that crashes
        # against BF16 project_out. Autocast in infer() handles this, but
        # we also promote the tokenizer to float32 for consistency (matches
        # Wan2GP's _promote_xl_quantizer_to_fp32_before_mmgp behavior).
        if base_model_type in ("ace_step_v1_5", "ace_step_v1_5_xl"):
            try:
                tokenizer = getattr(getattr(pipeline, "ace_step_transformer", None), "tokenizer", None)
                if tokenizer is not None:
                    tokenizer.float()
                    logger.info("ACE-Step: promoted tokenizer to float32")
            except Exception as e:
                logger.warning("ACE-Step: failed to promote tokenizer: %s", e)

        self._models[model_name] = {
            "model": pipeline,
            "pipe": pipe,
            "info": entry,
            "loaded_at": time.time(),
        }

    def _ensure_download_dir(self, model_type: str, registry, cfg) -> Path | None:
        """Create a writable download directory for a model with no local weights.

        PVC (/models) may be read-only. Use ckpts/ (writable working dir)
        as the download target for vendor models that auto-download.
        """
        ckpts_base = Path("ckpts").resolve()
        ckpts_base.mkdir(parents=True, exist_ok=True)
        return ckpts_base

    def _ensure_vendor_files(self, handler, base_model_type: str,
                              model_def: dict, model_path: Path | None = None) -> None:
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
                        # File exists at model_path — symlink into ckpts so
                        # the handler's files_locator can discover it.
                        if model_path and (model_path / fname).exists():
                            if not local.parent.exists():
                                local.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                local.symlink_to(model_path / fname)
                            except OSError:
                                pass
                            continue
                        try:
                            hf_hub_download(
                                repo_id, str(rel),
                                local_dir=str(ckpts_base),
                                local_dir_use_symlinks=False,
                            )
                            logger.info("Downloaded %s/%s → %s", repo_id, rel, local)
                        except Exception as e:
                            logger.debug("Download failed %s/%s: %s", repo_id, rel, e)

        # Step 2: check defaults URL for the main model file, download to ckpts
        self._ensure_main_model(handler, base_model_type, ckpts_base, model_path)

    def _ensure_main_model(self, handler, base_model_type: str,
                            ckpts_base: Path, model_path: Path | None = None) -> None:
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

            # Check ckpts, model_dir, and the resolved model_path
            if ((ckpts_base / filename).exists()
                    or (model_dir / filename).exists()
                    or (model_path and (model_path / filename).exists())):
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
    _NO_MMGP_MODELS = {"pixal3d", "see-through", "kimodo-soma-rp", "kimodo-soma-seed",
                       "kimodo-g1-rp", "kimodo-smplx-rp"}

    @staticmethod
    def _apply_mmgp_profile(pipe: dict, co_tenants: dict, is_cpu: bool,
                            model_type: str):
        """Apply mmgp VRAM profile. Returns offloadobj or None."""
        from services.wan2gp.custom_models.base_handler import get_handler_meta
        if not pipe or is_cpu:
            return None
        if model_type in Wan2GPService._NO_MMGP_MODELS:
            logger.info("Skipping mmgp for %s (self-managed GPU memory)", model_type)
            return None
        from mmgp import offload

        # Normalize dtypes — mmgp asserts all params in a module share one dtype.
        # Some models have mixed float32 + float16 (e.g. anigen from partial
        # fp16 checkpoint saving). Pre-convert everything to a uniform dtype
        # so mmgp's assertion passes. Use bfloat16 for all models; anigen's
        # spconv ops are handled via float16 autocast at inference time.
        # Also fix meta tensors — some models (ace_step) have parameters not
        # in the checkpoint (e.g. null_condition_emb initialized with randn)
        # that stay as meta tensors after mmgp's init_empty_weights().
        # Exception: index_tts2's handler loads the GPT with
        # default_dtype=float16 and other components in float32. mmgp's
        # internal hooks manage dtype conversion during forward passes.
        # Our normalization would break this mixed-dtype pipeline.
        # Exception: trellis decoders (ss_dec, shape_dec, tex_dec) are fp16
        # checkpoints. Converting fp16→bf16 loses enough precision that the
        # ss_decoder output becomes all-negative (no voxels pass threshold).
        # The handler's own normalization only converts fp32→bf16, preserving
        # fp16 weights correctly.
        target_dtype = torch.bfloat16
        # Check if any module has quantized (quanto) weights. These must not
        # be dtype-normalized: converting INT8 QTensor._data to bfloat16
        # dequantizes the weights and doubles model size (14GB→28GB).
        has_quantized = False
        for v in pipe.values():
            if not isinstance(v, torch.nn.Module):
                continue
            for name, p in v.named_parameters():
                if hasattr(p, '_data') and hasattr(p, '_scale'):
                    has_quantized = True
                    break
                # Quanto stores weights with int8 dtype
                if p.dtype == torch.int8:
                    has_quantized = True
                    break
            if has_quantized:
                break
        skip_dtype_norm = (
            model_type in ("index_tts2", "trellis")
            or model_type.startswith("kimodo")
            or has_quantized
        )
        logger.debug("MMGP skip_dtype_norm=%s model_type=%s", skip_dtype_norm, model_type)
        if skip_dtype_norm:
            for k, v in pipe.items():
                if not isinstance(v, torch.nn.Module):
                    continue
                dtypes = {p.dtype for p in v.parameters()}
                # Set _model_dtype for mmgp even when skipping dtype normalization.
                # mmgp uses this to determine the target dtype for async loading.
                if dtypes:
                    v._model_dtype = min(dtypes, key=lambda d: d.itemsize)
                else:
                    v._model_dtype = target_dtype
                logger.debug("MMGP %s: dtypes=%s _model_dtype=%s", k, dtypes, v._model_dtype)
        if not skip_dtype_norm:
            for k, v in pipe.items():
                if not isinstance(v, torch.nn.Module):
                    continue
                # Always set _model_dtype — some modules may have it set to
                # float32 from the handler, which causes mmgp's assertion.
                v._model_dtype = target_dtype
                for name, p in v.named_parameters():
                    if p.is_meta:
                        new_p = torch.nn.Parameter(
                            torch.randn(p.shape, device="cpu", dtype=target_dtype)
                        )
                        _set_param_by_path(v, name, new_p)
                    elif p.data.dtype in (torch.float32, torch.float16):
                        p.data = p.data.to(target_dtype)

        n_modules = sum(1 for v in pipe.values()
                        if isinstance(v, torch.nn.Module))

        # Models with many modules (trellis: 8, anigen: many) need the most
        # aggressive mmgp profile — profile 5 does no RAM pinning and swaps
        # modules to GPU one at a time. Simpler models (wan, hunyuan) can use
        # profile 4 which pins the transformer for speed.
        # Exception: see_through needs profile 4 because profile 5 causes
        # cascading device mismatches in its deeply nested UNet submodules
        # (GroupEmbedding, timestep_encoder, etc.).
        # Exception: wan/hunyuan video models with quantized weights (~7GB INT8
        # transformer) fit entirely in VRAM on 24GB cards. Profile 5 with 2GB
        # budget causes constant CPU↔GPU swapping and 20+ minute inference.
        # Wan/Hunyuan video models: use Wan2GP's native profile 4 defaults.
        # Profile 4 (LowRAM_LowVRAM) handles quantized 14B models well.
        # Distilled 1.3B models get larger budgets to minimize CPU↔GPU swapping.
        _is_wan_video = (model_type.startswith(("wan-", "wan_", "hunyuan"))
                         or model_type in ("t2v", "t2v_1.3B", "t2v_2_2",
                                           "i2v-14B", "i2v", "vace_1.3B",
                                           "phantom_1.3B", "fun_inp_1.3B",
                                           "recam_1.3B", "sky_df_1.3B"))
        if _is_wan_video:
            profile = MMGP_PROFILES["low_vram"]
            if "_1.3B" in model_type:
                # 1.3B distilled model: ~2.4GB transformer + ~6GB shared deps.
                # Generous budgets keep everything on GPU for fast generation.
                budgets_override = {"transformer": 3000, "text_encoder": 3000,
                                    "*": 3000}
            else:
                # Larger models: Wan2GP's default profile 4 budgets
                budgets_override = {"transformer": 100, "text_encoder": 100,
                                    "*": 3000}
        elif n_modules > 4 and model_type not in ("see-through", "trellis"):
            profile = MMGP_PROFILES["minimum"]
            budgets_override = {"*": 2000}
        elif model_type == "see-through":
            # See-through: 8 modules (~15GB bf16) don't all fit on GPU at once.
            # Handler manages stage-by-stage GPU/CPU placement internally.
            # Leave all modules on CPU; handler's _to_gpu/_to_cpu handles it.
            return None
        elif model_type == "trellis":
            # TRELLIS: ~14.4GB of weights (6 flow models + decoders).
            # Load all modules directly to CUDA.
            for v in pipe.values():
                if isinstance(v, torch.nn.Module):
                    v.to("cuda")
            return None
        elif model_type.startswith(("moss-", "moss_")):
            # MOSS models (Qwen3-based TTS ~16GB, DiT v2 ~7GB) fit entirely in VRAM.
            # mmgp's default transformer:250MB budget would keep the model
            # on CPU and swap tiny chunks — catastrophically slow for
            # autoregressive generation. Keep everything on GPU.
            for v in pipe.values():
                if isinstance(v, torch.nn.Module):
                    v.to("cuda")
            return None
        else:
            profile = MMGP_PROFILES["low_vram"]
            budgets_override = {"transformer": 250, "text_encoder": 250,
                                "*": 3000}

        logger.info("MMGP offload.profile() model_type=%s profile=%s", model_type, profile)
        offloadobj = offload.profile(
            pipe,
            profile_no=profile,
            quantizeTransformer=False,
            budgets=budgets_override,
            loras=[],
            perc_reserved_mem_max=0.5,
            vram_safety_coefficient=0.9,
            coTenantsMap=co_tenants,
        )

        # anigen: spconv doesn't support bfloat16 at all (not even via
        # autocast). Register forward pre/post hooks that temporarily
        # convert spconv params to float32 and cast SparseConvTensor
        # features bf16→float32→bf16 during forward.
        if model_type == "anigen":
            try:
                from spconv.pytorch.conv import SparseConvolution

                def _spconv_pre_hook(module, args):
                    # Cast SparseConvTensor features to float32
                    if args and hasattr(args[0], 'features'):
                        feat = args[0].features
                        if feat.dtype != torch.float32:
                            args = (args[0].replace_feature(feat.float()),) + args[1:]
                    # Cast module params to float32
                    for p in module.parameters():
                        if p.data.dtype != torch.float32:
                            p.data = p.data.float()
                    return args

                def _spconv_post_hook(module, input, output):
                    # Cast output features back to bfloat16
                    if hasattr(output, 'features') and output.features.dtype == torch.float32:
                        output = output.replace_feature(output.features.bfloat16())
                    # Restore module params to bfloat16
                    for p in module.parameters():
                        if p.data.dtype != torch.bfloat16:
                            p.data = p.data.bfloat16()
                    return output

                for mod in pipe.values():
                    if not isinstance(mod, torch.nn.Module):
                        continue
                    for name, submod in mod.named_modules():
                        if isinstance(submod, SparseConvolution):
                            submod.register_forward_pre_hook(_spconv_pre_hook)
                            submod.register_forward_hook(_spconv_post_hook)
            except ImportError:
                pass

        return offloadobj

    def _resolve_model_path(self, model_name: str, entry: dict,
                             registry, cfg) -> Path | None:
        model_type = entry.get("model_type", model_name)

        # Fast path: use weight_path already resolved by discover_models()
        wp = entry.get("weight_path")
        if wp and Path(wp).is_dir():
            return Path(wp)

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

        # Wan family: models stored under wan/ subdirectory with hyphenated names
        model_key_safe = model_type.replace("/", "-").replace(".", "-")
        for wan_name in [model_type.replace("_", "-"), model_key_safe]:
            wan_fallback = Path(cfg.models_root) / "wan2gp" / "wan" / wan_name
            if wan_fallback.is_dir() and any(wan_fallback.glob("*.safetensors")):
                return wan_fallback

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
        "moss_soundeffect_v2": "moss_v2",
        "moss-tts": "moss_tts",
        "moss-ttsd": "moss_ttsd",
        "moss-voicegenerator": "moss_voicegenerator",
        "trellis": "trellis",
        "kokoro": "kokoro",
        "faster_whisper": "faster_whisper",
        "vibevoice-asr": "vibevoice_asr",
        "anigen": "anigen",
        "see-through": "see_through",
        "hy-motion-1.0": "hy_motion",
        "hy-motion-1.0-lite": "hy_motion_lite",
        "pixal3d": "pixal3d",
        "lance-image": "lance_image",
        "lance-video": "lance_video",
        "lance-image-awq": "lance_image",
        "lance-video-awq": "lance_video",
    }

    # model_type → [(output_key, spec_module_key, registry_category, registry_name), ...]
    _PATH_MAP = {
        "moss-soundeffect": [
            ("moss_soundeffect_path", "language_model", "audio", "moss-soundeffect"),
            ("moss_audio_tokenizer_path", "audio_tokenizer", "audio", "moss-audio-tokenizer"),
        ],
        "moss_soundeffect_v2": [
            ("moss_soundeffect_v2_path", "pipeline_root", "audio", "moss-soundeffect-v2"),
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

        # ACE-Step 1.5: use SFT variant (full CFG, 30-50 steps) instead of
        # turbo (8-step distilled, no CFG). Same weights — only config differs.
        if base_model_type == "ace_step_v1_5":
            base["ace_step15_transformer_variant"] = "sft"

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

        # Try .pth/.pt files if no safetensors found (not all handlers support these)
        _PTH_SAFE_TYPES = {"wan", "hunyuan", "flux", "ace_step", "ace_step_v1_5"}
        if base_model_type in _PTH_SAFE_TYPES:
            for ext in ("*.pth", "*.pt"):
                for f in sorted(model_path.glob(ext),
                               key=lambda p: p.stat().st_size, reverse=True):
                    name = f.name
                    if any(p in name for p in exclude_patterns):
                        continue
                    if f.stat().st_size > 100 * 1024 * 1024:
                        return [str(f)]

        return []

    def _encode_output(self, output, payload: dict, defaults: dict) -> dict:
        if isinstance(output, dict):
            frames_tensor = output.get("x")
            audio = output.get("audio")
            sr = output.get("audio_sampling_rate", 24000)
        elif isinstance(output, torch.Tensor):
            frames_tensor = output
            audio = None
            sr = 24000
        else:
            frames_tensor = output
            audio = None
            sr = 24000

        # Detect audio output:
        # 1. Explicit "audio" key
        # 2. Output dict has "audio_sampling_rate" → audio even if key is "x"
        # 3. Tensor with audio-like shape (1D mono, or 2D/3D with channels<=2, samples>>1000)
        is_audio = False
        if audio is not None:
            is_audio = True
        elif isinstance(output, dict) and "audio_sampling_rate" in output:
            if frames_tensor is not None and isinstance(frames_tensor, torch.Tensor):
                audio = frames_tensor
                frames_tensor = None
                is_audio = True
        elif frames_tensor is not None and isinstance(frames_tensor, torch.Tensor):
            if frames_tensor.ndim == 1 and frames_tensor.shape[0] > 1000:
                audio = frames_tensor
                frames_tensor = None
                is_audio = True
            elif frames_tensor.ndim == 3 and frames_tensor.shape[0] <= 2 and frames_tensor.shape[1] <= 2 and frames_tensor.shape[2] > 1000:
                audio = frames_tensor
                frames_tensor = None
                is_audio = True
            elif frames_tensor.ndim == 2 and frames_tensor.shape[0] <= 2 and frames_tensor.shape[1] > 1000:
                audio = frames_tensor
                frames_tensor = None
                is_audio = True

        logger.info("Wan2GP _encode_output: frames_tensor=%s audio=%s is_audio=%s",
                     frames_tensor.shape if isinstance(frames_tensor, torch.Tensor) else type(frames_tensor),
                     audio.shape if isinstance(audio, torch.Tensor) else type(audio) if audio is not None else None,
                     is_audio)

        # Audio-first: if the model returned audio, that's the primary output
        if audio is not None:
            audio_np = audio.cpu().float().numpy() if isinstance(audio, torch.Tensor) else audio
            # Ensure shape is (channels, samples) then transpose to (samples, channels) for soundfile
            if audio_np.ndim == 3:
                audio_np = audio_np.squeeze(0)  # remove batch dim
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]  # (1, samples) → mono
            sr = int(sr if sr != 24000 else
                     payload.get("sample_rate",
                         output.get("audio_sampling_rate",
                         defaults.get("sample_rate", 24000))) if isinstance(output, dict)
                         else defaults.get("sample_rate", 24000))
            import soundfile as sf
            import io as audio_io
            audio_buf = audio_io.BytesIO()
            sf.write(audio_buf, audio_np.T, sr, format="WAV")
            return {
                "status": "ok",
                "data": base64.b64encode(audio_buf.getvalue()).decode(),
                "media_type": "audio/wav",
            }

        if frames_tensor is None:
            raise RuntimeError("Model returned no output")

        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().float().numpy()
        else:
            frames_np = frames_tensor
        msg = f"encode_input: shape={frames_np.shape} dtype={frames_np.dtype} min={frames_np.min():.2f} max={frames_np.max():.2f}"
        logger.info("Wan2GP: %s", msg)

        if frames_np.dtype != np.uint8:
            frames_np = ((frames_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

        # Wan2GP native models always output (C, F, H, W).
        # Transpose to (F, H, W, C) for encoding.
        if frames_np.ndim == 4:
            frames_np = frames_np.transpose(1, 2, 3, 0)
        elif frames_np.ndim == 3 and frames_np.shape[0] in (1, 3, 4):
            frames_np = frames_np.transpose(1, 2, 0)
        logger.info("Wan2GP: final shape=%s dtype=%s", frames_np.shape, frames_np.dtype)

        is_video = frames_np.ndim == 4 and frames_np.shape[0] > 1
        suffix = ".mp4" if is_video else ".png"
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()

        if is_video:
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

        return {
            "status": "ok",
            "data": base64.b64encode(data_bytes).decode(),
            "media_type": "video/mp4" if is_video else "image/png",
        }


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

    # Alias input_prompt → text for handlers that use 'text' parameter
    if "input_prompt" in kwargs and "text" not in kwargs:
        kwargs["text"] = kwargs["input_prompt"]

    if "seed" not in kwargs:
        kwargs["seed"] = -1

    # Wan2GP Gradio defaults — pipelines expect these even when unused
    kwargs.setdefault("model_mode", 0)
    kwargs.setdefault("audio_guide", None)

    for key in _BLOCKED_KEYS:
        if key in payload:
            logger.debug("Blocked key in payload: %s", key)

    return kwargs
