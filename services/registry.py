"""Centralized service registry — single source of truth for all deployments.

Every model service routes through the Forge → Wan2GP → family_handlers pipeline.
The registry maps public service names to metadata for API docs, dashboard, and routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServiceEntry:
    deployment: str
    """Ray Serve deployment name."""

    app: str
    """Ray Serve application name."""

    label: str
    """Human-readable name for dashboard display."""

    category: str
    """Service category: tts, asr, audio, creative, llm."""

    needs_gpu: bool
    """Whether this service uses GPU."""

    default_model: str
    """Default model name."""

    output_type: str
    """Primary output format: audio, json, model_3d, image, proxy, video."""

    model_aliases: dict[str, str] = field(default_factory=dict)
    description: str = ""


SERVICE_REGISTRY: dict[str, ServiceEntry] = {
    # ── Wan2GP — standalone GPU deployment (mmgp-managed, ALL models) ──────────
    "wan2gp": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Wan2GP Pool", category="creative",
        needs_gpu=True, default_model="wan/t2v-14B",
        output_type="video",
        model_aliases={"wan-t2v": "wan/t2v-14B", "wan-i2v": "wan/i2v-14B"},
        description="Wan2GP — unified model pool with mmgp VRAM management.",
    ),

    # ── Forge Services (subprocess-based) ──────────────────────────────────────
    "comfyui": ServiceEntry(
        deployment="forge", app="forge",
        label="ComfyUI", category="image",
        needs_gpu=True, default_model="comfyui",
        output_type="proxy",
        description="ComfyUI — node-based image/video generation pipeline.",
    ),
    "llm": ServiceEntry(
        deployment="forge", app="forge",
        label="LLM (llama.cpp)", category="llm",
        needs_gpu=True, default_model="qwen3.6-27b-q5_k_s-32k",
        output_type="json",
        description="llama.cpp LLM server — OpenAI-compatible chat completions.",
    ),

    # ── Model Engine services (route through Wan2GP deployment directly) ────────
    "kokoro": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Kokoro TTS", category="tts",
        needs_gpu=False, default_model="kokoro",
        output_type="audio",
        model_aliases={"tts-01-kokoro": "kokoro"},
        description="Kokoro 82M — fast CPU text-to-speech, multi-voice.",
    ),
    "espeak": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="eSpeak TTS", category="tts",
        needs_gpu=False, default_model="espeak",
        output_type="audio",
        model_aliases={"tts-01-espeak": "espeak"},
        description="eSpeak-NG — lightweight phoneme TTS, many languages.",
    ),
    "faster_whisper": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Faster-Whisper", category="asr",
        needs_gpu=False, default_model="faster_whisper",
        output_type="json",
        model_aliases={"whisper-1": "faster_whisper"},
        description="Faster-Whisper distil-large-v3 — fast CPU ASR.",
    ),
    "index_tts": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="IndexTTS", category="tts",
        needs_gpu=True, default_model="index_tts/v2",
        output_type="audio",
        model_aliases={"tts-01-index": "index_tts/v2"},
        description="IndexTTS v2 — high-quality neural TTS (blocked: transformers compat).",
    ),
    "moss_soundeffect": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="MOSS-SoundEffect", category="audio",
        needs_gpu=True, default_model="moss/moss-soundeffect",
        output_type="audio",
        description="MOSS-SoundEffect 8B — text-to-sound effects.",
    ),
    "ace_step": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="ACE-Step", category="audio",
        needs_gpu=True, default_model="ace_step/v1_5",
        output_type="audio",
        description="ACE-STEP 1.5 — text-to-music generation.",
    ),
    "vibevoice_asr": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="VibeVoice ASR", category="asr",
        needs_gpu=True, default_model="vibevoice_asr",
        output_type="json",
        description="VibeVoice ASR — speech recognition with diarization.",
    ),

    # ── Additional Wan2GP services (previously missing from registry) ──────────
    "trellis": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="TRELLIS 3D", category="3d",
        needs_gpu=True, default_model="trellis",
        output_type="model_3d",
        description="TRELLIS.2 4B — image-to-3D mesh generation.",
    ),
    "anigen": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="AniGen 3D", category="3d",
        needs_gpu=True, default_model="anigen",
        output_type="model_3d",
        description="AniGen — anime image-to-rigged-3D generation.",
    ),
    "z_image": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Wan2GP Image", category="image",
        needs_gpu=True, default_model="z_image",
        output_type="image",
        description="Wan2GP image generation (Flux, SD variants).",
    ),
    "faster_qwen3_tts": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Faster Qwen3-TTS", category="tts",
        needs_gpu=True, default_model="faster_qwen3_tts",
        output_type="audio",
        description="Qwen3-TTS with CUDA graph acceleration — 5x faster than baseline.",
    ),
    "hy_motion": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="HY-Motion", category="motion",
        needs_gpu=True, default_model="hy_motion",
        output_type="motion",
        description="HY-Motion 1.0 — text-to-3D motion generation.",
    ),
    "see_through": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="See-Through", category="creative",
        needs_gpu=True, default_model="see_through",
        output_type="image",
        description="See-Through — anime layer decomposition.",
    ),

    # ── Kimodo Motion (Forge-managed, standalone) ────────────────────────────
    "kimodo": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Kimodo Motion", category="motion",
        needs_gpu=True, default_model="kimodo-soma-rp",
        output_type="motion",
        description="Kimodo text-to-3D motion (NPZ output, SOMA 77-joint skeleton).",
    ),
    "lance": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="Lance Multimodal", category="creative",
        needs_gpu=True, default_model="lance/lance-video",
        output_type="video",
        model_aliases={"lance-t2v": "lance/lance-video", "lance-t2i": "lance/lance-image"},
        description="Lance 3B — ByteDance unified multimodal: t2i, t2v, image/video edit, understanding.",
    ),

    "kimodo_demo": ServiceEntry(
        deployment="forge", app="forge",
        label="Kimodo Demo", category="motion",
        needs_gpu=True, default_model="kimodo-soma-rp",
        output_type="proxy",
        description="Kimodo Viser demo — interactive 3D motion authoring.",
    ),

    # ── Avatar Pipeline (Forge-managed, staged VRAM) ──────────────────────────
    "avatar": ServiceEntry(
        deployment="forge", app="forge",
        label="Avatar Pipeline", category="avatar",
        needs_gpu=True, default_model="gem_smpl",
        output_type="video",
        description="Text-to-avatar: GEM gesture gen + SOMA body model + FluxRT rendering.",
    ),
}


def get_service(name: str) -> Optional[ServiceEntry]:
    return SERVICE_REGISTRY.get(name)


def resolve_model(model_name: str) -> Optional[tuple[str, ServiceEntry]]:
    for key, entry in SERVICE_REGISTRY.items():
        if model_name in entry.model_aliases:
            return key, entry
    return None


def get_all_services() -> dict[str, ServiceEntry]:
    return dict(SERVICE_REGISTRY)


def list_all_models(category: str | None = None) -> list[dict]:
    """Return all service entries as model dicts for API discovery."""
    models = []
    for key, entry in SERVICE_REGISTRY.items():
        if category and entry.category != category:
            continue
        models.append({
            "id": key,
            "label": entry.label,
            "category": entry.category,
            "needs_gpu": entry.needs_gpu,
            "output_type": entry.output_type,
            "default_model": entry.default_model,
            "description": entry.description,
            "model_aliases": list(entry.model_aliases.keys()),
        })
    return models
