"""Centralized service registry — single source of truth for all deployments.

Replaces duplicate dicts in gateway/ingress.py, gateway/dashboard.py,
and scattered GPU scheduling logic. Every service entry maps a public
name to its Ray Serve deployment metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServiceEntry:
    deployment: str
    """Ray Serve deployment name (e.g. "kokoro_tts")."""

    app: str
    """Ray Serve application name (usually same as deployment)."""

    label: str
    """Human-readable name for dashboard display (e.g. "Kokoro TTS")."""

    category: str
    """Service category: tts, asr, audio, creative, vision, multimodal, image, llm."""

    needs_gpu: bool
    """Whether to acquire the GPU scheduler before calling this service."""

    default_model: str
    """Default model name passed to _load()."""

    output_type: str
    """Primary output format: audio, json, model_3d, image, proxy."""

    model_aliases: dict[str, str] = field(default_factory=dict)
    """Map external model names to this service's default model.
    E.g. {"tts-01-kokoro": "kokoro"} for OpenAI-compatible TTS routing.
    """

    description: str = ""
    """One-line description for API docs and dashboard."""


SERVICE_REGISTRY: dict[str, ServiceEntry] = {
    # ── TTS ────────────────────────────────────────────────────────────────────
    "kokoro": ServiceEntry(
        deployment="kokoro_tts", app="kokoro_tts",
        label="Kokoro TTS", category="tts",
        needs_gpu=False, default_model="kokoro",
        output_type="audio",
        model_aliases={"tts-01-kokoro": "kokoro"},
        description="Kokoro 82M — fast CPU text-to-speech, multi-voice.",
    ),
    "espeak": ServiceEntry(
        deployment="espeak_tts", app="espeak_tts",
        label="eSpeak TTS", category="tts",
        needs_gpu=False, default_model="espeak-ng",
        output_type="audio",
        model_aliases={"tts-01-espeak": "espeak-ng"},
        description="eSpeak-NG — lightweight phoneme TTS, many languages.",
    ),
    "faster_qwen3_tts": ServiceEntry(
        deployment="faster_qwen3_tts", app="faster_qwen3_tts",
        label="Faster-Qwen3-TTS", category="tts",
        needs_gpu=True, default_model="qwen3-tts",
        output_type="audio",
        description="Qwen3-TTS 1.7B — fast GPU-accelerated TTS.",
    ),
    "index_tts": ServiceEntry(
        deployment="index_tts", app="index_tts",
        label="IndexTTS", category="tts",
        needs_gpu=True, default_model="index-tts",
        output_type="audio",
        model_aliases={"tts-01-index": "index-tts"},
        description="IndexTTS v2 — high-quality neural TTS (12GB model).",
    ),
    "qwen_tts": ServiceEntry(
        deployment="qwen_tts", app="qwen_tts",
        label="Qwen3-TTS", category="tts",
        needs_gpu=True, default_model="qwen3-tts",
        output_type="audio",
        model_aliases={"tts-01-qwen": "qwen3-tts"},
        description="Qwen3-TTS 1.7B — 9 premium voices with instruction control.",
    ),
    "vibevoice_cpp_gpu": ServiceEntry(
        deployment="vibevoice_cpp_gpu", app="vibevoice_cpp_gpu",
        label="VibeVoice CPP GPU", category="tts",
        needs_gpu=False, default_model="vibevoice-cpp",
        output_type="audio",
        model_aliases={"tts-01-vibevoice": "vibevoice-cpp"},
        description="VibeVoice 7B — multi-speaker synthesis with voice cloning.",
    ),
    "gpt_sovits": ServiceEntry(
        deployment="gpt_sovits", app="gpt_sovits",
        label="GPT-SoVITS", category="tts",
        needs_gpu=True, default_model="gpt-sovits",
        output_type="audio",
        model_aliases={"tts-01-gpt-sovits": "gpt-sovits"},
        description="GPT-SoVITS — voice cloning TTS from reference audio.",
    ),
    # ── ASR ────────────────────────────────────────────────────────────────────
    "faster_whisper": ServiceEntry(
        deployment="faster_whisper", app="faster_whisper",
        label="Faster-Whisper", category="asr",
        needs_gpu=False, default_model="faster-whisper",
        output_type="json",
        model_aliases={"whisper-1": "faster-whisper"},
        description="Faster-Whisper distil-large-v3 — fast CPU ASR.",
    ),
    "vibevoice_microsoft": ServiceEntry(
        deployment="vibevoice_microsoft", app="vibevoice_microsoft",
        label="VibeVoice Microsoft ASR", category="asr",
        needs_gpu=True, default_model="vibevoice-asr",
        output_type="json",
        model_aliases={"vibevoice-microsoft": "vibevoice-asr"},
        description="VibeVoice Microsoft — microsoft/VibeVoice-ASR 7B speech recognition.",
    ),
    "qwen_asr": ServiceEntry(
        deployment="qwen_asr", app="qwen_asr",
        label="Qwen ASR", category="asr",
        needs_gpu=True, default_model="qwen-asr",
        output_type="json",
        model_aliases={"qwen-asr": "qwen-asr"},
        description="Qwen ASR — GPU-accelerated speech recognition.",
    ),
    # ── Audio Generation ───────────────────────────────────────────────────────
    "moss_soundeffect": ServiceEntry(
        deployment="moss_soundeffect", app="moss_soundeffect",
        label="MOSS-SoundEffect", category="audio",
        needs_gpu=True, default_model="moss-soundeffect",
        output_type="audio",
        description="MOSS-SoundEffect 8B — text-to-sound effects.",
    ),
    "tangoflux": ServiceEntry(
        deployment="tangoflux", app="tangoflux",
        label="TangoFlux", category="audio",
        needs_gpu=True, default_model="tangoflux",
        output_type="audio",
        description="TangoFlux — flow-matching text-to-audio generation.",
    ),
    "ace_step": ServiceEntry(
        deployment="ace_step", app="ace_step",
        label="ACE-Step", category="audio",
        needs_gpu=True, default_model="ace-step",
        output_type="audio",
        description="ACE-STEP 1.5 — text-to-music generation.",
    ),
    # ── Creative / 3D ──────────────────────────────────────────────────────────
    "trellis": ServiceEntry(
        deployment="trellis", app="trellis",
        label="TRELLIS.2", category="creative",
        needs_gpu=True, default_model="trellis",
        output_type="model_3d",
        description="TRELLIS.2 4B — image-to-3D mesh generation.",
    ),
    "anigen": ServiceEntry(
        deployment="anigen", app="anigen",
        label="AniGen", category="creative",
        needs_gpu=True, default_model="anigen",
        output_type="model_3d",
        description="AniGen — anime image-to-3D with textures.",
    ),
    "hy_motion": ServiceEntry(
        deployment="hy_motion", app="hy_motion",
        label="HY-Motion", category="creative",
        needs_gpu=True, default_model="hy-motion-1.0",
        output_type="json",
        description="HY-Motion 1.0 — text-to-3D human motion.",
    ),
    "see_through": ServiceEntry(
        deployment="see_through", app="see_through",
        label="See-Through", category="creative",
        needs_gpu=True, default_model="see-through",
        output_type="image",
        description="See-Through — anime layer decomposition into PSD.",
    ),
    # ── Vision / Multimodal ────────────────────────────────────────────────────

    "phi4mm": ServiceEntry(
        deployment="phi4mm", app="phi4mm",
        label="Phi-4-MM", category="multimodal",
        needs_gpu=True, default_model="phi4-multimodal",
        output_type="json",
        description="Phi-4-multimodal — text+vision+speech input, text output.",
    ),
    # ── Image ──────────────────────────────────────────────────────────────────
    "comfyui": ServiceEntry(
        deployment="comfyui", app="comfyui",
        label="ComfyUI", category="image",
        needs_gpu=True, default_model="comfyui",
        output_type="proxy",
        description="ComfyUI — node-based image/video generation pipeline.",
    ),
    # ── LLM ────────────────────────────────────────────────────────────────────
    "llm": ServiceEntry(
        deployment="llm", app="llm",
        label="LLM (llama.cpp)", category="llm",
        needs_gpu=True, default_model="qwen3.6-27b-q4_k_xl",
        output_type="json",
        description="llama.cpp LLM server — OpenAI-compatible chat completions.",
    ),
    # ── Wan2GP Pool ──────────────────────────────────────────────────────────────
    "wan2gp": ServiceEntry(
        deployment="forge", app="forge",
        label="Wan2GP Pool", category="creative",
        needs_gpu=True, default_model="wan/t2v-14B",
        output_type="video",
        model_aliases={"wan-t2v": "wan/t2v-14B", "wan-i2v": "wan/i2v-14B"},
        description="Wan2GP — 90+ model variants via mmgp pool (video, image, audio).",
    ),
}


def get_service(name: str) -> Optional[ServiceEntry]:
    """Look up a service by name. Returns None if not found."""
    return SERVICE_REGISTRY.get(name)


def resolve_model(model_name: str) -> Optional[tuple[str, ServiceEntry]]:
    """Resolve an external model name to (service_key, ServiceEntry).

    Searches model_aliases across all services. Returns None if not found.
    """
    for key, entry in SERVICE_REGISTRY.items():
        if model_name in entry.model_aliases:
            return key, entry
    return None


def get_all_services() -> dict[str, ServiceEntry]:
    """Return the full registry (for dashboard / API listing)."""
    return dict(SERVICE_REGISTRY)
