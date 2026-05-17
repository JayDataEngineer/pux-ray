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
        needs_gpu=True, default_model="qwen3.6-27b-q4_k_xl",
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
    "vibevoice_tts": ServiceEntry(
        deployment="wan2gp", app="wan2gp",
        label="VibeVoice TTS", category="tts",
        needs_gpu=True, default_model="vibevoice_tts",
        output_type="audio",
        description="VibeVoice TTS — multi-speaker text-to-speech.",
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
