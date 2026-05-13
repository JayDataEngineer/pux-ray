"""Centralized service registry — single source of truth for all deployments.

Every model service routes through the Forge → Wan2GP → model_engine pipeline.
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
    # ── Forge Services ──────────────────────────────────────────────────────────
    "wan2gp": ServiceEntry(
        deployment="forge", app="forge",
        label="Wan2GP Pool", category="creative",
        needs_gpu=True, default_model="wan/t2v-14B",
        output_type="video",
        model_aliases={"wan-t2v": "wan/t2v-14B", "wan-i2v": "wan/i2v-14B"},
        description="Wan2GP — unified model pool with mmgp VRAM management.",
    ),
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

    # ── Model Engine services (route through Forge → Wan2GP) ────────────────────
    "kokoro": ServiceEntry(
        deployment="forge", app="forge",
        label="Kokoro TTS", category="tts",
        needs_gpu=False, default_model="kokoro",
        output_type="audio",
        model_aliases={"tts-01-kokoro": "kokoro"},
        description="Kokoro 82M — fast CPU text-to-speech, multi-voice.",
    ),
    "espeak": ServiceEntry(
        deployment="forge", app="forge",
        label="eSpeak TTS", category="tts",
        needs_gpu=False, default_model="espeak-ng",
        output_type="audio",
        model_aliases={"tts-01-espeak": "espeak-ng"},
        description="eSpeak-NG — lightweight phoneme TTS, many languages.",
    ),
    "faster_qwen3_tts": ServiceEntry(
        deployment="forge", app="forge",
        label="Faster-Qwen3-TTS", category="tts",
        needs_gpu=True, default_model="qwen3-tts",
        output_type="audio",
        description="Qwen3-TTS 1.7B — fast GPU-accelerated TTS.",
    ),
    "faster_whisper": ServiceEntry(
        deployment="forge", app="forge",
        label="Faster-Whisper", category="asr",
        needs_gpu=False, default_model="faster-whisper",
        output_type="json",
        model_aliases={"whisper-1": "faster-whisper"},
        description="Faster-Whisper distil-large-v3 — fast CPU ASR.",
    ),
    "index_tts": ServiceEntry(
        deployment="forge", app="forge",
        label="IndexTTS", category="tts",
        needs_gpu=True, default_model="index-tts",
        output_type="audio",
        model_aliases={"tts-01-index": "index-tts"},
        description="IndexTTS v2 — high-quality neural TTS (blocked: transformers compat).",
    ),
    "moss_soundeffect": ServiceEntry(
        deployment="forge", app="forge",
        label="MOSS-SoundEffect", category="audio",
        needs_gpu=True, default_model="moss-soundeffect",
        output_type="audio",
        description="MOSS-SoundEffect 8B — text-to-sound effects.",
    ),
    "ace_step": ServiceEntry(
        deployment="forge", app="forge",
        label="ACE-Step", category="audio",
        needs_gpu=True, default_model="ace-step",
        output_type="audio",
        description="ACE-STEP 1.5 — text-to-music generation.",
    ),
    "trellis": ServiceEntry(
        deployment="forge", app="forge",
        label="TRELLIS.2", category="creative",
        needs_gpu=True, default_model="trellis",
        output_type="model_3d",
        description="TRELLIS.2 4B — image-to-3D mesh generation.",
    ),
    "anigen": ServiceEntry(
        deployment="forge", app="forge",
        label="AniGen", category="creative",
        needs_gpu=True, default_model="anigen",
        output_type="model_3d",
        description="AniGen — anime image-to-3D with textures.",
    ),
    "hy_motion": ServiceEntry(
        deployment="forge", app="forge",
        label="HY-Motion", category="creative",
        needs_gpu=True, default_model="hy-motion-1.0",
        output_type="json",
        description="HY-Motion 1.0 — text-to-3D human motion.",
    ),
    "see_through": ServiceEntry(
        deployment="forge", app="forge",
        label="See-Through", category="creative",
        needs_gpu=True, default_model="see-through",
        output_type="image",
        description="See-Through — anime layer decomposition.",
    ),
    "vibevoice_asr": ServiceEntry(
        deployment="forge", app="forge",
        label="VibeVoice ASR", category="asr",
        needs_gpu=True, default_model="vibevoice-asr",
        output_type="json",
        description="VibeVoice ASR — speech recognition with diarization.",
    ),
    "vibevoice_tts": ServiceEntry(
        deployment="forge", app="forge",
        label="VibeVoice TTS", category="tts",
        needs_gpu=True, default_model="vibevoice-tts",
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
