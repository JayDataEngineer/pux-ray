"""TTS tools — unified speech synthesis across all engines.

- tts_speak: One endpoint for Kokoro, Qwen3-TTS, MOSS VoiceGenerator, eSpeak, IndexTTS.
- tts_voices: List available TTS engines with per-engine parameter schemas.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field

# ---------------------------------------------------------------------------
# Voice catalogs
# ---------------------------------------------------------------------------

KOKORO_VOICES = sorted([
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa", "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
])

QWEN3_VOICES = [
    "Aiden", "Chloe", "Ethan", "Marcus", "Ono_Anna", "Sohee",
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
]

ENGINES = [
    {
        "id": "kokoro",
        "label": "Kokoro (CPU)",
        "gpu": False,
        "description": "Fast CPU text-to-speech, multi-voice. Best for quick generation.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
            {"name": "voice", "type": "select", "label": "Voice", "default": "af_bella",
             "options": KOKORO_VOICES},
        ],
    },
    {
        "id": "qwen3_tts",
        "label": "Qwen3-TTS (GPU)",
        "gpu": True,
        "description": "Qwen3-TTS with CUDA graph acceleration. Supports voice design and cloning.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
            {"name": "mode", "type": "select", "label": "Mode", "default": "custom_voice",
             "options": ["custom_voice", "voice_design", "voice_clone"],
             "description": "custom_voice: preset speaker / voice_design: describe a voice / voice_clone: from reference audio"},
            {"name": "voice", "type": "select", "label": "Voice", "default": "Aiden",
             "options": QWEN3_VOICES},
            {"name": "instruct", "type": "textarea", "label": "Voice Instruction",
             "placeholder": "A warm female voice with a gentle southern accent..."},
            {"name": "language", "type": "select", "label": "Language", "default": "English",
             "options": ["English", "Chinese", "Japanese", "Korean"]},
        ],
    },
    {
        "id": "moss_tts",
        "label": "MOSS TTS (GPU)",
        "gpu": True,
        "description": "MOSS TTS — text-to-speech with voice cloning via reference audio.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
            {"name": "instruct", "type": "textarea", "label": "Instruction",
             "placeholder": "warm, friendly, slightly husky",
             "description": "Optional emotion/style instruction for the voice."},
            {"name": "language", "type": "select", "label": "Language", "default": "English",
             "options": ["English", "Chinese", "Japanese", "Korean"]},
        ],
    },
    {
        "id": "espeak",
        "label": "eSpeak (CPU)",
        "gpu": False,
        "description": "eSpeak-NG — lightweight phoneme TTS, many languages. Instant CPU inference.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
            {"name": "language", "type": "select", "label": "Language", "default": "en",
             "options": ["en", "fr", "de", "es", "it", "ja", "zh", "ko", "ru", "pt"]},
        ],
    },
    {
        "id": "index_tts",
        "label": "IndexTTS (GPU)",
        "gpu": True,
        "description": "IndexTTS v2 — high-quality neural TTS with voice cloning.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forge(ctx: Context) -> Any:
    fc = ctx.lifespan_context.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def tts_voices(ctx: Context | None = None) -> dict:
    """List available TTS engines and per-engine parameter schemas."""
    return {
        "engines": ENGINES,
        "voices": {
            "kokoro": KOKORO_VOICES,
            "qwen3_tts": QWEN3_VOICES,
        },
    }


async def tts_speak(
    text: Annotated[str, Field(
        description="The text to synthesize into speech.",
    )],
    engine: Annotated[str, Field(
        description="TTS engine to use.",
        enum=["kokoro", "qwen3_tts", "moss_tts", "espeak", "index_tts"],
    )] = "kokoro",
    mode: Annotated[str, Field(
        description="Qwen3-TTS mode: custom_voice (preset), voice_design (describe), voice_clone (from audio).",
        enum=["custom_voice", "voice_design", "voice_clone"],
    )] = "custom_voice",
    voice: Annotated[str | None, Field(
        description="Voice preset name. Kokoro: af_bella, af_nova, am_adam, etc. Qwen3: Aiden, Chloe, etc.",
    )] = None,
    instruct: Annotated[str | None, Field(
        description="Voice design instruction (qwen3_tts voice_design mode, moss_tts emotion/style).",
    )] = None,
    ref_audio_b64: Annotated[str | None, Field(
        description="Base64-encoded reference audio for voice cloning.",
    )] = None,
    language: Annotated[str, Field(
        description="Language for synthesis.",
        enum=["English", "Chinese", "Japanese", "Korean", "en", "fr", "de", "es", "it", "ja", "zh", "ko", "ru", "pt"],
    )] = "English",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Generate speech from text using any TTS engine.

    Engines:
      - kokoro: Fast CPU TTS with 50+ voice presets
      - qwen3_tts: GPU TTS with custom_voice/voice_design/voice_clone modes
      - moss_tts: GPU TTS with voice cloning from reference audio
      - espeak: Ultra-lightweight CPU TTS with 10 languages
      - index_tts: High-quality GPU TTS for voice cloning

    Args:
        text: The text to synthesize into speech
        engine: TTS engine to use (kokoro, qwen3_tts, moss_tts, espeak, index_tts)
        mode: Qwen3-TTS mode — custom_voice (preset speaker), voice_design (describe a voice), voice_clone (clone from audio)
        voice: Voice preset name (kokoro/qwen3_tts custom_voice mode)
        instruct: Voice design instruction text (qwen3_tts voice_design mode, moss_tts emotion/style)
        ref_audio_b64: Base64-encoded reference audio for voice cloning (qwen3_tts voice_clone mode)
        language: Language for synthesis (English, Chinese, Japanese, Korean / en, fr, de, etc. for espeak)
    """
    forge = _forge(ctx)

    if not text:
        return {"status": "error", "error": "text is required"}

    # ── Kokoro: direct service dispatch (CPU, multi-voice) ────────────────
    if engine == "kokoro":
        return await forge.invoke({
            "service": "kokoro",
            "text": text,
            "voice": voice or "af_bella",
        })

    # ── eSpeak: direct service dispatch (CPU, multi-language) ─────────────
    if engine == "espeak":
        return await forge.invoke({
            "service": "espeak",
            "text": text,
            "language": language or "en",
        })

    # ── Qwen3-TTS: GPU with mode-dependent routing ───────────────────────
    if engine == "qwen3_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "faster_qwen3_tts",
            "text": text,
            "language": language,
        }

        if ref_audio_b64:
            payload["ref_audio_b64"] = ref_audio_b64
            payload["mode"] = "voice_clone"
        elif mode == "voice_design":
            payload["mode"] = "voice_design"
            payload["instruct"] = instruct or ""
        else:
            payload["mode"] = "custom_voice"
            payload["voice"] = voice or "Aiden"

        return await forge.invoke(payload)

    # ── MOSS TTS: GPU voice cloning ──────────────────────────────────────
    if engine == "moss_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "moss-tts",
            "text": text,
            "language": language,
        }
        if instruct:
            payload["instruction"] = instruct
        if ref_audio_b64:
            payload["ref_audio_b64"] = ref_audio_b64
        return await forge.invoke(payload)

    # ── IndexTTS: GPU voice cloning ──────────────────────────────────────
    if engine == "index_tts":
        return await forge.invoke({
            "service": "wan2gp",
            "model": "index_tts/v2",
            "text": text,
        })

    return {"status": "error", "error": f"Unknown engine: {engine}. "
            f"Available: kokoro, qwen3_tts, moss_tts, espeak, index_tts"}
