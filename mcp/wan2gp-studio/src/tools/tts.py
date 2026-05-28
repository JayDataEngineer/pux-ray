"""TTS tools — unified speech synthesis across Kokoro, Qwen3-TTS, and MOSS.

- tts_speak: Generate speech from text with voice selection, design, or cloning.
- tts_voices: List available TTS engines and voice presets.
"""
from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context
from loguru import logger

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
        "modes": ["custom_voice"],
        "cpu": True,
    },
    {
        "id": "qwen3_tts",
        "label": "Qwen3-TTS (GPU)",
        "modes": ["custom_voice", "voice_design", "voice_clone"],
        "cpu": False,
    },
    {
        "id": "moss_voicegenerator",
        "label": "MOSS Voice Design (GPU)",
        "modes": ["voice_design"],
        "cpu": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forge(ctx: Context) -> Any:
    fc = ctx.lifespan.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def tts_voices(ctx: Context | None = None) -> dict:
    """List available TTS engines and voice presets."""
    return {
        "engines": ENGINES,
        "voices": {
            "kokoro": KOKORO_VOICES,
            "qwen3_tts": QWEN3_VOICES,
        },
    }


async def tts_speak(
    text: str,
    engine: str = "kokoro",
    mode: str = "custom_voice",
    voice: str | None = None,
    instruct: str | None = None,
    ref_audio_b64: str | None = None,
    language: str = "English",
    ctx: Context | None = None,
) -> dict:
    """Generate speech from text.

    Modes:
      - custom_voice: Use a preset speaker voice.
      - voice_design: Generate a voice from a text description (no audio input).
      - voice_clone: Clone a voice from a reference audio clip (base64).
    """
    forge = _forge(ctx)

    if not text:
        return {"status": "error", "error": "text is required"}

    if engine == "kokoro":
        return await forge.invoke({
            "service": "kokoro",
            "text": text,
            "voice": voice or "af_bella",
        })

    if engine == "qwen3_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "faster_qwen3_tts",
            "text": text,
            "language": language,
        }

        if ref_audio_b64:
            payload["ref_audio_b64"] = ref_audio_b64
            if not mode:
                payload["mode"] = "voice_clone"
        elif mode == "voice_design":
            payload["mode"] = "voice_design"
            payload["instruct"] = instruct or ""
        else:
            payload["mode"] = "custom_voice"
            payload["voice"] = voice or "Aiden"

        return await forge.invoke(payload)

    if engine == "moss_voicegenerator":
        payload = {
            "service": "wan2gp",
            "model": "moss-voicegenerator",
            "text": instruct or text,
        }
        return await forge.invoke(payload)

    return {"status": "error", "error": f"Unknown engine: {engine}"}
