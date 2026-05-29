"""Audio tools — speech recognition, sound effects, and music generation.

- transcribe: Speech-to-text (faster_whisper CPU, vibevoice_asr GPU)
- generate_sound: Text-to-sound-effect (MOSS-SoundEffect 8B GPU)
- generate_music: Text-to-music (ACE-Step 1.5 GPU)
"""
from __future__ import annotations

import base64
from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field


def _forge(ctx: Context) -> Any:
    fc = ctx.lifespan_context.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


async def transcribe(
    audio_b64: Annotated[str, Field(
        description="Base64-encoded audio file (wav, mp3, flac, ogg). "
                    "Required — provide the audio data to transcribe.",
    )],
    engine: Annotated[str, Field(
        description="ASR engine: 'whisper' (CPU, fast) or 'vibevoice' (GPU, diarization).",
    )] = "whisper",
    language: Annotated[str | None, Field(
        description="Language hint (e.g. 'en', 'ja', 'zh'). Auto-detects if omitted.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Transcribe speech from audio to text.

    Returns {status, text, language, segments?, speakers?}.
    Whisper is CPU-only and fast. VibeVoice adds speaker diarization on GPU.
    """
    forge = _forge(ctx)

    if engine == "vibevoice":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "vibevoice_asr",
            "audio_b64": audio_b64,
        }
        if language:
            payload["language"] = language
    else:
        payload = {
            "service": "wan2gp",
            "model": "faster_whisper",
            "audio_b64": audio_b64,
        }
        if language:
            payload["language"] = language

    return await forge.invoke(payload)


async def generate_sound(
    prompt: Annotated[str, Field(
        description="Description of the sound effect to generate. "
                    "E.g. 'thunder rumbling in the distance', 'sword unsheathing', "
                    "'crowd cheering in a stadium'.",
    )],
    duration_seconds: Annotated[float, Field(
        description="Target duration in seconds (1-30). Model may adjust slightly.",
    )] = 5.0,
    seed: Annotated[int | None, Field(
        description="Random seed for reproducibility.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate a sound effect from a text description.

    Uses MOSS-SoundEffect 8B on GPU. Returns {status, data (base64 wav), media_type}.
    """
    forge = _forge(ctx)

    payload: dict[str, Any] = {
        "service": "wan2gp",
        "model": "moss/moss-soundeffect",
        "prompt": prompt,
        "duration": duration_seconds,
    }
    if seed is not None:
        payload["seed"] = seed

    return await forge.invoke(payload)


async def generate_music(
    prompt: Annotated[str, Field(
        description="Description of the music to generate. "
                    "E.g. 'upbeat electronic dance music with heavy bass', "
                    "'calm piano piece with string accompaniment'.",
    )],
    lyrics: Annotated[str | None, Field(
        description="Optional lyrics for vocal music generation.",
    )] = None,
    duration_seconds: Annotated[float, Field(
        description="Target duration in seconds (5-60).",
    )] = 30.0,
    seed: Annotated[int | None, Field(
        description="Random seed for reproducibility.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate music from a text description.

    Uses ACE-Step 1.5 on GPU. Returns {status, data (base64 audio), media_type}.
    """
    forge = _forge(ctx)

    payload: dict[str, Any] = {
        "service": "wan2gp",
        "model": "tts/ace_step_v1_5",
        "prompt": prompt,
        "duration": duration_seconds,
    }
    if lyrics:
        payload["lyrics"] = lyrics
    if seed is not None:
        payload["seed"] = seed

    return await forge.invoke(payload)
