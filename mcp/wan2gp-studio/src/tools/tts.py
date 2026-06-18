"""TTS tools — unified speech synthesis across all engines.

- tts_speak: One endpoint for Kokoro, MOSS VoiceGenerator, eSpeak, IndexTTS.
- tts_voices: List available TTS engines with per-engine parameter schemas.

Note: Qwen3-TTS engine removed — superseded by MOSS VoiceGenerator
(instruction-following + multilingual TTS) and sherpa-onnx Kokoro (CPU TTS).
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

# Voice names match the sherpa-onnx kokoro-multi-lang-v1_0 speaker IDs
# (0–52). Source: k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
# Keep in sync with infra/docker/api_kokoro.py VOICE_NAME_TO_ID.
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

ENGINES = [
    {
        "id": "kokoro",
        "label": "Kokoro (CPU)",
        "gpu": False,
        "description": "Fast CPU text-to-speech via sherpa-onnx. 53 voices, EN+ZH.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True},
            {"name": "voice", "type": "select", "label": "Voice", "default": "af_bella",
             "options": KOKORO_VOICES},
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
        },
    }


async def tts_speak(
    text: Annotated[str, Field(
        description="The text to synthesize into speech.",
    )],
    engine: Annotated[str, Field(
        description="TTS engine to use.",
        enum=["kokoro", "moss_tts", "espeak", "index_tts"],
    )] = "kokoro",
    voice: Annotated[str | None, Field(
        description="Voice preset name. Kokoro: af_bella, af_nova, am_adam, etc.",
    )] = None,
    instruct: Annotated[str | None, Field(
        description="Voice design instruction (moss_tts emotion/style).",
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
    # ── MOSS TTS sampling params (1:1 with MOSS demo) ──
    max_new_tokens: Annotated[int, Field(
        description="[MOSS TTS] Maximum number of tokens to generate. Higher = longer audio. 4096 default.",
    )] = 4096,
    text_temperature: Annotated[float, Field(
        description="[MOSS TTS] Text sampling temperature. Higher = more diverse. 1.0 default.",
    )] = 1.0,
    text_top_p: Annotated[float, Field(
        description="[MOSS TTS] Text nucleus sampling threshold. 0.9 default.",
    )] = 0.9,
    text_top_k: Annotated[int, Field(
        description="[MOSS TTS] Text top-k sampling. 50 default.",
    )] = 50,
    text_repetition_penalty: Annotated[float, Field(
        description="[MOSS TTS] Text repetition penalty. 1.0 = disabled. Higher = less repetition.",
    )] = 1.0,
    audio_temperature: Annotated[float, Field(
        description="[MOSS TTS] Audio sampling temperature. Higher = more diverse. 1.0 default.",
    )] = 1.0,
    audio_top_p: Annotated[float, Field(
        description="[MOSS TTS] Audio nucleus sampling threshold. 0.9 default.",
    )] = 0.9,
    audio_top_k: Annotated[int, Field(
        description="[MOSS TTS] Audio top-k sampling. 50 default.",
    )] = 50,
    audio_repetition_penalty: Annotated[float, Field(
        description="[MOSS TTS] Audio repetition penalty. 1.0 = disabled. Higher = less repetition.",
    )] = 1.0,
    n_vq_for_inference: Annotated[int, Field(
        description="[MOSS TTS] Number of VQ codebooks for inference. Fewer = faster, lower quality. 32 default.",
    )] = 32,
    ctx: Context | None = None,
) -> dict:
    """Generate speech from text using any TTS engine.

    Engines:
      - kokoro: Fast CPU TTS with 53 voice presets (sherpa-onnx, EN+ZH)
      - moss_tts: GPU TTS with full sampling control (1:1 with demo)
      - espeak: Ultra-lightweight CPU TTS with 10 languages
      - index_tts: High-quality GPU TTS for voice cloning
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

    # ── MOSS TTS: GPU voice cloning with full sampling control ──────────
    if engine == "moss_tts":
        payload: dict[str, Any] = {
            "service": "wan2gp",
            "model": "moss-tts",
            "text": text,
            "language": language,
            "max_new_tokens": max_new_tokens,
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "text_repetition_penalty": text_repetition_penalty,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
            "n_vq_for_inference": n_vq_for_inference,
        }
        if instruct:
            payload["instruction"] = instruct
        if ref_audio_b64:
            payload["ref_audio_b64"] = ref_audio_b64
        if seed >= 0:
            payload["seed"] = seed
        return await forge.invoke(payload)

    # ── IndexTTS: GPU voice cloning ──────────────────────────────────────
    if engine == "index_tts":
        return await forge.invoke({
            "service": "wan2gp",
            "model": "index_tts/v2",
            "text": text,
        })

    return {"status": "error", "error": f"Unknown engine: {engine}. "
            f"Available: kokoro, moss_tts, espeak, index_tts"}
