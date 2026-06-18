"""TTS tools — unified speech synthesis across all engines.

- tts_speak: Generate speech via any registered TTS service.
- tts_voices: List available TTS engines dynamically from SERVICE_REGISTRY.

The engine list is built at import time from services.registry — any registered
service with category="tts" automatically appears.  No hardcoded engine dicts.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from loguru import logger
from pydantic import Field

from services.registry import SERVICE_REGISTRY

# ---------------------------------------------------------------------------
# Dynamic engine list from SERVICE_REGISTRY
# ---------------------------------------------------------------------------


def _param_spec_to_engine_param(spec: Any) -> dict:
    """Convert a ParamSpec dataclass to the engine param dict format."""
    p = {
        "name": spec.label.lower().replace(" ", "_") if spec.label else "unknown",
        "type": spec.type,
        "label": spec.label,
    }
    if spec.required:
        p["required"] = True
    if spec.default is not None:
        p["default"] = spec.default
    if spec.placeholder:
        p["placeholder"] = spec.placeholder
    if spec.description:
        p["description"] = spec.description
    if spec.options:
        p["options"] = spec.options
    return p


def _build_engines() -> list[dict]:
    """Build engine catalogue from SERVICE_REGISTRY (category='tts')."""
    engines = []
    for name, entry in SERVICE_REGISTRY.items():
        if entry.category != "tts":
            continue
        # Build engine label with (GPU)/(CPU) suffix
        gpu_suffix = " (GPU)" if entry.needs_gpu else " (CPU)"
        label = entry.label + gpu_suffix

        engines.append({
            "id": name,
            "label": label,
            "gpu": entry.needs_gpu,
            "description": entry.description or f"{entry.label} TTS",
            "params": [_param_spec_to_engine_param(p) for p in (entry.params_schema or [])],
        })
    return engines


ENGINES = _build_engines()

# Keep KOKORO_VOICES available for direct import (used by audio tools and tests)
KOKORO_VOICES = sorted({
    v
    for engine in ENGINES
    for param in engine.get("params", [])
    if param.get("options") and param["name"] == "voice"
    for v in param["options"]
})


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
    """List available TTS engines and per-engine parameter schemas.

    Built dynamically from SERVICE_REGISTRY — any service with category='tts'
    is automatically included.  No hardcoded engine list.
    """
    return {
        "engines": [e for e in ENGINES if e["id"] in _HANDLED_ENGINES],
        "voices": {
            "kokoro": KOKORO_VOICES,
        },
    }


# Engines that tts_speak can actually handle (has a dispatch branch).
# Other category='tts' services (e.g. moss_voicegenerator) have dedicated tools.
_HANDLED_ENGINES = {"kokoro", "moss_tts", "espeak"}

_ENGINE_IDS = sorted(_HANDLED_ENGINES & {e["id"] for e in ENGINES})


async def tts_speak(
    text: Annotated[str, Field(
        description="The text to synthesize into speech.",
    )],
    engine: Annotated[str, Field(
        description="TTS engine to use.",
        enum=_ENGINE_IDS,
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
    """Generate speech from text using any registered TTS engine.

    Engines are loaded dynamically from SERVICE_REGISTRY (category='tts').
    See `tts_voices` for per-engine parameter schemas.
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
            "service": "moss_tts",
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

    return {"status": "error", "error": f"Unknown engine: {engine}. "
            f"Available: {', '.join(_ENGINE_IDS)}"}
