"""HTTP client for Qwen3-TTS via API ingress.

Routes all TTS inference through the Ray cluster. Never loads the model locally.

Endpoints:
    POST /tts/faster-qwen3-tts         — base model inference
    GET  /tts/faster-qwen3-tts/health  — health check

Usage:
    from qwen_tts.client import generate

    wav_bytes = generate("Hello world", speaker="Ryan")
    wav_bytes = generate("Hello world", speaker="Ryan", emotion="happy")
"""

import os
from typing import Optional

import httpx

RAY_API_URL = os.environ.get(
    "RAY_API_URL",
    "http://100.86.69.57:30080",
)
TTS_ENDPOINT = f"{RAY_API_URL}/tts/faster-qwen3-tts"


def health_check(timeout: float = 10.0) -> bool:
    """Check if the Qwen TTS service on Ray is reachable."""
    try:
        resp = httpx.get(f"{TTS_ENDPOINT}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def list_speakers(timeout: float = 10.0) -> list[str]:
    """Get the list of available base speakers from the TTS service."""
    try:
        resp = httpx.get(f"{TTS_ENDPOINT}/speakers", timeout=timeout)
        resp.raise_for_status()
        return resp.json()["speakers"]
    except Exception:
        return []


def generate(
    text: str,
    speaker: str = "Ryan",
    language: str = "auto",
    instruct: Optional[str] = None,
    emotion: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_scale: float = 0.3,
    seed: Optional[int] = None,
    timeout: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> bytes:
    """Generate speech from text via Ray Qwen3-TTS.

    Args:
        text: The text to synthesize.
        speaker: Speaker name (base model voices: Ryan, Sohee, Vivian, etc.).
        language: Language code, "auto" detects from text.
        instruct: Raw emotion instruct string (overrides emotion preset).
        emotion: Named emotion preset (see EMOTION_PRESETS).
        lora_path: Path to LoRA adapter on the Ray cluster.
        lora_scale: LoRA strength (0.2-0.5, default 0.3).
        seed: Random seed for reproducibility.
        timeout: Request timeout in seconds.
        client: Optional httpx client for connection reuse.

    Returns:
        Raw WAV audio bytes.

    Raises:
        httpx.HTTPError: On HTTP/connection errors.
        RuntimeError: On TTS service error.
    """
    from qwen_tts.emotions import emotion_to_instruct

    if instruct is None and emotion is not None:
        instruct = emotion_to_instruct(emotion)

    payload: dict = {
        "input": text,
        "voice": speaker,
        "language": language,
    }
    if instruct:
        payload["instruct"] = instruct
    if lora_path:
        payload["lora_path"] = lora_path
        payload["lora_scale"] = lora_scale
    if seed is not None:
        payload["seed"] = seed

    endpoint = TTS_ENDPOINT

    _client = client or httpx
    resp = _client.post(endpoint, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TTS failed ({resp.status_code}): {resp.text[:500]}"
        )
    return resp.content


def generate_lora(
    text: str,
    speaker: str,
    lora_path: str,
    instruct: Optional[str] = None,
    emotion: Optional[str] = None,
    lora_scale: float = 0.3,
    seed: Optional[int] = None,
    timeout: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> bytes:
    """Generate speech using a LoRA-adapted voice.

    Convenience wrapper around generate() with lora_path required.
    """
    return generate(
        text=text,
        speaker=speaker,
        lora_path=lora_path,
        instruct=instruct,
        emotion=emotion,
        lora_scale=lora_scale,
        seed=seed,
        timeout=timeout,
        client=client,
    )


def generate_stream(
    text: str,
    speaker: str = "Ryan",
    language: str = "auto",
    instruct: Optional[str] = None,
    emotion: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: float = 120.0,
) -> bytes:
    """Generate speech with streaming response (for long texts).

    The endpoint streams audio chunks as they're generated.
    Returns the complete WAV bytes assembled from all chunks.
    """
    from qwen_tts.emotions import emotion_to_instruct

    if instruct is None and emotion is not None:
        instruct = emotion_to_instruct(emotion)

    payload: dict = {
        "input": text,
        "voice": speaker,
        "language": language,
        "stream": True,
    }
    if instruct:
        payload["instruct"] = instruct

    with httpx.stream(
        "POST", TTS_ENDPOINT, json=payload, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        for chunk in resp.iter_raw(chunk_size=chunk_size):
            chunks.append(chunk)
        return b"".join(chunks)
