"""Forge adapter for CrispASR — C++ speech recognition.

CrispASR runs as a separate container (CPU-only or fractional GPU).
OpenAI-compatible /v1/audio/transcriptions endpoint.
26 ASR backends: Whisper, Parakeet, Canary, Voxtral, etc.

CPU-only mode: no GPU VRAM needed. Can coexist with any service.
GPU mode: fractional allocation for faster transcription.
"""
from __future__ import annotations

import logging
import os
import time
import base64
import tempfile

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

ASR_URL = os.environ.get("ASR_URL", "http://asr-service:8080")


class ASRForgeService(ForgeService):
    """Calls CrispASR server via OpenAI-compatible HTTP API.

    VRAM: 0 (CPU-only by default) or fractional if GPU-accelerated.
    Does NOT trigger GPU eviction — coexists with all services.
    """

    service_name = "asr"
    default_model = "parakeet-tdt-0.6b"
    persistence = Persistence.TRANSIENT
    vram_mb = 0  # CPU-only — no GPU claim

    def __init__(self):
        super().__init__()
        self._healthy = False

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Check ASR server health."""
        for url in [ASR_URL, "http://localhost:8080", "http://127.0.0.1:8080"]:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(f"{url}/health")
                    if resp.status_code == 200:
                        self._asr_url = url
                        data = resp.json()
                        self._healthy = True
                        logger.info("ASR: server healthy at %s (backend: %s)",
                                    url, data.get("backend", "?"))
                        break
            except Exception:
                continue

        if not self._healthy:
            logger.warning("ASR: server not reachable — will try on first request")

        self._loaded = True

    def unload(self) -> None:
        """ASR server stays running — just mark unloaded in forge."""
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        """Transcribe audio via CrispASR OpenAI-compatible API.

        Accepts:
          - audio_b64: base64-encoded audio file
          - response_format: json | text | srt | verbose_json
          - language: ISO-639-1 code
          - translate: bool (translate to English)
          - diarize: bool (speaker diarization)
          - hotwords: comma-separated bias terms
        """
        audio_b64 = payload.get("audio_b64", "")
        if not audio_b64:
            return {"status": "error", "error": "No audio provided"}

        url = getattr(self, "_asr_url", ASR_URL)
        response_format = payload.get("response_format", "json")

        # Decode base64 audio to temp file
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=300) as client:
                # Use /inference for extended features, /v1/audio/transcriptions for OpenAI compat
                endpoint = f"{url}/inference"

                with open(temp_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    data = {"response_format": response_format}

                    # Optional params
                    for key in ("language", "prompt", "translate", "diarize",
                                "hotwords", "hotwords_boost", "detect_language"):
                        if key in payload:
                            data[key] = str(payload[key])

                    resp = client.post(endpoint, files=files, data=data)
        finally:
            os.unlink(temp_path)

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error",
                    "error": f"ASR returned {resp.status_code}: {resp.text[:200]}"}

        result = resp.json()
        return {
            "status": "success",
            "output": {
                "type": "transcription",
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
                "backend": result.get("backend", ""),
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": "crispasr",
                "duration_s": result.get("duration"),
            },
        }

    def actual_vram_mb(self) -> int:
        """ASR is CPU-only — reports 0 VRAM."""
        return 0
