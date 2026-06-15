"""Forge adapter for CrispASR — C++ speech recognition.

Two modes:
  FAST:    Parakeet TDT 0.6B — CPU-only, sub-second, no diarization
  QUALITY: VibeVoice-7B     — 7B LLM, joint transcription + speaker diarization

CPU-only — does NOT trigger GPU eviction. Coexists with everything.
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

# Model configs
ASR_MODELS = {
    "fast": {
        "model": "/models/asr/parakeet-tdt-0.6b-v3.gguf",
        "backend": "parakeet",
        "description": "Parakeet TDT 0.6B — fast, CPU-only",
    },
    "quality": {
        "model": "/models/asr/vibevoice-asr-q4_k.gguf",
        "backend": "vibevoice",
        "description": "VibeVoice-7B — joint ASR + diarization",
    },
}


class ASRForgeService(ForgeService):
    """Calls CrispASR server via OpenAI-compatible HTTP API.

    VRAM: 0 — CPU-only, does NOT trigger GPU eviction.
    """

    service_name = "asr"
    default_model = "fast"
    persistence = Persistence.TRANSIENT
    vram_mb = 0  # CPU-only

    def __init__(self):
        super().__init__()
        self._healthy = False
        self._current_mode = "fast"

    def _find_server(self) -> str | None:
        """Find reachable CrispASR server."""
        for url in [ASR_URL, "http://localhost:8080", "http://127.0.0.1:8080"]:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(f"{url}/health")
                    if resp.status_code == 200:
                        return url
            except Exception:
                continue
        return None

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Check ASR server. model_name selects fast/quality mode."""
        mode = model_name or self.default_model
        if mode not in ASR_MODELS:
            mode = self.default_model

        url = self._find_server()
        if url:
            self._asr_url = url
            self._healthy = True
            self._current_mode = mode

            # Switch model if different from what's loaded
            cfg = ASR_MODELS[mode]
            try:
                with httpx.Client(timeout=120) as client:
                    health = client.get(f"{url}/health")
                    loaded_backend = health.json().get("backend", "")
                    if loaded_backend != cfg["backend"]:
                        logger.info("ASR: switching to %s (%s)", mode, cfg["description"])
                        client.post(f"{url}/load", data={
                            "model": cfg["model"],
                        })
                        logger.info("ASR: switched to %s", cfg["backend"])
            except Exception as e:
                logger.warning("ASR: model switch failed: %s", e)
        else:
            logger.warning("ASR: server not reachable")

        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, payload: dict) -> dict:
        """Transcribe audio.

        payload:
          audio_b64: base64-encoded audio (required)
          mode: "fast" or "quality" (optional, overrides load)
          diarize: bool (quality mode auto-diarizes)
          response_format: json | text | srt | verbose_json
          language: ISO-639-1
          translate: bool (translate to English)
          hotwords: comma-separated bias terms
        """
        audio_b64 = payload.get("audio_b64", "")
        if not audio_b64:
            return {"status": "error", "error": "No audio provided"}

        # Check mode override
        mode = payload.get("mode", self._current_mode)
        if mode not in ASR_MODELS:
            mode = self.default_model

        # Switch model if needed
        if mode != self._current_mode:
            self.load(mode)

        url = getattr(self, "_asr_url", ASR_URL)
        response_format = payload.get("response_format", "verbose_json")

        # Decode audio to temp file
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=600) as client:
                with open(temp_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    data = {"response_format": response_format}

                    # Quality mode: enable diarization by default
                    if mode == "quality" and "diarize" not in payload:
                        data["diarize"] = "true"

                    # Optional params
                    for key in ("language", "prompt", "translate", "diarize",
                                "hotwords", "hotwords_boost", "detect_language"):
                        if key in payload:
                            data[key] = str(payload[key])

                    resp = client.post(f"{url}/inference", files=files, data=data)
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
                "mode": mode,
                "speakers": [s.get("speaker", "") for s in result.get("segments", [])
                             if s.get("speaker")] if mode == "quality" else [],
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": mode,
                "backend": result.get("backend", ""),
                "audio_duration_s": result.get("duration"),
            },
        }

    def actual_vram_mb(self) -> int:
        """CPU-only — 0 VRAM."""
        return 0
