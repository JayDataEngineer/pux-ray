"""Forge adapter for CrispASR — C++ speech recognition with diarization.

Source: github.com/CrispStrobe/CrispASR (upstream, unmodified)
Docker: forge-reg.local:30500/tech-noir/asr:latest (built from infra/docker/Dockerfile.asr)
API: OpenAI-compatible POST /v1/audio/transcriptions

Two modes — both do speaker diarization:
  base:  VibeVoice-7B Q4_K (vibevoice-asr-q4_k.gguf) — 9.19% DER, joint ASR+diarization
  turbo: VibeVoice-Realtime 0.5B Q8_0 — low-latency streaming

The diarization pool (base) and diarization-turbo pool are separate containers.
Mode selects which pool URL to call; this adapter talks to whichever is reachable.

VRAM: 1.5GB — coexists with GPU models without eviction.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time

import httpx

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

# K8s service name or direct host:port
ASR_URL = os.environ.get("ASR_URL", "http://asr-service:8080")
ASR_TURBO_URL = os.environ.get("ASR_TURBO_URL", "http://asr-turbo-service:8080")

# Mode configs — maps to the CrispASR container running that model
ASR_MODES = {
    "base": {
        "model": "/models/vibevoice-cpp/vibevoice-asr-q4_k.gguf",
        "backend": "vibevoice",
        "description": "VibeVoice-7B Q4_K — 9.19% DER, joint ASR + diarization",
        "pool_urls": [ASR_URL, "http://localhost:8051", "http://127.0.0.1:8051"],
    },
    "turbo": {
        "model": "/models/vibevoice-cpp/vibevoice-realtime-0.5B-q8_0.gguf",
        "backend": "vibevoice",
        "description": "VibeVoice-Realtime 0.5B Q8_0 — low-latency streaming",
        "pool_urls": [ASR_TURBO_URL, "http://localhost:8055", "http://127.0.0.1:8055"],
    },
}


class ASRForgeService(ForgeService):
    """Calls CrispASR server via OpenAI-compatible HTTP API.

    Modes: "base" (VibeVoice-7B Q4_K) or "turbo" (VibeVoice-Realtime 0.5B).
    Each mode has its own pool container; this adapter probes both and uses
    whichever is reachable for the requested mode.

    VRAM: 1.5GB per container — coexists with GPU models.
    """

    service_name = "asr"
    default_model = "base"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self._current_mode = "base"
        self._mode_urls: dict[str, str] = {}

    def _find_server(self, mode: str) -> str | None:
        """Find a reachable CrispASR container for the given mode."""
        cfg = ASR_MODES.get(mode, ASR_MODES["base"])
        for url in cfg["pool_urls"]:
            try:
                with httpx.Client(timeout=5) as client:
                    if client.get(f"{url}/health").status_code == 200:
                        return url
            except Exception:
                continue
        return None

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        mode = model_name or self.default_model
        if mode not in ASR_MODES:
            mode = "base"
        self._current_mode = mode
        url = self._find_server(mode)
        if url:
            self._mode_urls[mode] = url
            logger.info("ASR: %s mode reachable at %s", mode, url)
        else:
            logger.warning("ASR: %s mode server not reachable — will retry on first request", mode)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self._mode_urls.clear()

    def infer(self, payload: dict) -> dict:
        """Transcribe + diarize audio via CrispASR OpenAI-compatible API.

        payload:
          audio_b64:   base64 audio (required)
          mode:        "base" or "turbo" (default: base)
          language:    ISO-639-1 (optional)
          translate:   bool (optional)
          hotwords:    comma-separated (optional)
          max_speakers: int hint for diarization clustering (optional)
          vad:         bool (optional)
        """
        audio_b64 = payload.get("audio_b64", "")
        if not audio_b64:
            return {"status": "error", "error": "No audio provided"}

        mode = payload.get("mode", self._current_mode)
        if mode not in ASR_MODES:
            mode = "base"

        url = self._mode_urls.get(mode) or self._find_server(mode)
        if not url:
            return {"status": "error", "error": f"CrispASR {mode} server not reachable"}
        self._mode_urls[mode] = url

        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        # OpenAI-compatible multipart form for /v1/audio/transcriptions
        data: dict[str, str] = {
            "response_format": "verbose_json",
            "diarize": "true",
        }
        for key in ("language", "prompt", "translate", "hotwords",
                    "hotwords_boost", "detect_language", "vad"):
            if key in payload:
                data[key] = str(payload[key])
        if payload.get("max_speakers"):
            data["diarize_max_speakers"] = str(payload["max_speakers"])

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=600) as client:
                with open(temp_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    resp = client.post(
                        f"{url}/v1/audio/transcriptions",
                        files=files,
                        data=data,
                    )
        finally:
            os.unlink(temp_path)

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error",
                    "error": f"CrispASR {resp.status_code}: {resp.text[:200]}"}

        result = resp.json()

        speakers = []
        for seg in result.get("segments", []):
            spk = seg.get("speaker") or seg.get("speaker_id")
            if spk and spk not in speakers:
                speakers.append(spk)

        return {
            "status": "success",
            "output": {
                "type": "transcription",
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
                "mode": mode,
                "speakers": speakers,
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "mode": mode,
                "audio_duration_s": result.get("duration"),
                "num_speakers": len(speakers),
            },
        }

    def actual_vram_mb(self) -> int:
        return 0
