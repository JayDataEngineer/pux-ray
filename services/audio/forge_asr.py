"""Forge adapter for CrispASR — C++ speech recognition with diarization.

Two modes — BOTH do speaker diarization:
  FAST:    Pyannote v3 + TitaNet (5.7MB model, 10.15% DER, CPU-friendly)
  QUALITY: VibeVoice-7B (14GB model, 9.19% DER, joint ASR+diarization)

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

# Mode configs
# FAST: Whisper/Canary transcription + Pyannote v3 diarization (modular pipeline)
# QUALITY: VibeVoice-7B joint transcription + diarization (monolithic)
ASR_MODES = {
    "fast": {
        "model": "/models/asr/parakeet-tdt-0.6b-v3.gguf",
        "backend": "parakeet",
        "diarize_method": "pyannote",
        "segment_model": "/models/asr/pyannote-v3-segmentation-f32.gguf",
        "embedder": "auto",  # TitaNet-Large
        "description": "Pyannote v3 + TitaNet — 10.15% DER, CPU-friendly",
    },
    "quality": {
        "model": "/models/asr/vibevoice-asr-q4_k.gguf",
        "backend": "vibevoice",
        "diarize_method": None,  # Built into VibeVoice
        "segment_model": None,
        "embedder": None,
        "description": "VibeVoice-7B — 9.19% DER, joint ASR + diarization",
    },
}


class ASRForgeService(ForgeService):
    """Calls CrispASR server via OpenAI-compatible HTTP API.

    Both modes produce speaker-diarized transcripts.
    VRAM: 0 — CPU-only, does NOT trigger GPU eviction.
    """

    service_name = "asr"
    default_model = "fast"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self._healthy = False
        self._current_mode = "fast"

    def _find_server(self) -> str | None:
        for url in [ASR_URL, "http://localhost:8080", "http://127.0.0.1:8080"]:
            try:
                with httpx.Client(timeout=5) as client:
                    if client.get(f"{url}/health").status_code == 200:
                        return url
            except Exception:
                continue
        return None

    def load(self, model_name: str | None = None, quant: str | None = None) -> None:
        """Check ASR server and switch mode."""
        mode = model_name or self.default_model
        if mode not in ASR_MODES:
            mode = self.default_mode

        url = self._find_server()
        if url:
            self._asr_url = url
            self._healthy = True
            self._switch_mode(url, mode)
        else:
            logger.warning("ASR: server not reachable")

        self._loaded = True

    def _switch_mode(self, url: str, mode: str):
        """Switch CrispASR to the requested mode."""
        cfg = ASR_MODES[mode]
        try:
            with httpx.Client(timeout=120) as client:
                health = client.get(f"{url}/health").json()
                loaded_backend = health.get("backend", "")

                if loaded_backend != cfg["backend"]:
                    logger.info("ASR: switching to %s (%s)", mode, cfg["description"])
                    client.post(f"{url}/load", data={"model": cfg["model"]})
                    logger.info("ASR: loaded %s", cfg["backend"])

            self._current_mode = mode
        except Exception as e:
            logger.warning("ASR: mode switch failed: %s", e)
            self._current_mode = mode  # Assume it worked

    def infer(self, payload: dict) -> dict:
        """Transcribe + diarize audio.

        payload:
          audio_b64: base64 audio (required)
          mode: "fast" or "quality" (optional override)
          language: ISO-639-1
          translate: bool
          hotwords: comma-separated
          max_speakers: int (hint for clustering)
          vad: bool (voice activity detection)
        """
        audio_b64 = payload.get("audio_b64", "")
        if not audio_b64:
            return {"status": "error", "error": "No audio"}

        # Mode override
        mode = payload.get("mode", self._current_mode)
        if mode not in ASR_MODES:
            mode = self.default_mode

        # Switch if needed
        url = getattr(self, "_asr_url", ASR_URL)
        if mode != self._current_mode:
            self._switch_mode(url, mode)

        cfg = ASR_MODES[mode]

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=600) as client:
                with open(temp_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    data = {"response_format": "verbose_json"}

                    # FAST mode: configure modular diarization pipeline
                    if mode == "fast":
                        data["diarize"] = "true"
                        data["diarize_method"] = cfg["diarize_method"]
                        if cfg.get("segment_model"):
                            data["diarize_segment_model"] = cfg["segment_model"]
                        data["diarize_embedder"] = cfg.get("embedder", "auto")
                        if payload.get("max_speakers"):
                            data["diarize_max_speakers"] = str(payload["max_speakers"])

                    # QUALITY mode: VibeVoice diarizes natively, no extra config needed

                    # Common params
                    for key in ("language", "prompt", "translate", "hotwords",
                                "hotwords_boost", "detect_language", "vad"):
                        if key in payload:
                            data[key] = str(payload[key])

                    resp = client.post(f"{url}/inference", files=files, data=data)
        finally:
            os.unlink(temp_path)

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            return {"status": "error",
                    "error": f"ASR {resp.status_code}: {resp.text[:200]}"}

        result = resp.json()

        # Extract speakers from segments
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
                "backend": result.get("backend", ""),
                "mode": mode,
                "speakers": speakers,
            },
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "mode": mode,
                "backend": result.get("backend", ""),
                "audio_duration_s": result.get("duration"),
                "num_speakers": len(speakers),
            },
        }

    def actual_vram_mb(self) -> int:
        return 0  # CPU-only
