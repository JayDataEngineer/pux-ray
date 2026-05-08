"""Faster-Whisper ASR - CPU-capable speech recognition.

Uses the faster-whisper library (CTranslate2 backend).
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("FASTER_WHISPER_MODEL_PATH", "/models/asr/faster-whisper")


@serve.deployment(
    name="faster_whisper",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0},
    max_ongoing_requests=4,
)
class FasterWhisperASR(BaseGPUDeployment):
    """Faster-Whisper ASR. CPU by default."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = None

    def _load(self, model_name: str = "distil-large-v3") -> None:
        from faster_whisper import WhisperModel

        model_path = Path(MODEL_PATH)
        if model_path.exists() and any(model_path.iterdir()):
            self.model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
        else:
            raise FileNotFoundError(
                f"Faster-Whisper model not found at {MODEL_PATH}. "
                f"Run model-sync to download it."
            )

        self.model_name = model_name
        logger.info("Faster-Whisper loaded: %s (CPU)", model_name)

    def _unload(self) -> None:
        self.model = None
        self.model_name = None
        super()._unload()

    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        model: str = "distil-large-v3",
    ) -> dict:
        """Transcribe audio bytes. Returns segments and text."""
        if not self.is_loaded():
            self.load_model(model)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        segments, info = self.model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        result_segments = []
        full_text = []
        for seg in segments:
            result_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
            full_text.append(seg.text)

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "text": " ".join(full_text),
            "segments": result_segments,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def _extract_input(self, inp) -> dict:
        result = super()._extract_input(inp)
        if inp.audio_b64:
            from services.base import _b64_decode
            result["audio"] = _b64_decode(inp.audio_b64)
        return result

    async def __call__(self, request):
        """TNAP endpoint. Supports JSON with audio_b64 or multipart."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()
        content_type = request.headers.get("content-type", "")

        try:
            if "multipart/form-data" in content_type:
                form = await request.form()
                audio_file = form["file"]
                audio_bytes = await audio_file.read()
                language = form.get("language")

                if "config" in form:
                    requested = InferenceConfig(**json.loads(str(form["config"])))
                    if requested != self.config:
                        self.config = requested

                result = await asyncio.to_thread(
                    lambda: self.transcribe(audio_bytes, language),
                )
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                audio_bytes = extracted.get("audio")
                if not audio_bytes:
                    return JSONResponse(self.handle_error("audio_b64 required"), status_code=400)

                language = extracted.get("language")
                result = await asyncio.to_thread(
                    lambda: self.transcribe(audio_bytes, language),
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    json.dumps(result).encode("utf-8"),
                    "application/json",
                    latency_ms,
                    extra_metrics={"language": result.get("language", "")},
                )
            )
        except Exception as e:
            logger.error("faster_whisper error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)