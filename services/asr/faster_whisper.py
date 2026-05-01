"""Faster-Whisper ASR - CPU-capable speech recognition.

Uses the faster-whisper library (CTranslate2 backend).
Can run on CPU or GPU.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="faster_whisper",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0},
    max_ongoing_requests=4,
)
class FasterWhisperASR:
    """Faster-Whisper ASR. CPU by default."""

    def __init__(self):
        self.model = None
        self.model_name = None

    def _ensure_model(self, model_size: str = "distil-large-v3"):
        if self.model is not None and self.model_name == model_size:
            return

        from faster_whisper import WhisperModel
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        model_path = registry.get_path("asr", "faster-whisper")

        if model_path and model_path.exists() and any(model_path.iterdir()):
            self.model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
        else:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self.model_name = model_size
        logger.info("Faster-Whisper loaded: %s (CPU)", model_size)

    async def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        model: str = "distil-large-v3",
    ) -> dict:
        """Transcribe audio bytes. Returns segments and text."""
        self._ensure_model(model)

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

    async def __call__(self, request):
        form = await request.form()
        audio_file = form["file"]
        audio_bytes = await audio_file.read()
        model_name = form.get("model", "distil-large-v3")
        language = form.get("language")

        result = await self.transcribe(
            audio=audio_bytes,
            language=language,
            model=model_name,
        )
        from starlette.responses import JSONResponse
        return JSONResponse(result)
