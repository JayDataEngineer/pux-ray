"""Faster-Whisper orchestrator — CTranslate2 transcribe() call.
 
No nn.Modules, no forward() calls. CTranslate2 handles everything internally.
"""
from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FasterWhisperOrchestrator:
    """Faster-Whisper ASR via CTranslate2."""

    def __init__(self, model):
        self.model = model

    def transcribe(self, *, audio_b64: Optional[str] = None, audio_path: Optional[str] = None,
                   language: Optional[str] = None, beam_size: int = 5,
                   seed: int = -1) -> dict:
        import soundfile as sf

        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
        elif audio_path:
            audio_bytes = Path(audio_path).read_bytes()
        else:
            raise ValueError("audio_b64 or audio_path required")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = self.model.transcribe(
                tmp_path,
                language=language,
                beam_size=beam_size,
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

            return {
                "status": "success",
                "text": " ".join(full_text),
                "segments": result_segments,
                "language": info.language,
                "language_probability": info.language_probability,
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    generate = transcribe
