"""eSpeak orchestrator — subprocess call to espeak-ng binary.
 
No nn.Modules, no forward() calls. Just subprocess.exec().
"""
from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class EspeakOrchestrator:
    """eSpeak TTS via subprocess."""

    def __init__(self, bin_path: str = "espeak-ng"):
        self.bin_path = bin_path

    def generate(self, *, text: str = "", voice: str = "en", speed: int = 175, pitch: int = 50,
                 seed: int = -1) -> dict:
        if not text:
            raise ValueError("text required")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            cmd = [
                self.bin_path,
                "-v", voice,
                "-s", str(speed),
                "-p", str(pitch),
                "-w", out_path,
                text,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            wav_bytes = Path(out_path).read_bytes()
        finally:
            Path(out_path).unlink(missing_ok=True)

        return {
            "status": "success",
            "data": base64.b64encode(wav_bytes).decode(),
            "media_type": "audio/wav",
        }
