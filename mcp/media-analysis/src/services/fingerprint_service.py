"""Audio fingerprinting service.

Generates Chromaprint fingerprints for audio identification.
Uses fpcalc (Chromaprint binary) for fingerprinting.
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from ..settings import get_settings


class FingerprintService:
    """Audio fingerprinting via Chromaprint/fpcalc."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def fingerprint_audio(self, audio_url: str) -> dict:
        """Generate an audio fingerprint for identification."""
        settings = get_settings()
        if not settings.is_enabled("fingerprint"):
            return {"success": False, "error": "Audio fingerprinting is disabled"}

        try:
            # Download audio
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(audio_url, headers={"User-Agent": "MediaAnalysis/1.0"})
                response.raise_for_status()

            suffix = Path(audio_url).suffix or ".wav"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(response.content)
            tmp.close()

            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self._fingerprint_sync, tmp.name,
                    )
                    return result
            finally:
                os.unlink(tmp.name)

        except Exception as e:
            logger.error(f"Audio fingerprinting error: {e}")
            return {"success": False, "error": f"Audio fingerprinting error: {str(e)[:200]}"}

    def _fingerprint_sync(self, audio_path: str) -> dict:
        """Run fpcalc on audio file and parse output."""
        result = subprocess.run(
            ["fpcalc", "-json", audio_path],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"fpcalc failed: {result.stderr[:200]}")

        import json
        data = json.loads(result.stdout)

        return {
            "success": True,
            "fingerprint": data.get("fingerprint", ""),
            "duration": data.get("duration", 0),
        }

    async def close(self) -> None:
        pass


_fingerprint_service: FingerprintService | None = None


def get_fingerprint_service() -> FingerprintService:
    global _fingerprint_service
    if _fingerprint_service is None:
        _fingerprint_service = FingerprintService()
    return _fingerprint_service
