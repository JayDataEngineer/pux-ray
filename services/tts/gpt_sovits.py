"""GPT-SoVITS — Voice cloning TTS.

Clones voices from reference audio using GPT-SoVITS.
Runs inside Ray-managed container (tech-noir/gptsovits:latest).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class GPTSoVITSDeployment(BaseGPUDeployment):
    """GPT-SoVITS voice cloning TTS via Ray native container."""

    def _load(self, model_name: str = "gpt-sovits") -> None:
        self.model_name = model_name
        self.model = True
        logger.info("GPT-SoVITS ready")

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            return JSONResponse({"error": "input text is required"}, status_code=400)

        refer_wav_path = body.get("refer_wav_path", "")

        data = await asyncio.to_thread(
            self._synthesize,
            text=text,
            refer_wav_path=refer_wav_path,
            prompt_text=body.get("prompt_text", ""),
            prompt_language=body.get("prompt_language", "en"),
            text_language=body.get("text_language", "en"),
        )

        return Response(content=data, media_type="audio/wav")

    def _synthesize(
        self, text: str, refer_wav_path: str,
        prompt_text: str, prompt_language: str, text_language: str,
    ) -> bytes:
        import subprocess

        tmpdir = tempfile.mkdtemp(prefix="gptsovits_")
        try:
            refer_bytes = Path(refer_wav_path).read_bytes() if refer_wav_path else b""
            refer_path = Path(tmpdir) / "reference.wav"
            refer_path.write_bytes(refer_bytes)
            output_path = Path(tmpdir) / "output.wav"

            result = subprocess.run(
                [
                    "python", "api_v2.py",
                    "--text", text,
                    "--text_language", text_language,
                    "--refer_wav", str(refer_path),
                    "--prompt_text", prompt_text,
                    "--prompt_language", prompt_language,
                    "--output", str(output_path),
                ],
                capture_output=True, text=True, timeout=300,
                cwd="/opt/gpt-sovits",
            )
            if result.returncode != 0:
                raise RuntimeError(f"GPT-SoVITS failed: {result.stderr[-500:]}")

            if not output_path.exists():
                raise RuntimeError("GPT-SoVITS did not produce audio output")

            return output_path.read_bytes()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
