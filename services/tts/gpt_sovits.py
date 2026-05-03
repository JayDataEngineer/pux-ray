"""GPT-SoVITS - voice cloning TTS.

Clones voices from reference audio using GPT-SoVITS.
Supports zero-shot voice cloning from a short reference clip.

Uses CLIToolMixin because GPT-SoVITS has its own dependency tree
that conflicts with the main venv.

Repo: https://github.com/RVC-Boss/GPT-SoVITS
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from ray import serve

from services.base import BaseGPUDeployment, CLIToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "num_cpus": 0.5,
    },
)
class GPTSoVITSDeployment(BaseGPUDeployment, CLIToolMixin):
    """GPT-SoVITS voice cloning TTS via subprocess CLI."""

    def _load(self, model_name: str = "gpt-sovits") -> None:
        self._init_cli("services.tts.gpt_sovits")
        self.model = True
        self.model_name = model_name
        logger.info("GPT-SoVITS CLI ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        self.model = None

    async def synthesize(
        self,
        text: str,
        refer_wav: bytes = b"",
        refer_wav_path: str = "",
        prompt_text: str = "",
        prompt_language: str = "en",
        text_language: str = "en",
    ) -> bytes:
        """Synthesize speech with voice cloning.

        Args:
            text: Text to synthesize.
            refer_wav: Reference audio bytes for voice cloning.
            refer_wav_path: Path to reference audio file (alternative to bytes).
            prompt_text: Transcript of reference audio.
            prompt_language: Language of reference audio transcript.
            text_language: Language of text to synthesize.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        tmpdir = tempfile.mkdtemp(prefix="gptsovits_")
        try:
            output_path = Path(tmpdir) / "output.wav"

            args = [
                "--text", text,
                "--output", str(output_path),
                "--text_language", text_language,
            ]

            # Handle reference audio: bytes uploaded or file path
            if refer_wav:
                refer_path = Path(tmpdir) / "reference.wav"
                refer_path.write_bytes(refer_wav)
                args.extend([
                    "--refer_wav", str(refer_path),
                    "--prompt_text", prompt_text,
                    "--prompt_language", prompt_language,
                ])
            elif refer_wav_path:
                args.extend([
                    "--refer_wav", refer_wav_path,
                    "--prompt_text", prompt_text,
                    "--prompt_language", prompt_language,
                ])

            self._run_cli(args, timeout=300, cwd=self._working_dir)

            if not output_path.exists():
                raise RuntimeError("GPT-SoVITS did not produce audio output")

            return output_path.read_bytes()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        audio = await self.synthesize(
            text=text,
            refer_wav_path=body.get("refer_wav_path", ""),
            prompt_text=body.get("prompt_text", ""),
            prompt_language=body.get("prompt_language", "en"),
            text_language=body.get("text_language", "en"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
