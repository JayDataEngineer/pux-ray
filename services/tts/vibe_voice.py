"""VibeVoice TTS - long-form multi-speaker speech synthesis.

Generates expressive, long-form audio (podcasts, conversations) from text
using the VibeVoice 7B model. Supports up to 4 speakers, up to 45 min output.

Uses CLIToolMixin because VibeVoice requires:
- transformers==4.51.3 (pinned, conflicts with main venv)
- Compiled flash-attn for CUDA
- Custom vibevoice package (VibeVoiceForConditionalGenerationInference)

Code repo: https://github.com/microsoft/VibeVoice
Model weights: vibevoice/VibeVoice-7B on HuggingFace (18.7GB, community re-upload)
ASR is separate: services.asr.gpu_asr.VibeVoiceASRDeployment
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
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0.01,
        "num_cpus": 0.5,
    },
)
class VibeVoiceDeployment(BaseGPUDeployment, CLIToolMixin):
    """VibeVoice long-form multi-speaker TTS via subprocess CLI."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        self._init_cli("services.tts.vibe_voice")
        self.model = True
        self.model_name = model_name
        logger.info("VibeVoice CLI ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        self.model = None

    async def synthesize(
        self,
        text: str,
        speaker_names: list[str] | None = None,
        model_path: str = "vibevoice/VibeVoice-7B",
    ) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Script text. Use "Speaker 1: ...\nSpeaker 2: ..." for
                  multi-speaker. Plain text becomes single-speaker.
            speaker_names: Voice names mapped to speakers (default: ["Andrew"]).
                           Must match voice files in VibeVoice/voices/ directory.
            model_path: HuggingFace model ID or local path.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        if not speaker_names:
            speaker_names = ["Andrew"]

        tmpdir = tempfile.mkdtemp(prefix="vibevoice_")
        try:
            # Normalize text: wrap plain text as single-speaker script
            if "Speaker " not in text and "speaker " not in text:
                text = f"Speaker 1: {text}"

            txt_path = Path(tmpdir) / "input.txt"
            txt_path.write_text(text)

            args = [
                "--model_path", model_path,
                "--txt_path", str(txt_path),
                "--speaker_names", *speaker_names,
                "--output_dir", tmpdir,
                "--device", "cuda",
            ]

            self._run_cli(args, timeout=600, cwd=self._working_dir)

            # Find generated audio (script outputs {filename}_generated.wav)
            audio_files = sorted(
                Path(tmpdir).glob("*_generated.wav"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            # Fallback: any wav file that isn't our input
            if not audio_files:
                audio_files = sorted(
                    [f for f in Path(tmpdir).iterdir()
                     if f.suffix == ".wav" and f.name != "input.txt"],
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )

            if not audio_files:
                raise RuntimeError("VibeVoice did not produce audio output")

            return audio_files[0].read_bytes()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = speaker_names.split(",")

        audio = await self.synthesize(
            text=text,
            speaker_names=speaker_names,
            model_path=body.get("model_path", "vibevoice/VibeVoice-7B"),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
