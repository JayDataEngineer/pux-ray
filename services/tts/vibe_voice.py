"""VibeVoice TTS - long-form multi-speaker speech synthesis.

Generates expressive, long-form audio (podcasts, conversations) from text
using the VibeVoice 7B model. Supports up to 4 speakers, up to 45 min output.

Uses CLIToolMixin subprocess pattern — the tool's venv has compiled
flash-attn and pinned transformers==4.51.3.

Code repo: https://github.com/microsoft/VibeVoice
Model weights: vibevoice/VibeVoice-7B on HuggingFace (18.7GB, community re-upload)
ASR is separate: services.asr.gpu_asr.VibeVoiceASRDeployment
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import Response

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
        logger.info("VibeVoice CLI ready")

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        body = await request.json()
        text = body.get("input", "")
        if not text:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "input text is required"}, status_code=400)

        speaker_names = body.get("speaker_names", ["Andrew"])
        if isinstance(speaker_names, str):
            speaker_names = speaker_names.split(",")

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "input.txt"
            txt_path.write_text(
                "\n".join(f"Speaker {i+1}: {text}" for i in range(len(speaker_names)))
            )

            result = self._run_cli(
                args=[
                    "--model_path", "/models/VibeVoice-7B",
                    "--txt_path", str(txt_path),
                    "--speaker_names", *speaker_names,
                    "--output_dir", tmpdir,
                ],
                extra_env={"MODEL_PATH": "/models/VibeVoice-7B"},
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"VibeVoice failed (exit {result.returncode}): "
                    f"{result.stderr[-500:]}"
                )

            # Find the output wav file
            wav_files = list(Path(tmpdir).glob("*.wav"))
            if not wav_files:
                raise RuntimeError("VibeVoice produced no output file")

            audio_data = wav_files[0].read_bytes()

        return Response(content=audio_data, media_type="audio/wav")
