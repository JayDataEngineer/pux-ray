"""ACE-STEP - Music generation from text prompts.

Generates music from text descriptions. Called via subprocess using
ACE-STEP's own venv Python (torch 2.10+cu128) because it has
incompatible dependencies with other tools.

Requires ~8GB VRAM. Model loads fresh per subprocess call.
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
    name="ace_step",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0.01},
)
class ACEStepDeployment(BaseGPUDeployment, CLIToolMixin):
    """ACE-STEP music generation via subprocess CLI."""

    def _load(self, model_name: str = "ace-step") -> None:
        self._init_cli("services.creative.ace_step")
        self.model = True
        self.model_name = model_name
        logger.info("ACE-STEP CLI tool ready (model loads per-call in subprocess)")

    def _unload(self) -> None:
        self.model = None

    async def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        bpm: int = 120,
        instrumental: bool = True,
        seed: int = -1,
        audio_format: str = "wav",
        task_type: str = "text2music",
    ) -> bytes:
        """Generate music from text prompt. Returns audio bytes."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        tmpdir = tempfile.mkdtemp(prefix="acestep_")
        try:
            # ACE-STEP requires a TOML config file
            config_path = Path(tmpdir) / "config.toml"
            toml_lines = [
                f'caption = """{prompt}"""',
                f'task_type = "{task_type}"',
                f'instrumental = {str(instrumental).lower()}',
                f'duration = {duration}',
                f'bpm = {bpm}',
                f'seed = {seed}',
                f'batch_size = 1',
                f'audio_format = "{audio_format}"',
                f'save_dir = "{tmpdir}"',
            ]
            config_path.write_text("\n".join(toml_lines) + "\n")

            args = ["-c", str(config_path)]
            self._run_cli(args, timeout=600)

            # Find generated audio files
            audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
            audio_files = sorted(
                [f for f in Path(tmpdir).iterdir()
                 if f.suffix.lower() in audio_exts and f.name != "config.toml"],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            if not audio_files:
                raise RuntimeError("ACE-STEP did not produce audio output")

            return audio_files[0].read_bytes()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def __call__(self, request):
        body = await request.json()
        prompt = body.get("prompt", body.get("caption", ""))
        if not prompt:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        audio_data = await self.generate_music(
            prompt=prompt,
            duration=body.get("duration", 30),
            bpm=body.get("bpm", 120),
            instrumental=body.get("instrumental", True),
            seed=body.get("seed", -1),
            audio_format=body.get("audio_format", "wav"),
        )
        from starlette.responses import Response
        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}
        fmt = body.get("audio_format", "wav")
        return Response(content=audio_data, media_type=media_types.get(fmt, "audio/wav"))
