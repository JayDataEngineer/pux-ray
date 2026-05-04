"""ACE-STEP — Music generation from text prompts.

Generates music from text descriptions. Runs inside Ray-managed
container (tech-noir/acestep:latest).

Requires ~8GB VRAM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="ace_step",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class ACEStepDeployment(BaseGPUDeployment):
    """ACE-STEP music generation via Ray native container."""

    def _load(self, model_name: str = "ace-step") -> None:
        self.model_name = model_name
        self.model = True
        logger.info("ACE-STEP ready")

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        body = await request.json()
        prompt = body.get("prompt", body.get("caption", ""))
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        audio_format = body.get("audio_format", "wav")
        data = await asyncio.to_thread(
            self._generate,
            prompt=prompt,
            duration=body.get("duration", 30),
            bpm=body.get("bpm", 120),
            instrumental=body.get("instrumental", True),
            seed=body.get("seed", -1),
            audio_format=audio_format,
            task_type=body.get("task_type", "text2music"),
        )

        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}
        return Response(content=data, media_type=media_types.get(audio_format, "audio/wav"))

    def _generate(
        self, prompt: str, duration: int, bpm: int, instrumental: bool,
        seed: int, audio_format: str, task_type: str,
    ) -> bytes:
        import subprocess

        tmpdir = tempfile.mkdtemp(prefix="acestep_")
        try:
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
                'thinking = false',
            ]
            config_path.write_text("\n".join(toml_lines) + "\n")

            result = subprocess.run(
                ["python", "cli.py", "-c", str(config_path)],
                capture_output=True, text=True, timeout=600,
                cwd="/opt/acestep",
                env={**os.environ, "ACESTEP_CHECKPOINTS_DIR": os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/models/audio/acestep")},
            )
            if result.returncode != 0:
                raise RuntimeError(f"ACE-STEP failed: {result.stderr[-500:]}")

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
