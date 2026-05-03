"""ACE-STEP API server — text-to-music generation inside Docker."""
from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import Response
import uvicorn

app = FastAPI(title="ACE-STEP API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(data: dict):
    """Generate music from text prompt. Returns audio bytes."""
    prompt = data.get("prompt", data.get("caption", ""))
    if not prompt:
        return {"error": "prompt is required"}

    duration = data.get("duration", 30)
    bpm = data.get("bpm", 120)
    instrumental = data.get("instrumental", True)
    seed = data.get("seed", -1)
    audio_format = data.get("audio_format", "wav")
    task_type = data.get("task_type", "text2music")

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

        import subprocess
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

        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}
        return Response(
            content=audio_files[0].read_bytes(),
            media_type=media_types.get(audio_format, "audio/wav"),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
