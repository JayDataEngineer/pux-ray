"""Compose step executor — ffmpeg-based audio/video composition.

Handles mixing, concatenation, and overlay operations. Executes ffmpeg
subprocesses with strict timeouts and comprehensive error capture.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 300  # 5 minutes max per operation

# Method → handler mapping
_METHODS = {}


def _register(method_name: str):
    def decorator(fn):
        _METHODS[method_name] = fn
        return fn
    return decorator


class ComposeStepExecutor(StepExecutor):
    """Run ffmpeg composition operations on artifact files."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        method = params.pop("_method", params.pop("method", ""))
        if not method:
            raise ValueError("Compose step missing 'method' param")

        handler = _METHODS.get(method)
        if not handler:
            raise ValueError(f"Unknown compose method: {method}. Available: {list(_METHODS.keys())}")

        output_path = await handler(params, context)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        outputs = {}
        if output_path and output_path.exists():
            ref = await context.artifacts.store_from_file(
                context.run_id, context.step_id, "output", output_path
            )
            outputs["output"] = str(ref.file_path)

        return StepResult(outputs=outputs, duration_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Compose methods
# ---------------------------------------------------------------------------

@_register("ffmpeg_mix")
async def mix_audio(params: dict, context: StepContext) -> Path:
    """Mix multiple audio tracks into one.

    Params:
      tracks: list of file paths to audio files
      weights: optional list of volume weights (default: equal)
    """
    tracks = params.get("tracks", [])
    if not tracks:
        raise ValueError("ffmpeg_mix requires 'tracks' param")

    weights = params.get("weights")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    inputs = []
    for t in tracks:
        path = Path(t)
        if not path.exists():
            raise FileNotFoundError(f"Audio track not found: {path}")
        cmd.extend(["-i", str(path)])
        inputs.append(path)

    # Complex filter for mixing
    n = len(inputs)
    if weights:
        filter_parts = [f"[{i}]volume={weights[i]}[a{i}]" for i in range(n)]
        mix_inputs = "".join(f"[a{i}]" for i in range(n))
        filter_str = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={n}:duration=longest[out]"
    else:
        mix_inputs = "".join(f"[{i}]" for i in range(n))
        filter_str = f"{mix_inputs}amix=inputs={n}:duration=longest[out]"

    cmd.extend(["-filter_complex", filter_str, "-map", "[out]"])

    output = Path(tempfile.mktemp(suffix=".wav", dir=_tmp_dir(context)))
    cmd.append(str(output))

    await _run_ffmpeg(cmd)
    return output


@_register("ffmpeg_concat")
async def concat_video(params: dict, context: StepContext) -> Path:
    """Concatenate video files.

    Params:
      segments: list of file paths to video files
    """
    segments = params.get("segments", [])
    if not segments:
        raise ValueError("ffmpeg_concat requires 'segments' param")

    # Create concat list file
    concat_file = Path(tempfile.mktemp(suffix=".txt", dir=_tmp_dir(context)))
    lines = []
    for s in segments:
        path = Path(s)
        if not path.exists():
            raise FileNotFoundError(f"Video segment not found: {path}")
        lines.append(f"file '{path}'")
    concat_file.write_text("\n".join(lines))

    output = Path(tempfile.mktemp(suffix=".mp4", dir=_tmp_dir(context)))
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output),
    ]

    await _run_ffmpeg(cmd)
    concat_file.unlink(missing_ok=True)
    return output


@_register("ffmpeg_overlay")
async def overlay_audio(params: dict, context: StepContext) -> Path:
    """Overlay audio onto a video.

    Params:
      video: path to video file
      audio: path to audio file
    """
    video = Path(params.get("video", ""))
    audio = Path(params.get("audio", ""))

    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    output = Path(tempfile.mktemp(suffix=".mp4", dir=_tmp_dir(context)))
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output),
    ]

    await _run_ffmpeg(cmd)
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_ffmpeg(cmd: list[str]) -> str:
    """Run ffmpeg with strict timeout and error capture."""
    logger.info("Running: %s", " ".join(cmd[:6]) + ("..." if len(cmd) > 6 else ""))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_FFMPEG_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s")

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {err_msg[-500:]}")

    return stdout.decode(errors="replace")


def _tmp_dir(context: StepContext) -> str:
    """Get or create a temp directory for this run's compose outputs."""
    d = Path(f"/tmp/workflow_compose/{context.run_id}")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
