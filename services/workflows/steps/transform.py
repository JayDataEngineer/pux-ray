"""Transform step executor — image/video transforms (resize, convert, extract).

Lightweight CPU operations between GPU steps. Uses PIL for images,
ffmpeg for video transformations.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)

_METHODS = {}


def _register(method_name: str):
    def decorator(fn):
        _METHODS[method_name] = fn
        return fn
    return decorator


class TransformStepExecutor(StepExecutor):
    """Run image/video transform operations."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        method = params.pop("_method", params.pop("method", ""))
        if not method:
            raise ValueError("Transform step missing 'method' param")

        handler = _METHODS.get(method)
        if not handler:
            raise ValueError(f"Unknown transform method: {method}. Available: {list(_METHODS.keys())}")

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
# Transform methods
# ---------------------------------------------------------------------------

@_register("resize")
async def resize_image(params: dict, context: StepContext) -> Path:
    """Resize an image.

    Params:
      image: path to image file
      width: target width
      height: target height
    """
    from PIL import Image

    src = Path(params.get("image", ""))
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))

    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")

    img = Image.open(src)
    img = img.resize((width, height), Image.LANCZOS)

    output = Path(tempfile.mktemp(suffix=".png"))
    img.save(output, "PNG")
    return output


@_register("convert")
async def convert_format(params: dict, context: StepContext) -> Path:
    """Convert image/video format.

    Params:
      input: path to input file
      format: target format (png, jpg, webp, mp4, webm)
    """
    src = Path(params.get("input", ""))
    target_fmt = params.get("format", "png")

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    ext_map = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp", "mp4": ".mp4", "webm": ".webm"}
    ext = ext_map.get(target_fmt, f".{target_fmt}")

    if src.suffix.lstrip(".").lower() in ("mp4", "webm", "avi", "mov"):
        # Video conversion via ffmpeg
        output = Path(tempfile.mktemp(suffix=ext))
        cmd = ["ffmpeg", "-y", "-i", str(src), str(output)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-300:]}")
        return output

    # Image conversion via PIL
    from PIL import Image
    img = Image.open(src)
    if target_fmt in ("jpg", "jpeg") and img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    output = Path(tempfile.mktemp(suffix=ext))
    img.save(output, target_fmt.upper() if target_fmt != "jpg" else "JPEG")
    return output


@_register("extract_frame")
async def extract_frame(params: dict, context: StepContext) -> Path:
    """Extract a single frame from a video.

    Params:
      video: path to video file
      timestamp: time position (e.g., "00:00:01" or frame number)
    """
    src = Path(params.get("video", ""))
    timestamp = params.get("timestamp", "0")

    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    output = Path(tempfile.mktemp(suffix=".png"))
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(src),
        "-frames:v", "1",
        str(output),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract_frame failed: {stderr.decode(errors='replace')[-300:]}")
    return output


@_register("composite")
async def composite_images(params: dict, context: StepContext) -> Path:
    """Overlay one image onto another at a position specified by keypoints.

    Params:
      base_image: path to the base/background image
      overlay_image: path to the image to overlay (e.g., refined face)
      position: either "center" or a dict/JSON with x, y (and optionally width, height)
    """
    from PIL import Image
    import json

    base_path = Path(params.get("base_image", ""))
    overlay_path = Path(params.get("overlay_image", ""))
    position = params.get("position", "center")

    if not base_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_path}")
    if not overlay_path.exists():
        raise FileNotFoundError(f"Overlay image not found: {overlay_path}")

    base = Image.open(base_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Parse position
    if isinstance(position, str):
        try:
            pos = json.loads(position)
        except (json.JSONDecodeError, TypeError):
            pos = position
    else:
        pos = position

    if isinstance(pos, dict):
        # Keypoint format: {x, y} or {x, y, width, height}
        x = int(pos.get("x", 0))
        y = int(pos.get("y", 0))
        w = pos.get("width")
        h = pos.get("height")
        if w and h:
            overlay = overlay.resize((int(w), int(h)), Image.LANCZOS)
    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
        # Keypoint array format: first element is the primary point
        if isinstance(pos[0], dict):
            x = int(pos[0].get("x", 0))
            y = int(pos[0].get("y", 0))
        else:
            x, y = int(pos[0]), int(pos[1])
    else:
        # Default: center the overlay
        x = (base.width - overlay.width) // 2
        y = (base.height - overlay.height) // 2

    # Paste overlay onto base
    base.paste(overlay, (x, y), overlay)

    output = Path(tempfile.mktemp(suffix=".png"))
    base.convert("RGB").save(output, "PNG")
    return output
