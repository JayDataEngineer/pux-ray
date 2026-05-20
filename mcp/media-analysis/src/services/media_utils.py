"""Shared media loading utilities.

Handles data URIs, HTTP URLs, and raw base64 strings uniformly.
All services should use load_image() instead of their own _load_image().
"""

import base64
import io
import os
import re
import uuid
from pathlib import Path

import httpx
from PIL import Image

_DATA_URI_RE = re.compile(r"^data:([\w/+.-]+);base64,(.+)$", re.DOTALL)

# Allowed MIME types for uploads
ALLOWED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/webp", "image/bmp", "image/tiff",
    "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
    "audio/flac", "audio/x-wav", "audio/x-m4a",
    "video/mp4", "video/webm", "video/avi",
}

MIME_TO_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/x-wav": ".wav",
    "audio/x-m4a": ".m4a",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/avi": ".avi",
}

UPLOAD_DIR = Path("/tmp/media-uploads")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


async def load_image(source: str) -> Image.Image:
    """Load a PIL Image from a URL, data URI, or raw base64 string."""
    if not source:
        raise ValueError("image source is empty")

    # Data URI: data:image/png;base64,iVBOR...
    m = _DATA_URI_RE.match(source)
    if m:
        data = base64.b64decode(m.group(2))
        img = Image.open(io.BytesIO(data))
        img.load()
        return img

    # HTTP(S) URL
    if source.startswith("http://") or source.startswith("https://"):
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(source, headers={"User-Agent": "MediaAnalysis/1.0"})
            resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        return img

    # Fall back: treat as raw base64
    try:
        data = base64.b64decode(source)
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        raise ValueError(f"Cannot load image from source (length={len(source)}, prefix={source[:40]}...)")


async def load_bytes(source: str) -> bytes:
    """Load raw bytes from a URL or data URI. Same logic as load_image but returns bytes."""
    if not source:
        raise ValueError("media source is empty")

    m = _DATA_URI_RE.match(source)
    if m:
        return base64.b64decode(m.group(2))

    if source.startswith("http://") or source.startswith("https://"):
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(source, headers={"User-Agent": "MediaAnalysis/1.0"})
            resp.raise_for_status()
        return resp.content

    try:
        return base64.b64decode(source)
    except Exception:
        raise ValueError(f"Cannot load media from source (length={len(source)}, prefix={source[:40]}...)")


def sanitize_mime(mime_type: str) -> str:
    """Validate and normalize a MIME type. Raises ValueError on disallowed types."""
    mime_type = mime_type.strip().lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Disallowed MIME type: {mime_type!r}. Allowed: {sorted(ALLOWED_MIME_TYPES)}")
    return mime_type


def save_upload(data_b64: str, mime_type: str) -> str:
    """Decode base64 data, validate size, save to upload dir, return filename.

    Returns the filename (e.g. 'abc123.png'), NOT the full path.
    """
    mime_type = sanitize_mime(mime_type)

    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")

    if len(raw) > MAX_UPLOAD_SIZE:
        raise ValueError(f"Upload too large: {len(raw)} bytes (max {MAX_UPLOAD_SIZE})")

    ext = MIME_TO_EXT.get(mime_type, ".bin")
    filename = f"{uuid.uuid4().hex}{ext}"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename

    # Write atomically via temp file to avoid partial reads
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.rename(path)

    return filename


def cleanup_uploads(max_age_seconds: int = 3600) -> int:
    """Delete uploaded files older than max_age_seconds. Returns count deleted."""
    if not UPLOAD_DIR.exists():
        return 0

    import time
    cutoff = time.time() - max_age_seconds
    count = 0
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and not f.suffix == ".tmp" and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            count += 1
    return count
