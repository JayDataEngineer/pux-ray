"""Artifact store — file-based intermediate data management.

Stores workflow artifacts (images, video, audio) on PVC. The interface is
abstracted so the backend can be swapped to S3/MinIO for multi-node clusters
without changing any StepExecutor code.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE = Path("/models/workflows")


class ArtifactRef:
    """Lightweight reference to a stored artifact."""

    __slots__ = ("run_id", "step_id", "name", "file_path", "media_type",
                 "url", "size_bytes", "created_at")

    def __init__(
        self,
        run_id: str,
        step_id: str,
        name: str,
        file_path: Path,
        media_type: str,
        url: str,
        size_bytes: int,
        created_at: datetime,
    ):
        self.run_id = run_id
        self.step_id = step_id
        self.name = name
        self.file_path = file_path
        self.media_type = media_type
        self.url = url
        self.size_bytes = size_bytes
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "name": self.name,
            "file_path": str(self.file_path),
            "media_type": self.media_type,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactRef:
        d = dict(d)
        d["file_path"] = Path(d["file_path"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


class ArtifactStore:
    """Manages workflow artifacts on disk.

    Storage layout:
        /mnt/data/workflows/{run_id}/{step_id}/{name}.ext

    The public interface is stable. For multi-node clusters, subclass and
    override _write_bytes / _read_bytes / _move_file to use S3/MinIO.
    """

    def __init__(self, base_dir: Path = _DEFAULT_BASE, url_prefix: str = "/v1/wf"):
        self.base_dir = base_dir
        self.url_prefix = url_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(
        self,
        run_id: str,
        step_id: str,
        name: str,
        data: bytes,
        media_type: str,
    ) -> ArtifactRef:
        """Write artifact bytes to disk."""
        ext = _ext_for_media(media_type)
        dest = self._artifact_path(run_id, step_id, name, ext)
        dest.parent.mkdir(parents=True, exist_ok=True)

        await self._write_bytes(dest, data)

        now = datetime.now(timezone.utc)
        return ArtifactRef(
            run_id=run_id,
            step_id=step_id,
            name=name,
            file_path=dest,
            media_type=media_type,
            url=self._url_for(run_id, step_id, dest.name),
            size_bytes=len(data),
            created_at=now,
        )

    async def store_from_file(
        self,
        run_id: str,
        step_id: str,
        name: str,
        src: Path,
        media_type: str | None = None,
    ) -> ArtifactRef:
        """Move/copy a file into artifact storage."""
        ext = src.suffix.lstrip(".") or "bin"
        dest = self._artifact_path(run_id, step_id, name, ext)
        dest.parent.mkdir(parents=True, exist_ok=True)

        await self._move_file(src, dest)

        if media_type is None:
            media_type = _media_for_ext(ext)

        now = datetime.now(timezone.utc)
        return ArtifactRef(
            run_id=run_id,
            step_id=step_id,
            name=name,
            file_path=dest,
            media_type=media_type,
            url=self._url_for(run_id, step_id, dest.name),
            size_bytes=dest.stat().st_size,
            created_at=now,
        )

    def get_path(self, run_id: str, step_id: str, name: str) -> Path | None:
        """Find artifact file by run/step/name. Returns None if missing."""
        step_dir = self.base_dir / run_id / step_id
        if not step_dir.exists():
            return None
        for f in step_dir.iterdir():
            if f.stem == name:
                return f
        return None

    def url_for(self, run_id: str, step_id: str, name: str) -> str:
        path = self.get_path(run_id, step_id, name)
        filename = path.name if path else f"{name}.bin"
        return self._url_for(run_id, step_id, filename)

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    # ------------------------------------------------------------------
    # Overridable I/O (swap to S3 for multi-node)
    # ------------------------------------------------------------------

    async def _write_bytes(self, dest: Path, data: bytes) -> None:
        dest.write_bytes(data)

    async def _move_file(self, src: Path, dest: Path) -> None:
        shutil.move(str(src), str(dest))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _artifact_path(self, run_id: str, step_id: str, name: str, ext: str) -> Path:
        return self.base_dir / run_id / step_id / f"{name}.{ext}"

    def _url_for(self, run_id: str, step_id: str, filename: str) -> str:
        return f"{self.url_prefix}/runs/{run_id}/artifacts/{step_id}/{filename}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEDIA_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "audio/wav": "wav",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "application/json": "json",
    "application/octet-stream": "bin",
    "model/gltf-binary": "glb",
    "model/gltf+json": "gltf",
    "model/obj": "obj",
    "application/zip": "zip",
}

_EXT_MEDIA = {v: k for k, v in _MEDIA_EXT.items()}


def _ext_for_media(media_type: str) -> str:
    base = media_type.split(";")[0].strip()
    return _MEDIA_EXT.get(base, "bin")


def _media_for_ext(ext: str) -> str:
    return _EXT_MEDIA.get(ext.lower(), "application/octet-stream")
