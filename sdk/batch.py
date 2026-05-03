"""Batch generation helper — groups jobs by model type to avoid VRAM thrashing.

Submits all same-type jobs together so the GPU loads each model only once,
waits for the group to finish, then moves on. Prevents the costly
load/unload cycle when mixing job types on a single GPU.

Usage:
    from sdk.batch import BatchBuilder

    results = await (
        BatchBuilder()
        .add("ace_step", prompt="chill synthwave", duration=30)
        .add("ace_step", prompt="dark ambient drone", duration=30)
        .add("trellis", image=image_bytes)
        .run()
    )
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdk.client import RayClient

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3


@dataclass
class JobResult:
    job_id: str
    job_type: str
    params: dict[str, Any]
    status: str = "pending"
    data: bytes | None = None
    error: str | None = None


class BatchBuilder:
    """Accumulate jobs and run them grouped by type."""

    def __init__(self, base_url: str = "http://localhost:18080"):
        self._base_url = base_url
        self._items: list[tuple[str, dict[str, Any]]] = []

    def add(self, job_type: str, **kwargs) -> BatchBuilder:
        self._items.append((job_type, kwargs))
        return self

    async def run(self, output_dir: str | Path | None = None) -> list[JobResult]:
        """Submit jobs grouped by type, wait for each group, return results.

        If output_dir is given, saves binary results to files named
        {job_type}_{job_id}.{ext}.
        """
        if not self._items:
            return []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results: list[JobResult] = []

        # Group by type to minimize model swaps
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, (job_type, params) in enumerate(self._items):
            groups[job_type].append((idx, params))

        # Preserve insertion order, deduplicated by type
        type_order = list(dict.fromkeys(t for t, _ in self._items))

        async with RayClient(base_url=self._base_url, timeout=600) as client:
            for job_type in type_order:
                group = groups[job_type]
                logger.info("Submitting %d %s job(s)", len(group), job_type)

                # Submit all jobs in this group
                job_ids: list[tuple[int, str]] = []
                for orig_idx, params in group:
                    job_id = await client.submit_job(job_type, **params)
                    results.append(JobResult(
                        job_id=job_id, job_type=job_type, params=params,
                    ))
                    job_ids.append((orig_idx, job_id))
                    logger.info("  %s -> %s", job_type, job_id)

                # Poll until all jobs in this group finish
                pending = {job_id for _, job_id in job_ids}
                while pending:
                    for job_id in list(pending):
                        status = await client.job_status(job_id)
                        if status["status"] in ("completed", "error"):
                            pending.discard(job_id)
                            r = next(r for r in results if r.job_id == job_id)
                            r.status = status["status"]
                            if status["status"] == "error":
                                r.error = status.get("error", "unknown error")
                                logger.error("  %s failed: %s", job_id, r.error)
                            else:
                                try:
                                    r.data = await client.job_result(job_id)
                                except Exception as e:
                                    r.status = "error"
                                    r.error = str(e)
                                    logger.error("  %s result fetch failed: %s", job_id, e)

                                if output_dir and r.data:
                                    ext = _ext_for_type(job_type, r.data)
                                    path = output_dir / f"{job_type}_{job_id}{ext}"
                                    path.write_bytes(r.data)
                                    logger.info("  %s saved to %s", job_id, path.name)

                    if pending:
                        await asyncio.sleep(_POLL_INTERVAL)

                logger.info("%s group done (%d jobs)", job_type, len(group))

        return results


def _ext_for_type(job_type: str, data: bytes) -> str:
    """Guess file extension from job type and data."""
    if job_type == "ace_step":
        return ".wav"
    if job_type in ("trellis", "anigen"):
        return ".glb"
    if job_type == "comfyui":
        return ".png"
    # Fallback: sniff magic bytes
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"glTF" or data[:4] == b"\x00\x00\x00\x2c":
        return ".glb"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".bin"
