"""Video scene/shot detection service.

Uses PySceneDetect for content-adaptive shot boundary detection.
No ML model needed — uses algorithmic detection.
"""

import asyncio
import os
import tempfile
from typing import Optional

import httpx
from loguru import logger

from ..settings import get_settings


class SceneService:
    """Video shot/scene detection via PySceneDetect."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def detect_scenes(
        self,
        video_url: str,
        detector: str = "content",
        threshold: float = 27.0,
    ) -> dict:
        """Detect shot boundaries and scene changes in a video."""
        settings = get_settings()
        if not settings.is_enabled("scene"):
            return {"success": False, "error": "Scene detection is disabled"}

        try:
            video_path = await self._download_video(video_url)
            if not video_path:
                return {"success": False, "error": "Failed to download video"}

            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self._detect_sync, video_path, detector, threshold,
                    )
                    return result
            finally:
                os.unlink(video_path)

        except Exception as e:
            logger.error(f"Scene detection error: {e}")
            return {"success": False, "error": f"Scene detection error: {str(e)[:200]}"}

    def _detect_sync(
        self,
        video_path: str,
        detector: str,
        threshold: float,
    ) -> dict:
        from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector

        video = open_video(video_path)
        scene_manager = SceneManager()

        if detector == "adaptive":
            scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
        else:
            scene_manager.add_detector(ContentDetector(threshold=threshold))

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scenes.append({
                "scene": i + 1,
                "start": {
                    "timecode": str(start),
                    "frame": start.get_frames(),
                    "seconds": start.get_seconds(),
                },
                "end": {
                    "timecode": str(end),
                    "frame": end.get_frames(),
                    "seconds": end.get_seconds(),
                },
                "duration_seconds": round(end.get_seconds() - start.get_seconds(), 3),
            })

        return {
            "success": True,
            "scenes": scenes,
            "total_scenes": len(scenes),
            "detector": detector,
        }

    async def _download_video(self, url: str) -> str | None:
        """Download video to temp file."""
        from pathlib import Path

        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "MediaAnalysis/1.0"})
                response.raise_for_status()

            suffix = Path(url).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(response.content)
            tmp.close()
            return tmp.name

        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None

    async def close(self) -> None:
        pass


_scene_service: SceneService | None = None


def get_scene_service() -> SceneService:
    global _scene_service
    if _scene_service is None:
        _scene_service = SceneService()
    return _scene_service
