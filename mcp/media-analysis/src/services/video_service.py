"""Video analysis service.

Extracts keyframes via FFmpeg, checks temporal consistency with SSIM,
delegates frames to vision/tagger services.
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from ..settings import get_settings


def _compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute structural similarity between two images (grayscale, downsampled)."""
    img1 = img1.convert("L").resize((256, 256))
    img2 = img2.convert("L").resize((256, 256))

    a = np.array(img1, dtype=np.float64)
    b = np.array(img2, dtype=np.float64)

    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mean_a) * (b - mean_b)).mean()

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (2 * mean_a * mean_b + c1) * (2 * cov + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)

    return float(numerator / denominator)


class VideoService:
    """Video analysis: frame extraction + SSIM temporal consistency."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def analyze(
        self,
        video_url: str | None = None,
        video_path: str | None = None,
        max_frames: int | None = None,
        ssim_threshold: float | None = None,
    ) -> dict:
        """Extract keyframes and check temporal consistency."""
        settings = get_settings()
        if not settings.is_enabled("video"):
            return {"success": False, "error": "Video analysis is disabled"}

        if max_frames is None:
            max_frames = settings.video_max_frames
        if ssim_threshold is None:
            ssim_threshold = settings.video_ssim_threshold

        # Download or use local file
        if video_url and not video_path:
            video_path = await self._download_video(video_url)
            if not video_path:
                return {"success": False, "error": "Failed to download video"}

        if not video_path or not os.path.exists(video_path):
            return {"success": False, "error": "No video file available"}

        try:
            frames = await self._extract_frames(video_path, max_frames)
            if not frames:
                return {"success": False, "error": "No frames extracted from video"}

            # Compute SSIM between consecutive frames
            ssim_scores = []
            scene_changes = []
            for i in range(1, len(frames)):
                score = _compute_ssim(frames[i - 1], frames[i])
                ssim_scores.append(round(score, 4))
                if score < ssim_threshold:
                    scene_changes.append({
                        "frame": i,
                        "ssim": round(score, 4),
                    })

            avg_ssim = round(sum(ssim_scores) / len(ssim_scores), 4) if ssim_scores else 1.0

            return {
                "success": True,
                "frames_extracted": len(frames),
                "avg_ssim": avg_ssim,
                "ssim_scores": ssim_scores,
                "scene_changes": scene_changes,
                "total_scenes": len(scene_changes) + 1,
            }

        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            return {"success": False, "error": f"Video analysis error: {str(e)[:200]}"}

    async def _download_video(self, url: str) -> str | None:
        """Download video to temp file."""
        import httpx

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

    async def _extract_frames(self, video_path: str, max_frames: int) -> list[Image.Image]:
        """Extract evenly-spaced frames from video via FFmpeg."""
        # Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True,
        )

        if probe.returncode != 0 or not probe.stdout.strip():
            logger.error("ffprobe failed — is FFmpeg installed?")
            return []

        duration = float(probe.stdout.strip())
        if duration <= 0:
            return []

        # Extract frames at evenly spaced timestamps
        frames = []
        interval = duration / (max_frames + 1)

        for i in range(1, max_frames + 1):
            timestamp = interval * i
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

            result = subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
                 "-frames:v", "1", "-y", tmp.name],
                capture_output=True, text=True,
            )

            if result.returncode == 0 and os.path.exists(tmp.name):
                try:
                    img = Image.open(tmp.name)
                    img.load()
                    frames.append(img)
                except Exception:
                    pass
            os.unlink(tmp.name)

        return frames

    async def close(self) -> None:
        logger.info("Video service closed")


_video_service: VideoService | None = None


def get_video_service() -> VideoService:
    global _video_service
    if _video_service is None:
        _video_service = VideoService()
    return _video_service
