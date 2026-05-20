"""Face detection and recognition service.

Uses InsightFace ONNX models (buffalo_l) for face detection,
alignment, and embedding extraction. Lazy-loads on first request.
"""

import asyncio
import time
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from ..settings import get_settings, get_device
from .media_utils import load_image


class FaceService:
    """InsightFace face detection + recognition via ONNX."""

    def __init__(self):
        self._app = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Face model failed to load: {self._load_error}")
            return

        async with self._lock:
            if self._loaded:
                if self._load_error:
                    raise RuntimeError(f"Face model failed to load: {self._load_error}")
                return

            settings = get_settings()
            if not settings.is_enabled("face"):
                raise RuntimeError("Face detection is disabled")

            try:
                logger.info(f"Loading face model: {settings.face_model}")
                start = time.time()

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync)

                elapsed = time.time() - start
                logger.info(f"Face model loaded in {elapsed:.1f}s")
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                self._loaded = True
                logger.error(f"Failed to load face model: {e}")
                raise

    def _load_model_sync(self) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        settings = get_settings()
        device = get_device()
        ctx_id = 0 if device == "cuda" else -1

        self._app = FaceAnalysis(
            name=settings.face_model,
            root=settings.model_cache_dir,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            ),
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    async def detect_faces(
        self,
        image_url: str,
        max_faces: int = 10,
    ) -> dict:
        """Detect and recognize faces in an image."""
        await self._ensure_loaded()

        try:
            image = await load_image(image_url)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)[:200]}"}

        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._detect_sync, image, max_faces,
                )
                return result

            except Exception as e:
                logger.error(f"Face detection error: {e}")
                return {"success": False, "error": f"Detection error: {str(e)[:200]}"}

    def _detect_sync(self, image: Image.Image, max_faces: int) -> dict:
        import cv2

        # Convert PIL to cv2 format (BGR)
        img_array = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        faces = self._app.get(img_bgr)

        # Limit results
        faces = faces[:max_faces]

        results = []
        for face in faces:
            face_info = {
                "bbox": face.bbox.tolist() if face.bbox is not None else None,
                "confidence": float(face.det_score) if face.det_score is not None else None,
            }

            # Add landmarks if available
            if face.kps is not None:
                face_info["landmarks"] = {
                    "left_eye": face.kps[0].tolist(),
                    "right_eye": face.kps[1].tolist(),
                    "nose": face.kps[2].tolist(),
                    "left_mouth": face.kps[3].tolist(),
                    "right_mouth": face.kps[4].tolist(),
                }

            # Add embedding (normalized 512-d vector)
            if face.embedding is not None:
                face_info["embedding_dim"] = len(face.embedding)
                # Don't return full embedding in response (too large)
                face_info["has_embedding"] = True

            results.append(face_info)

        return {
            "success": True,
            "faces": results,
            "count": len(results),
        }

    async def close(self) -> None:
        if self._app is not None:
            del self._app
            self._app = None
            self._loaded = False
            logger.info("Face model unloaded")


_face_service: FaceService | None = None


def get_face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service
