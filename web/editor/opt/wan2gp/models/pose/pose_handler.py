"""DWPose — Face detection and keypoint extraction via mediapipe.

Returns face bounding box, keypoints, and cropped face image.
No model download needed — mediapipe is already installed in the image.

Available as MCP tool via the Wan2GP `run` endpoint.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from models.base_handler import BaseFamilyHandler, _make_handler_cls


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["dwpose"]
    FAMILY = "pose"
    FAMILY_INFOS = {"dwpose": (401, "DWPose Face Detection")}
    MODEL_DEF = {"audio_only": False, "image_outputs": True}
    DEFAULTS = {}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        pipeline = _PosePipeline()
        pipe_dict = {}
        return pipeline, pipe_dict


class _PosePipeline:
    """Mediapipe-based face detection + keypoint pipeline."""

    _face_mesh: Any = None

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._face_mesh

    def generate(self, *, image_b64: str = "", **kw) -> dict:
        if not image_b64:
            return {
                "status": "success",
                "data": base64.b64encode(json.dumps({
                    "keypoints": [],
                    "message": "No image provided"
                }).encode()).decode(),
                "media_type": "application/json",
                "face_crop": "",
            }

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        np_img = np.array(img)
        results = self.face_mesh.process(np_img)

        keypoints = []
        face_crops_b64 = []

        if results.multi_face_landmarks:
            h, w = np_img.shape[:2]
            for face_idx, landmarks in enumerate(results.multi_face_landmarks):
                kps = []
                for lm in landmarks.landmark:
                    kps.append({"x": lm.x * w, "y": lm.y * h, "z": (lm.z or 0.0) * w})
                keypoints.append({"face_id": face_idx, "landmarks": kps[:100]})
                xs = [lm.x * w for lm in landmarks.landmark[:20]]
                ys = [lm.y * h for lm in landmarks.landmark[:20]]
                if xs and ys:
                    x_min, x_max = max(0, int(min(xs)) - 20), min(w, int(max(xs)) + 20)
                    y_min, y_max = max(0, int(min(ys)) - 30), min(h, int(max(ys)) + 10)
                    if x_max > x_min and y_max > y_min:
                        crop = img.crop((x_min, y_min, x_max, y_max))
                        buf = io.BytesIO()
                        crop.save(buf, format="PNG")
                        face_crops_b64.append(base64.b64encode(buf.getvalue()).decode())

        result_data = {"keypoints": keypoints, "num_faces": len(keypoints)}
        result = {
            "status": "success",
            "data": base64.b64encode(json.dumps(result_data).encode()).decode(),
            "media_type": "application/json",
        }
        if face_crops_b64:
            result["face_crop"] = face_crops_b64[0]
        return result
