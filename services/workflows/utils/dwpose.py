"""DWPose skeleton extraction — standalone, no ComfyUI dependency.

Detects human poses in images using YOLOX (person detection) + RTMPose
(keypoint estimation) via ONNXRuntime. Models downloaded from HuggingFace
on first use.

Usage:
    from services.workflows.utils.dwpose import skeleton_from_image

    image = cv2.imread("mesh.png")          # RGB/HWC uint8
    skeleton = skeleton_from_image(image)    # white bg + skeleton overlay

    # Or from bytes:
    skeleton_b64 = skeleton_from_image_b64(image_b64, "base64")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── COCO-18 Keypoint Constants ───────────────────────────────────────────────

KEYPOINT_NAMES = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle",
    14: "REye", 15: "LEye", 16: "REar", 17: "LEar",
}

SKELETON_BONES = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),
]

BONE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]

DET_MODEL_REPO = "yzd-v/DWPose"
DET_MODEL_FILE = "yolox_l.onnx"
POSE_MODEL_REPO = "yzd-v/DWPose"
POSE_MODEL_FILE = "dw-ll_ucoco_384.onnx"


# ─── Model Management ─────────────────────────────────────────────────────────

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_models_dir: Path | None = None
_det_model_path: Path | None = None
_pose_model_path: Path | None = None
_detector = None


def set_models_dir(path: str | Path) -> None:
    """Override the model cache directory (default: ~/.cache/tech-noir/dwpose/)."""
    global _models_dir
    _models_dir = Path(path)


def _get_models_dir() -> Path:
    global _models_dir
    if _models_dir is None:
        _models_dir = Path.home() / ".cache" / "tech-noir" / "dwpose"
    return _ensure_dir(_models_dir)


def _download_model(repo_id: str, filename: str) -> Path:
    """Download model from HuggingFace, return local path."""
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=_get_models_dir())
    return Path(local_path)


def _get_det_model() -> Path:
    global _det_model_path
    if _det_model_path is None:
        path = _get_models_dir() / DET_MODEL_FILE
        if not path.exists():
            logger.info("Downloading DWPose detection model from HuggingFace...")
            _download_model(DET_MODEL_REPO, DET_MODEL_FILE)
        _det_model_path = path
    return _det_model_path


def _get_pose_model() -> Path:
    global _pose_model_path
    if _pose_model_path is None:
        path = _get_models_dir() / POSE_MODEL_FILE
        if not path.exists():
            logger.info("Downloading DWPose pose model from HuggingFace...")
            _download_model(POSE_MODEL_REPO, POSE_MODEL_FILE)
        _pose_model_path = path
    return _pose_model_path


# ─── ONNX Inference (adapted from Wan2GP preprocessing) ──────────────────────

class _Wholebody:
    """ONNX person detection + pose estimation pipeline."""

    def __init__(self, device: str = "cpu"):
        import onnxruntime as ort
        det_path = str(_get_det_model())
        pose_path = str(_get_pose_model())
        providers = (["CPUExecutionProvider"]
                     if device == "cpu"
                     else ["CUDAExecutionProvider"])
        self.session_det = ort.InferenceSession(det_path, providers=providers)
        self.session_pose = ort.InferenceSession(pose_path, providers=providers)

    def __call__(self, ori_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = self._inference_detector(ori_img)
        keypoints, scores = self._inference_pose(boxes, ori_img)
        # Insert neck (average of shoulders 5,6 → position 17)
        neck_kp = np.mean(keypoints[:, [5, 6]], axis=1, keepdims=True)
        neck_sc = np.logical_and(scores[:, [5]] > 0.3, scores[:, [6]] > 0.3).astype(float)
        keypoints = np.concatenate([keypoints[:, :17], neck_kp, keypoints[:, 17:]], axis=1)
        scores = np.concatenate([scores[:, :17], neck_sc, scores[:, 17:]], axis=1)
        # Reorder MMPose indices → COCO-18/OpenPose order (affects indices 1-17)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        keypoints[:, openpose_idx] = keypoints[:, mmpose_idx]
        scores[:, openpose_idx] = scores[:, mmpose_idx]
        return keypoints, scores, boxes

    def _inference_detector(self, oriImg: np.ndarray) -> np.ndarray:
        session = self.session_det
        input_shape = (640, 640)
        img, ratio = self._preprocess(oriImg, input_shape)
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = self._demo_postprocess(output[0], input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls = dets[:, :4], dets[:, 4], dets[:, 5]
            mask = (final_scores > 0.3) & (final_cls == 0)
            return final_boxes[mask]
        return np.array([])

    def _inference_pose(self, out_bbox: np.ndarray, oriImg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.session_pose.get_inputs()[0].shape[2]
        w = self.session_pose.get_inputs()[0].shape[3]
        model_input_size = (w, h)
        resized_img, center, scale = self._preprocess_pose(oriImg, out_bbox, model_input_size)
        outputs = self._run_pose(resized_img)
        return self._postprocess_pose(outputs, model_input_size, center, scale)

    @staticmethod
    def _preprocess(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                              interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized
        return padded.transpose(2, 0, 1).astype(np.float32), r

    @staticmethod
    def _demo_postprocess(outputs: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
        grids, strides = [], []
        for s in [8, 16, 32]:
            h, w = img_size[0] // s, img_size[1] // s
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
            strides.append(np.full((1, h * w, 1), s))
        grids = np.concatenate(grids, 1)
        strides = np.concatenate(strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
        return outputs

    @staticmethod
    def _multiclass_nms(boxes, scores, nms_thr, score_thr):
        final = []
        for cls_ind in range(scores.shape[1]):
            mask = scores[:, cls_ind] > score_thr
            if not mask.sum():
                continue
            valid_boxes = boxes[mask]
            valid_scores = scores[mask, cls_ind]
            keep = _nms(valid_boxes, valid_scores, nms_thr)
            if len(keep):
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None],
                                       np.ones((len(keep), 1)) * cls_ind], 1)
                final.append(dets)
        return np.concatenate(final, 0) if final else None

    @staticmethod
    def _get_warp_matrix(center, scale, output_size):
        src_w = scale[0]
        dst_w = float(output_size[0])
        dst_h = float(output_size[1])
        src_dir = np.array([0., src_w * -0.5])
        dst_dir = np.array([0., dst_w * -0.5])
        src = np.zeros((3, 2), dtype=np.float32)
        src[0] = center
        src[1] = center + src_dir
        src[2] = src[0] + np.array([-src_dir[1], src_dir[0]])
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0] = [dst_w * 0.5, dst_h * 0.5]
        dst[1] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
        dst[2] = dst[0] + np.array([-dst_dir[1], dst_dir[0]])
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))

    @staticmethod
    def _preprocess_pose(img: np.ndarray, out_bbox: np.ndarray,
                          input_size: tuple[int, int]) -> tuple:
        if len(out_bbox) == 0:
            out_bbox = [[0, 0, img.shape[1], img.shape[0]]]
        out_img, out_center, out_scale = [], [], []
        w, h = input_size
        aspect = w / h
        for i in range(len(out_bbox)):
            x0, y0, x1, y1 = out_bbox[i]
            center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            scale = np.array([x1 - x0, y1 - y0]) * 1.25
            if scale[0] > scale[1] * aspect:
                scale[1] = scale[0] / aspect
            else:
                scale[0] = scale[1] * aspect
            warp_mat = _Wholebody._get_warp_matrix(center, scale, (w, h))
            resized = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            resized = ((resized - mean) / std).astype(np.float32)
            out_img.append(resized)
            out_center.append(center)
            out_scale.append(scale)
        return out_img, out_center, out_scale

    def _run_pose(self, resized_img: list) -> list:
        sess = self.session_pose
        all_out = []
        for img in resized_img:
            inp = {sess.get_inputs()[0].name: img.transpose(2, 0, 1)[None]}
            out_names = [o.name for o in sess.get_outputs()]
            all_out.append(sess.run(out_names, inp))
        return all_out

    @staticmethod
    def _postprocess_pose(outputs, model_input_size, center, scale):
        all_key, all_score = [], []
        for i, out in enumerate(outputs):
            simcc_x, simcc_y = out
            N, K, Wx = simcc_x.shape
            x_locs = np.argmax(simcc_x.reshape(N * K, -1), axis=1)
            y_locs = np.argmax(simcc_y.reshape(N * K, -1), axis=1)
            max_x = np.amax(simcc_x.reshape(N * K, -1), axis=1)
            max_y = np.amax(simcc_y.reshape(N * K, -1), axis=1)
            vals = np.where(max_x > max_y, max_y, max_x)
            locs = np.stack([x_locs, y_locs], -1).astype(float)
            locs[vals <= 0] = -1
            locs = locs.reshape(N, K, 2) / 2.0
            vals = vals.reshape(N, K)
            locs = locs / model_input_size * scale[i] + center[i] - scale[i] / 2
            all_key.append(locs[0])
            all_score.append(vals[0])
        return np.array(all_key), np.array(all_score)


def _nms(boxes, scores, nms_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= nms_thr)[0] + 1]
    return keep


# ─── Skeleton Rendering ───────────────────────────────────────────────────────

def _draw_skeleton(canvas: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                   threshold: float = 0.3) -> np.ndarray:
    """Draw COCO-18 skeleton keypoints and bones onto a canvas."""
    h, w = canvas.shape[:2]
    for bone_idx, (i, j) in enumerate(SKELETON_BONES):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if scores[i] > threshold and scores[j] > threshold:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            color = BONE_COLORS[bone_idx % len(BONE_COLORS)]
            cv2.line(canvas, pt1, pt2, color, thickness=max(2, w // 256))
    for i in range(len(keypoints)):
        if scores[i] > threshold:
            center = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            cv2.circle(canvas, center, radius=max(3, w // 128),
                       color=(0, 0, 255), thickness=-1)
    return canvas


# ─── Public API ───────────────────────────────────────────────────────────────

_detector_instance: _Wholebody | None = None


def _get_detector() -> _Wholebody:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = _Wholebody(device="cpu")
    return _detector_instance


def detect_poses(image: np.ndarray) -> np.ndarray:
    """Detect human poses in an image and return keypoints.

    Args:
        image: (H, W, 3) uint8 numpy array (RGB or BGR).

    Returns:
        (N, 18, 2) array of keypoints per detected person.
    """
    detector = _get_detector()
    keypoints, scores, _ = detector(image)
    return keypoints


def skeleton_from_image(image: np.ndarray,
                        output_width: int = 1024,
                        output_height: int = 1024) -> np.ndarray:
    """Detect pose in an image and render skeleton overlay on white background.

    Args:
        image: (H, W, 3) uint8 numpy array (RGB or BGR).
        output_width, output_height: Output image dimensions.

    Returns:
        (output_height, output_width, 3) uint8 — white bg + skeleton overlay.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    detector = _get_detector()
    keypoints, scores, _ = detector(image)

    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)

    if keypoints is not None and len(keypoints) > 0:
        canvas = _draw_skeleton(canvas, keypoints[0], scores[0])

    return canvas


def skeleton_from_image_b64(image_b64: str,
                            output_width: int = 1024,
                            output_height: int = 1024) -> str:
    """Like skeleton_from_image but base64 in/out.

    Args:
        image_b64: Base64-encoded image bytes.
        output_width, output_height: Output image dimensions.

    Returns:
        Base64-encoded PNG of skeleton overlay.
    """
    import base64, io
    from PIL import Image
    image_bytes = base64.b64decode(image_b64)
    buf = io.BytesIO(image_bytes)
    pil = Image.open(buf).convert("RGB")
    np_img = np.array(pil)

    skeleton = skeleton_from_image(np_img, output_width, output_height)

    result_pil = Image.fromarray(skeleton)
    out_buf = io.BytesIO()
    result_pil.save(out_buf, format="PNG")
    return base64.b64encode(out_buf.getvalue()).decode()
