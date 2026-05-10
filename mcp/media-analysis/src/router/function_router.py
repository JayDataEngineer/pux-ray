"""FunctionGemma router for intelligent tool selection.

Loads a GGUF model via llama-cpp-python. Routes user queries to the
appropriate tool by generating structured function calls.
"""

import json
import re
import time
from typing import Optional

from loguru import logger

from ..settings import get_settings


# Tool definitions for FunctionGemma's function calling
TOOL_SCHEMAS = [
    {
        "name": "analyze_image",
        "description": "Analyze an image: caption, OCR, object detection, dense region captions",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
            "task": {"type": "string", "description": "One of: caption, detailed_caption, more_detailed_caption, ocr, object_detection"},
        },
    },
    {
        "name": "detect_objects",
        "description": "Detect and locate objects in an image with bounding boxes and labels",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
            "confidence": {"type": "float", "description": "Minimum confidence threshold (0-1)"},
        },
    },
    {
        "name": "tag_image",
        "description": "Tag an image with content labels and categories",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
        },
    },
    {
        "name": "extract_colors",
        "description": "Extract dominant colors and color palette from an image",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
            "color_count": {"type": "integer", "description": "Number of colors to extract (2-20, default 5)"},
        },
    },
    {
        "name": "read_barcodes",
        "description": "Read QR codes and barcodes from an image",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
        },
    },
    {
        "name": "extract_exif",
        "description": "Extract EXIF metadata from an image (camera, GPS, dates, etc.)",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
        },
    },
    {
        "name": "detect_faces",
        "description": "Detect and recognize faces in an image with bounding boxes and landmarks",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
        },
    },
    {
        "name": "classify_nsfw",
        "description": "Classify image for NSFW content (safe, questionable, unsafe)",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
        },
    },
    {
        "name": "segment_image",
        "description": "Segment an image into object masks using SAM 2",
        "parameters": {
            "image_url": {"type": "string", "description": "URL of the image"},
            "mode": {"type": "string", "description": "Segmentation mode: auto, points, or box"},
        },
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe speech from an audio file to text",
        "parameters": {
            "audio_url": {"type": "string", "description": "URL of the audio file"},
        },
    },
    {
        "name": "classify_audio",
        "description": "Classify audio events and sound types (speech, music, environment)",
        "parameters": {
            "audio_url": {"type": "string", "description": "URL of the audio file"},
        },
    },
    {
        "name": "fingerprint_audio",
        "description": "Generate an audio fingerprint for identification via Chromaprint",
        "parameters": {
            "audio_url": {"type": "string", "description": "URL of the audio file"},
        },
    },
    {
        "name": "diarize_audio",
        "description": "Identify who speaks when in audio (speaker diarization)",
        "parameters": {
            "audio_url": {"type": "string", "description": "URL of the audio file"},
        },
    },
    {
        "name": "check_video",
        "description": "Analyze a video: extract keyframes, detect scene changes, temporal consistency",
        "parameters": {
            "video_url": {"type": "string", "description": "URL of the video file"},
        },
    },
    {
        "name": "detect_scenes",
        "description": "Detect shot boundaries and scene changes in a video using content-adaptive detection",
        "parameters": {
            "video_url": {"type": "string", "description": "URL of the video file"},
            "detector": {"type": "string", "description": "Detector: content or adaptive"},
        },
    },
]


def _build_prompt(query: str, media_url: str) -> str:
    """Build the routing prompt for FunctionGemma."""
    tools_str = json.dumps(TOOL_SCHEMAS, indent=2)
    return (
        f"You are a tool router. Given a user query and a media URL, "
        f"select the best tool and extract parameters.\n\n"
        f"Available tools:\n{tools_str}\n\n"
        f"User query: {query}\n"
        f"Media URL: {media_url}\n\n"
        f"Respond with ONLY a JSON object: "
        f'{{"tool": "<tool_name>", "params": {{...}}}}'
    )


class FunctionRouter:
    """FunctionGemma GGUF router via llama-cpp-python."""

    _instance = None

    def __init__(self):
        self._llm = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        """Load model synchronously at startup."""
        if self._loaded:
            if self._load_error:
                raise RuntimeError(f"Router model failed to load: {self._load_error}")
            return

        settings = get_settings()
        if not settings.router_enabled:
            raise RuntimeError("Router is disabled")

        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading router model: {settings.router_model}")
            start = time.time()

            model_path = hf_hub_download(
                settings.router_model,
                filename=settings.router_filename,
                cache_dir=settings.model_cache_dir,
            )

            self._llm = Llama(
                model_path=model_path,
                n_ctx=settings.router_n_ctx,
                n_threads=settings.router_n_threads,
                verbose=False,
            )

            elapsed = time.time() - start
            logger.info(f"Router model loaded in {elapsed:.1f}s")
            self._loaded = True

        except Exception as e:
            self._load_error = str(e)
            self._loaded = True
            logger.error(f"Failed to load router model: {e}")
            raise

    def route(self, query: str, media_url: str) -> dict:
        """Route a query to the appropriate tool.

        Returns dict with 'tool' and 'params' keys, or falls back to
        rule-based routing if FunctionGemma fails.
        """
        if not self._loaded or self._llm is None:
            return self._fallback_route(query, media_url)

        try:
            prompt = _build_prompt(query, media_url)
            settings = get_settings()

            response = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            text = response["choices"][0]["message"]["content"].strip()

            # Extract JSON from response
            match = re.search(r'\{[^{}]+\}', text)
            if match:
                result = json.loads(match.group())
                if "tool" in result:
                    return result

            return self._fallback_route(query, media_url)

        except Exception as e:
            logger.warning(f"Router inference failed, using fallback: {e}")
            return self._fallback_route(query, media_url)

    def _fallback_route(self, query: str, media_url: str) -> dict:
        """Rule-based fallback routing based on URL extension and query keywords."""
        url_lower = media_url.lower()
        query_lower = query.lower()

        # Route by media type (URL extension)
        audio_exts = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma")
        video_exts = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv")
        image_exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff")

        # Image-specific keywords (check before generic type routing)
        if any(kw in query_lower for kw in ("color", "palette", "dominant color", "color scheme")):
            return {"tool": "extract_colors", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("qr", "barcode", "qr code", "scan code")):
            return {"tool": "read_barcodes", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("exif", "metadata", "camera info", "gps", "date taken")):
            return {"tool": "extract_exif", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("face", "faces", "facial", "who is", "face detect")):
            return {"tool": "detect_faces", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("nsfw", "nsfl", "safe for work", "content warning", "nude")):
            return {"tool": "classify_nsfw", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("segment", "mask", "silhouette", "cut out")):
            return {"tool": "segment_image", "params": {"image_url": media_url}}

        # Audio-specific keywords
        if any(kw in query_lower for kw in ("sound", "audio classify", "what sound", "audio event", "audio type")):
            return {"tool": "classify_audio", "params": {"audio_url": media_url}}

        if any(kw in query_lower for kw in ("fingerprint", "identify audio", "chromaprint", "audio id")):
            return {"tool": "fingerprint_audio", "params": {"audio_url": media_url}}

        if any(kw in query_lower for kw in ("speaker", "diarize", "who speaking", "speakers", "who is talking")):
            return {"tool": "diarize_audio", "params": {"audio_url": media_url}}

        # Video-specific keywords
        if any(kw in query_lower for kw in ("scene", "shot", "cut point", "scene change", "shot detect")):
            return {"tool": "detect_scenes", "params": {"video_url": media_url}}

        # Generic type routing by URL extension
        if url_lower.endswith(audio_exts):
            return {"tool": "transcribe_audio", "params": {"audio_url": media_url}}

        if url_lower.endswith(video_exts):
            return {"tool": "check_video", "params": {"video_url": media_url}}

        # Existing image keyword routing
        if any(kw in query_lower for kw in ("detect", "object", "bounding box", "locate")):
            return {"tool": "detect_objects", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("tag", "label", "category")):
            return {"tool": "tag_image", "params": {"image_url": media_url}}

        if any(kw in query_lower for kw in ("ocr", "read text")):
            return {
                "tool": "analyze_image",
                "params": {"image_url": media_url, "task": "ocr"},
            }

        # Default: detailed image analysis
        if url_lower.endswith(image_exts) or "image" in query_lower or "photo" in query_lower:
            return {"tool": "analyze_image", "params": {"image_url": media_url}}

        # Ultimate fallback
        return {"tool": "analyze_image", "params": {"image_url": media_url}}

    def close(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded = False
            logger.info("Router model unloaded")


_router: FunctionRouter | None = None


def get_router() -> FunctionRouter:
    global _router
    if _router is None:
        _router = FunctionRouter()
    return _router
