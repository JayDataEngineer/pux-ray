"""MCP tools for media analysis.

Provides both:
- process(query, media_url): single tool routed by FunctionGemma
- Direct tool access for callers who know what they want
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Annotated

import httpx
from fastmcp import FastMCP, Context
from loguru import logger
from pydantic import Field


# ============================================================
# Upload Tool
# ============================================================

async def upload(
    data: Annotated[str, Field(
        description="Base64-encoded image/audio/video data"
    )],
    mime_type: Annotated[str, Field(
        description="MIME type of the data (e.g. image/png, audio/wav, video/mp4)"
    )] = "image/png",
    ctx: Context | None = None,
) -> dict:
    """Upload base64-encoded media data and get a temporary URL.

    Use this when you have raw base64 data (e.g. from a screenshot or recording)
    instead of a remote URL. Returns a URL you can pass to any tool's imageSource/audioSource/videoSource parameter.
    Files are automatically cleaned up after 1 hour.
    """
    from ..services.media_utils import save_upload, UPLOAD_DIR
    from ..settings import get_settings

    try:
        filename = save_upload(data, mime_type)
        settings = get_settings()
        url = f"http://localhost:{settings.port}/tmp/{filename}"
        return {"success": True, "url": url, "filename": filename}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)[:200]}"}


# ============================================================
# Image Tools
# ============================================================

async def analyze_image(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS), or data URI (data:image/png;base64,...). Supports PNG, JPG, JPEG."
    )],
    prompt: Annotated[str, Field(
        description="Detailed text prompt describing what to analyze in the image"
    )],
    task: Annotated[str | None, Field(
        description="Florence-2 task: caption, detailed_caption, more_detailed_caption, ocr, object_detection. Default: more_detailed_caption"
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Analyze an image using Florence-2 vision model.

    Supports captions, OCR, object detection, and dense region captions.
    """
    from ..services.vision_service import get_vision_service

    task_map = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "ocr": "<OCR>",
        "object_detection": "<OD>",
    }

    florence_task = task_map.get(task, "<MORE_DETAILED_CAPTION>") if task else "<MORE_DETAILED_CAPTION>"

    service = get_vision_service()
    result = await service.analyze(
        image_url=imageSource,
        task=florence_task,
        text_input=prompt if florence_task == "<CAPTION_TO_PHRASE_GROUNDING>" else None,
    )

    return result


async def detect_objects(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI). Supports PNG, JPG, JPEG."
    )],
    confidence: Annotated[float, Field(
        description="Minimum confidence threshold (0-1, default 0.25)",
        ge=0.0,
        le=1.0,
    )] = 0.25,
    ctx: Context | None = None,
) -> dict:
    """Detect and locate objects in an image with bounding boxes using YOLOv8."""
    from ..services.yolo_service import get_yolo_service

    service = get_yolo_service()
    return await service.detect(image_url=imageSource, confidence=confidence)


async def tag_image(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    threshold: Annotated[float, Field(
        description="Minimum confidence threshold for tags (0-1, default 0.35)",
        ge=0.0,
        le=1.0,
    )] = 0.35,
    ctx: Context | None = None,
) -> dict:
    """Tag an image with content labels and categories using WD14 tagger."""
    from ..services.tagger_service import get_tagger_service

    service = get_tagger_service()
    return await service.tag(image_url=imageSource, threshold=threshold)


async def extract_colors(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    color_count: Annotated[int, Field(
        description="Number of colors to extract (2-20, default 5)",
        ge=2,
        le=20,
    )] = 5,
    ctx: Context | None = None,
) -> dict:
    """Extract dominant colors and a color palette from an image.

    Uses ColorThief for fast palette extraction — no ML model needed.
    Returns the dominant color and a palette of RGB values.
    """
    from ..services.color_service import get_color_service

    service = get_color_service()
    return await service.extract_colors(image_url=imageSource, color_count=color_count)


async def read_barcodes(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    ctx: Context | None = None,
) -> dict:
    """Read QR codes and barcodes from an image.

    Detects QR codes, EAN, UPC, Code128, and other barcode formats.
    Returns the decoded data and bounding rectangles for each code found.
    """
    from ..services.barcode_service import get_barcode_service

    service = get_barcode_service()
    return await service.read_barcodes(image_url=imageSource)


async def extract_exif(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    ctx: Context | None = None,
) -> dict:
    """Extract EXIF metadata from an image.

    Returns camera info, GPS coordinates, timestamps, lens data,
    and other metadata embedded in the image file.
    """
    from ..services.exif_service import get_exif_service

    service = get_exif_service()
    return await service.extract_exif(image_url=imageSource)


async def detect_faces(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    max_faces: Annotated[int, Field(
        description="Maximum number of faces to return (default 10)",
        ge=1,
        le=50,
    )] = 10,
    ctx: Context | None = None,
) -> dict:
    """Detect and recognize faces in an image using InsightFace.

    Returns face bounding boxes, landmarks (eyes, nose, mouth),
    confidence scores, and embedding availability for each face.
    """
    from ..services.face_service import get_face_service

    service = get_face_service()
    return await service.detect_faces(image_url=imageSource, max_faces=max_faces)


async def classify_nsfw(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    threshold: Annotated[float, Field(
        description="NSFW threshold (0-1, default 0.5)",
        ge=0.0,
        le=1.0,
    )] = 0.5,
    ctx: Context | None = None,
) -> dict:
    """Classify an image for NSFW content using NudeNet.

    Returns an NSFW score and per-class probabilities (safe, questionable, unsafe).
    Useful for content moderation pipelines.
    """
    from ..services.nsfw_service import get_nsfw_service

    service = get_nsfw_service()
    return await service.classify_nsfw(image_url=imageSource, threshold=threshold)


async def segment_image(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI)"
    )],
    mode: Annotated[str, Field(
        description="Segmentation mode: auto (grid points), points (explicit coords), or box (bounding box)"
    )] = "auto",
    points: Annotated[list[list[float]] | None, Field(
        description="Point prompts [[x,y], ...] for points mode"
    )] = None,
    point_labels: Annotated[list[int] | None, Field(
        description="Point labels: 1=foreground, 0=background"
    )] = None,
    box: Annotated[list[float] | None, Field(
        description="Box prompt [x1,y1,x2,y2] for box mode"
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Segment an image into object masks using SAM 2.

    Three modes:
    - auto: Generates a grid of points across the image and produces all masks
    - points: Uses explicit (x,y) coordinates as prompts
    - box: Uses a bounding box [x1,y1,x2,y2] as prompt (works well with detect_objects output)
    """
    from ..services.segment_service import get_segment_service

    service = get_segment_service()
    return await service.segment(
        image_url=imageSource,
        mode=mode,
        points=points,
        point_labels=point_labels,
        box=box,
    )


# ============================================================
# Audio Tools
# ============================================================

async def transcribe_audio(
    audioSource: Annotated[str, Field(
        description="Remote URL to the audio file (supports MP3, WAV, FLAC, OGG)"
    )],
    ctx: Context | None = None,
) -> dict:
    """Transcribe speech from an audio file to text using Parakeet TDT v3."""
    from ..services.asr_service import get_asr_service

    # Download audio and convert to WAV (onnx-asr requires WAV)
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(audioSource, headers={"User-Agent": "MediaAnalysis/1.0"})
            response.raise_for_status()

        # Write raw download to temp file
        suffix = "." + audioSource.rsplit(".", 1)[-1] if "." in audioSource else ".wav"
        raw = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        raw.write(response.content)
        raw.close()

        # Convert to WAV via ffmpeg
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
            capture_output=True, check=True,
        )
        os.unlink(raw.name)

        service = get_asr_service()
        result = await service.transcribe(wav.name)

        os.unlink(wav.name)
        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to process audio: {str(e)[:200]}"}


async def classify_audio(
    audioSource: Annotated[str, Field(
        description="Remote URL to the audio file"
    )],
    top_k: Annotated[int, Field(
        description="Number of top labels to return (default 10)",
        ge=1,
        le=50,
    )] = 10,
    ctx: Context | None = None,
) -> dict:
    """Classify audio events and sound types using PANNs.

    Detects speech, music, environmental sounds, and other audio events
    based on the AudioSet 527-class taxonomy.
    """
    from ..services.audio_classify_service import get_audio_classify_service

    service = get_audio_classify_service()
    return await service.classify_audio(audio_url=audioSource, top_k=top_k)


async def fingerprint_audio(
    audioSource: Annotated[str, Field(
        description="Remote URL to the audio file"
    )],
    ctx: Context | None = None,
) -> dict:
    """Generate an audio fingerprint for identification via Chromaprint.

    Returns a fingerprint hash and duration. Use for audio identification
    and duplicate detection.
    """
    from ..services.fingerprint_service import get_fingerprint_service

    service = get_fingerprint_service()
    return await service.fingerprint_audio(audio_url=audioSource)


async def diarize_audio(
    audioSource: Annotated[str, Field(
        description="Remote URL to the audio file"
    )],
    num_speakers: Annotated[int | None, Field(
        description="Number of speakers (null=auto-detect)",
        ge=1,
        le=20,
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Perform speaker diarization on audio using Pyannote 3.1.

    Identifies who speaks when. Returns timestamped segments with speaker labels.
    Requires MEDIA_PYANNOTE_ENABLED=true and MEDIA_PYANNOTE_TOKEN to be set.
    """
    from ..services.diarize_service import get_diarize_service

    service = get_diarize_service()
    return await service.diarize(audio_url=audioSource, num_speakers=num_speakers)


# ============================================================
# Video Tools
# ============================================================

async def check_video(
    videoSource: Annotated[str, Field(
        description="Remote URL to the video file (supports MP4, AVI, MKV, MOV, WEBM)"
    )],
    max_frames: Annotated[int, Field(
        description="Maximum number of frames to extract (default 10)",
        ge=1,
        le=50,
    )] = 10,
    ctx: Context | None = None,
) -> dict:
    """Analyze a video: extract keyframes, detect scene changes, check temporal consistency."""
    from ..services.video_service import get_video_service

    service = get_video_service()
    return await service.analyze(video_url=videoSource, max_frames=max_frames)


async def detect_scenes(
    videoSource: Annotated[str, Field(
        description="Remote URL to the video file"
    )],
    detector: Annotated[str, Field(
        description="Detector type: content (default) or adaptive"
    )] = "content",
    threshold: Annotated[float, Field(
        description="Detection threshold (default 27.0 for content detector)"
    )] = 27.0,
    ctx: Context | None = None,
) -> dict:
    """Detect shot boundaries and scene changes in a video using PySceneDetect.

    Uses content-adaptive detection algorithms to find precise cut points
    between scenes. Returns timestamps, frame numbers, and durations.
    """
    from ..services.scene_service import get_scene_service

    service = get_scene_service()
    return await service.detect_scenes(video_url=videoSource, detector=detector, threshold=threshold)


# ============================================================
# Microsoft Vision Tools
# ============================================================

async def detect_objects_text(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI). Supports PNG, JPG, JPEG."
    )],
    prompt: Annotated[str, Field(
        description="Text description of objects to detect, separated by periods (e.g. 'a cat . a remote control . a dog')"
    )],
    confidence: Annotated[float, Field(
        description="Minimum confidence threshold (0-1, default 0.35)",
        ge=0.0,
        le=1.0,
    )] = 0.35,
    ctx: Context | None = None,
) -> dict:
    """Detect and locate objects in an image using text prompts via Grounding DINO.

    Unlike YOLOv8 (fixed 80-class detection), Grounding DINO finds ANY object you describe.
    Separate multiple objects with periods: 'a cat . a dog . a coffee mug'
    Returns bounding boxes, confidence scores, and labels.
    """
    from ..services.grounding_dino_service import get_grounding_dino_service

    service = get_grounding_dino_service()
    return await service.detect(
        image_url=imageSource,
        text_prompt=prompt,
        threshold=confidence,
    )


async def phi4_vision(
    imageSource: Annotated[str, Field(
        description="URL to the image (HTTP/HTTPS or data URI). Supports PNG, JPG, JPEG."
    )],
    prompt: Annotated[str, Field(
        description="What to ask about the image (e.g. 'What is shown in this image?', 'Read the text on the sign')"
    )],
    max_tokens: Annotated[int, Field(
        description="Maximum tokens to generate (default 2048)",
        ge=1,
        le=8192,
    )] = 2048,
    ctx: Context | None = None,
) -> dict:
    """Analyze an image using advanced AI vision models with comprehensive understanding capabilities.

    Uses Gemma 4 E4B (GGUF, IQ4_NL) for visual reasoning — describe, reason, and answer questions about images.
    More capable than Florence-2 for complex reasoning tasks, chart analysis, and multi-step visual understanding.
    """
    from ..services.phi4_vision_service import get_phi4_vision_service

    service = get_phi4_vision_service()
    return await service.chat(
        image_url=imageSource,
        prompt=prompt,
        max_new_tokens=max_tokens,
    )


async def kosmos_ocr(
    imageSource: Annotated[str, Field(
        description="URL to the document image (HTTP/HTTPS or data URI). Supports PNG, JPG, JPEG."
    )],
    mode: Annotated[str, Field(
        description="Output mode: 'markdown' for formatted text (default), 'ocr' for text with bounding box coordinates"
    )] = "markdown",
    max_tokens: Annotated[int, Field(
        description="Maximum tokens to generate (default 4096)",
        ge=1,
        le=16384,
    )] = 4096,
    ctx: Context | None = None,
) -> dict:
    """Extract text from document images using Kosmos-2.5.

    Converts document images (receipts, forms, invoices, screenshots) to structured text.
    Use 'markdown' mode for clean formatted output, 'ocr' mode for text with spatial coordinates.
    """
    from ..services.kosmos_service import get_kosmos_service

    service = get_kosmos_service()
    return await service.convert(
        image_url=imageSource,
        mode=mode,
        max_new_tokens=max_tokens,
    )


# ============================================================
# Smart Router (process)
# ============================================================

async def process(
    query: Annotated[str, Field(
        description="What you want to do with the media (e.g., 'describe this image', 'transcribe this audio')"
    )],
    media_url: Annotated[str, Field(
        description="URL to the media file (HTTP/HTTPS or data URI for images)"
    )],
    ctx: Context | None = None,
) -> dict:
    """Process media using the best tool for the job.

    Routes your query to the right model automatically:
    - Images → analyze, detect objects, tag, extract colors, read barcodes, EXIF, faces, NSFW, segment
    - Audio → transcribe, classify events, fingerprint, diarize
    - Video → analyze keyframes, detect scenes

    You can also call the individual tools directly if you know which one you need.
    """
    from ..router.function_router import get_router
    from ..services.vision_service import get_vision_service
    from ..services.yolo_service import get_yolo_service
    from ..services.tagger_service import get_tagger_service
    from ..services.asr_service import get_asr_service
    from ..services.video_service import get_video_service
    from ..services.color_service import get_color_service
    from ..services.barcode_service import get_barcode_service
    from ..services.exif_service import get_exif_service
    from ..services.scene_service import get_scene_service
    from ..services.fingerprint_service import get_fingerprint_service
    from ..services.face_service import get_face_service
    from ..services.nsfw_service import get_nsfw_service
    from ..services.audio_classify_service import get_audio_classify_service
    from ..services.segment_service import get_segment_service
    from ..services.diarize_service import get_diarize_service

    router = get_router()
    route = router.route(query, media_url)

    tool_name = route.get("tool", "analyze_image")
    params = route.get("params", {})

    logger.info(f"Router: '{query}' → {tool_name}")

    try:
        if tool_name == "analyze_image":
            service = get_vision_service()
            return await service.analyze(
                image_url=params.get("image_url", media_url),
                task=params.get("task", "<MORE_DETAILED_CAPTION>"),
            )

        elif tool_name == "detect_objects":
            service = get_yolo_service()
            return await service.detect(
                image_url=params.get("image_url", media_url),
                confidence=params.get("confidence", 0.25),
            )

        elif tool_name == "tag_image":
            service = get_tagger_service()
            return await service.tag(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "transcribe_audio":
            # Download audio and convert to WAV
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(media_url, headers={"User-Agent": "MediaAnalysis/1.0"})
                response.raise_for_status()

            suffix = "." + media_url.rsplit(".", 1)[-1] if "." in media_url else ".wav"
            raw = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            raw.write(response.content)
            raw.close()

            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
                capture_output=True, check=True,
            )
            os.unlink(raw.name)

            service = get_asr_service()
            result = await service.transcribe(wav.name)
            os.unlink(wav.name)
            return result

        elif tool_name == "check_video":
            service = get_video_service()
            return await service.analyze(video_url=media_url)

        elif tool_name == "extract_colors":
            service = get_color_service()
            return await service.extract_colors(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "read_barcodes":
            service = get_barcode_service()
            return await service.read_barcodes(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "extract_exif":
            service = get_exif_service()
            return await service.extract_exif(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "detect_scenes":
            service = get_scene_service()
            return await service.detect_scenes(
                video_url=params.get("video_url", media_url),
            )

        elif tool_name == "fingerprint_audio":
            service = get_fingerprint_service()
            return await service.fingerprint_audio(
                audio_url=params.get("audio_url", media_url),
            )

        elif tool_name == "detect_faces":
            service = get_face_service()
            return await service.detect_faces(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "classify_nsfw":
            service = get_nsfw_service()
            return await service.classify_nsfw(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "classify_audio":
            service = get_audio_classify_service()
            return await service.classify_audio(
                audio_url=params.get("audio_url", media_url),
            )

        elif tool_name == "segment_image":
            service = get_segment_service()
            return await service.segment(
                image_url=params.get("image_url", media_url),
            )

        elif tool_name == "diarize_audio":
            service = get_diarize_service()
            return await service.diarize(
                audio_url=params.get("audio_url", media_url),
            )

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.error(f"Process error for {tool_name}: {e}")
        return {"success": False, "error": f"Tool execution error: {str(e)[:200]}"}
