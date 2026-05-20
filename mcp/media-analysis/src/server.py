"""Media Analysis MCP Server.

Single `process(query, media_url)` tool routed by FunctionGemma.
Individual tools also available for direct access.

Image tools:
- Florence-2-base: image analysis (captions, OCR, detection)
- YOLOv8-nano: object detection with bounding boxes
- WD14 tagger: image tagging / content classification
- InsightFace: face detection + recognition
- NudeNet: NSFW content detection
- SAM 2: image segmentation
- ColorThief: dominant color palette
- pyzbar: QR/barcode reading
- Pillow: EXIF metadata extraction

Audio tools:
- Parakeet TDT v3: speech-to-text
- PANNs: audio event classification
- Pyannote 3.1: speaker diarization (requires HF token)
- Chromaprint: audio fingerprinting

Video tools:
- FFmpeg + SSIM: temporal analysis
- PySceneDetect: shot/scene detection

Upload:
- upload: accept base64 data, return a temporary URL

Router:
- FunctionGemma 270M GGUF: sub-second routing
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from loguru import logger

from .settings import get_settings


@lifespan
async def service_lifespan(server: FastMCP):
    """Initialize and cleanup services on startup/shutdown."""
    from .router.function_router import get_router
    from .services.vision_service import get_vision_service
    from .services.yolo_service import get_yolo_service
    from .services.tagger_service import get_tagger_service
    from .services.asr_service import get_asr_service
    from .services.video_service import get_video_service
    from .services.color_service import get_color_service
    from .services.barcode_service import get_barcode_service
    from .services.exif_service import get_exif_service
    from .services.scene_service import get_scene_service
    from .services.fingerprint_service import get_fingerprint_service
    from .services.face_service import get_face_service
    from .services.nsfw_service import get_nsfw_service
    from .services.audio_classify_service import get_audio_classify_service
    from .services.segment_service import get_segment_service
    from .services.diarize_service import get_diarize_service
    from .services.grounding_dino_service import get_grounding_dino_service
    from .services.phi4_vision_service import get_phi4_vision_service
    from .services.kosmos_service import get_kosmos_service
    from .services.idle_watcher import get_idle_watcher

    settings = get_settings()
    logger.info(f"Initializing media analysis services (profile={settings.profile}, device={settings.device})...")

    # Start idle watcher (auto-unloads models after MEDIA_IDLE_TIMEOUT)
    idle_watcher = get_idle_watcher()
    await idle_watcher.start()

    # Periodic upload cleanup (every 10 minutes, delete files older than 1 hour)
    import time as _time

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)
            try:
                n = cleanup_uploads(max_age_seconds=3600)
                if n:
                    logger.info(f"Cleaned up {n} expired upload(s)")
            except Exception as e:
                logger.warning(f"Upload cleanup error: {e}")

    cleanup_task = asyncio.create_task(_cleanup_loop())

    # Load router at startup (small model, needed for every request)
    router = get_router()
    if settings.router_enabled:
        try:
            router._ensure_loaded()
            logger.info("FunctionGemma router loaded")
        except Exception as e:
            logger.warning(f"Router load failed, will use rule-based fallback: {e}")

    # All services lazy-load on first use
    vision_service = get_vision_service()
    yolo_service = get_yolo_service()
    tagger_service = get_tagger_service()
    asr_service = get_asr_service()
    video_service = get_video_service()
    color_service = get_color_service()
    barcode_service = get_barcode_service()
    exif_service = get_exif_service()
    scene_service = get_scene_service()
    fingerprint_service = get_fingerprint_service()
    face_service = get_face_service()
    nsfw_service = get_nsfw_service()
    audio_classify_service = get_audio_classify_service()
    segment_service = get_segment_service()
    diarize_service = get_diarize_service()
    grounding_dino_service = get_grounding_dino_service()
    phi4_vision_service = get_phi4_vision_service()
    kosmos_service = get_kosmos_service()

    try:
        yield {
            "idle_watcher": idle_watcher,
            "router": router,
            "vision_service": vision_service,
            "yolo_service": yolo_service,
            "tagger_service": tagger_service,
            "asr_service": asr_service,
            "video_service": video_service,
            "color_service": color_service,
            "barcode_service": barcode_service,
            "exif_service": exif_service,
            "scene_service": scene_service,
            "fingerprint_service": fingerprint_service,
            "face_service": face_service,
            "nsfw_service": nsfw_service,
            "audio_classify_service": audio_classify_service,
            "segment_service": segment_service,
            "diarize_service": diarize_service,
            "grounding_dino_service": grounding_dino_service,
            "phi4_vision_service": phi4_vision_service,
            "kosmos_service": kosmos_service,
        }
    finally:
        logger.info("Shutting down services...")
        cleanup_task.cancel()
        await idle_watcher.stop()
        router.close()
        await vision_service.close()
        await yolo_service.close()
        await tagger_service.close()
        await asr_service.close()
        await video_service.close()
        await color_service.close()
        await barcode_service.close()
        await exif_service.close()
        await scene_service.close()
        await fingerprint_service.close()
        await face_service.close()
        await nsfw_service.close()
        await audio_classify_service.close()
        await segment_service.close()
        await diarize_service.close()
        await grounding_dino_service.close()
        await phi4_vision_service.close()
        await kosmos_service.close()
        logger.info("Shutdown complete")


settings = get_settings()

mcp = FastMCP(
    name="media-analysis-mcp",
    instructions=(
        "Media analysis server for images, audio, and video. "
        "Use 'process' to let the server automatically route your request, "
        "or call individual tools directly.\n\n"
        "Upload: upload base64 data and get a temp URL for use with other tools. "
        "All imageSource/audioSource/videoSource params also accept data: URIs directly.\n\n"
        "Image tools: analyze_image, detect_objects, tag_image, extract_colors, "
        "read_barcodes, extract_exif, detect_faces, classify_nsfw, segment_image\n"
        "Audio tools: transcribe_audio, classify_audio, fingerprint_audio, diarize_audio\n"
        "Video tools: check_video, detect_scenes"
    ),
    lifespan=service_lifespan,
)


# ========== TOOL REGISTRATION ==========

from .tools.media_tools import (
    # Smart router
    process,
    # Upload
    upload,
    # Image tools
    analyze_image,
    detect_objects,
    tag_image,
    extract_colors,
    read_barcodes,
    extract_exif,
    detect_faces,
    classify_nsfw,
    segment_image,
    # Audio tools
    transcribe_audio,
    classify_audio,
    fingerprint_audio,
    diarize_audio,
    # Video tools
    check_video,
    detect_scenes,
    # Microsoft vision tools
    detect_objects_text,
    phi4_vision,
    kosmos_ocr,
)

mcp.add_tool(process)
mcp.add_tool(upload)
mcp.add_tool(analyze_image)
mcp.add_tool(detect_objects)
mcp.add_tool(tag_image)
mcp.add_tool(extract_colors)
mcp.add_tool(read_barcodes)
mcp.add_tool(extract_exif)
mcp.add_tool(detect_faces)
mcp.add_tool(classify_nsfw)
mcp.add_tool(segment_image)
mcp.add_tool(transcribe_audio)
mcp.add_tool(classify_audio)
mcp.add_tool(fingerprint_audio)
mcp.add_tool(diarize_audio)
mcp.add_tool(check_video)
mcp.add_tool(detect_scenes)
mcp.add_tool(detect_objects_text)
mcp.add_tool(phi4_vision)
mcp.add_tool(kosmos_ocr)


# ========== ASGI APP (for uvicorn --workers) ==========

import asyncio
from starlette.responses import Response
from starlette.routing import Route
from .services.media_utils import UPLOAD_DIR, cleanup_uploads

_original_http_app = mcp.http_app


def _http_app_with_upload(**kwargs):
    app = _original_http_app(**kwargs)

    async def serve_upload(request):
        """Serve an uploaded file from /tmp/media-uploads/."""
        filename = request.path_params["path"]
        # Sanitize: no path traversal
        if "/" in filename or ".." in filename:
            return Response(status_code=400)
        filepath = UPLOAD_DIR / filename
        if not filepath.exists():
            return Response(status_code=404)

        data = filepath.read_bytes()
        # Guess content type from extension
        ext = filepath.suffix.lower()
        ct_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
            ".flac": "audio/flac", ".m4a": "audio/x-m4a",
            ".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/avi",
        }
        content_type = ct_map.get(ext, "application/octet-stream")
        return Response(content=data, media_type=content_type)

    # Inject the /tmp/{path} route before the MCP catch-all
    app.router.routes.insert(0, Route("/tmp/{path:path}", serve_upload))

    return app


mcp.http_app = _http_app_with_upload

app = mcp.http_app(stateless_http=True)


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    port = settings.port
    host = settings.host

    logger.info(f"Media Analysis MCP server starting on {host}:{port}")
    logger.info(f"Direct access: http://localhost:{port}/mcp")
    logger.info(f"Profile: {settings.profile} | Device: {settings.device} | Idle timeout: {settings.idle_timeout}s")
    logger.info(f"Router: {'FunctionGemma + fallback' if settings.router_enabled else 'rule-based fallback'}")
    logger.info("Upload: upload (base64 → temp URL), /tmp/ serves uploaded files")
    logger.info("Image tools: analyze_image, detect_objects, tag_image, extract_colors, read_barcodes, extract_exif, detect_faces, classify_nsfw, segment_image")
    logger.info("Audio tools: transcribe_audio, classify_audio, fingerprint_audio, diarize_audio")
    logger.info("Video tools: check_video, detect_scenes")
    logger.info("Microsoft vision: detect_objects_text, phi4_vision (Gemma 4 E4B GGUF), kosmos_ocr")

    mcp.run(transport="http", host=host, port=port)
