"""Status and model discovery tools."""
from __future__ import annotations

from typing import Annotated

from fastmcp import Context
from pydantic import Field

MODEL_FAMILIES = {
    "wan": {
        "name": "WAN",
        "type": "video",
        "variants": ["wan/t2v", "wan/i2v"],
        "description": "Text-to-video and image-to-video generation",
    },
    "hunyuan": {
        "name": "Hunyuan",
        "type": "video",
        "variants": ["hunyuan/t2v", "hunyuan/i2v", "hunyuan/h-video"],
        "description": "High-quality video generation",
    },
    "ltx2": {
        "name": "LTX-Video",
        "type": "video",
        "variants": ["ltx2"],
        "description": "LTX-Video enhanced generation",
    },
    "flux": {
        "name": "Flux",
        "type": "image",
        "variants": ["flux", "flux_schnell", "flux2_dev", "flux2_klein_4b", "flux_chroma"],
        "description": "Diffusion image generation",
    },
    "qwen": {
        "name": "QWEN",
        "type": "image",
        "variants": ["qwen-image-edit"],
        "description": "Image editing with QWEN guidance",
    },
    "trellis": {
        "name": "TRELLIS",
        "type": "3d",
        "variants": ["trellis"],
        "description": "Image-to-3D mesh generation",
    },
    "anigen": {
        "name": "AniGen",
        "type": "3d",
        "variants": ["anigen"],
        "description": "Image-to-rigged-3D generation",
    },
    "moss": {
        "name": "MOSS",
        "type": "audio",
        "variants": ["moss-soundeffect", "moss-tts", "moss-ttsd", "moss-voicegenerator"],
        "description": "Sound effects and audio generation",
    },
    "kokoro": {
        "name": "Kokoro",
        "type": "audio",
        "variants": ["kokoro"],
        "description": "Multi-voice TTS (CPU)",
    },
    "ace_step": {
        "name": "ACE-Step",
        "type": "music",
        "variants": ["ace_step"],
        "description": "Text-to-music generation",
    },
}


async def list_models(
    ctx: Context | None = None,
) -> dict:
    """List all available GPU model families and their variants.

    Returns model families grouped by type (video, image, 3d, audio, music)
    with available variant names for use in generate_* tools.
    """
    # Try to get live status from Forge
    gpu_status = {}
    if ctx is not None:
        forge = ctx.lifespan_context.get("forge_client")
        if forge:
            try:
                gpu_status = await forge.status()
            except Exception:
                pass

    return {
        "families": MODEL_FAMILIES,
        "gpu_status": gpu_status,
    }


async def forge_status(
    detailed: Annotated[bool, Field(
        description="Include per-service VRAM breakdown",
    )] = False,
    ctx: Context | None = None,
) -> dict:
    """Check GPU status, VRAM usage, and currently loaded services.

    Returns GPU device info, allocated/free VRAM, and which services are loaded.
    Use this before generation to verify available resources.
    """
    if ctx is None:
        return {"status": "error", "error": "No MCP context"}

    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        return {"status": "error", "error": "Forge client not initialized"}

    try:
        status = await forge.status()
    except Exception as e:
        return {"status": "error", "error": f"Failed to reach Forge: {e}"}

    if not detailed:
        status.pop("gpu_nodes", None)

    return status
