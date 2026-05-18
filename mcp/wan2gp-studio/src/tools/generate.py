"""Generation tools — video, image, 3D, audio via the Forge gateway."""
from __future__ import annotations

from typing import Annotated

from fastmcp import Context
from pydantic import Field

# Maps tool parameter names to wan2gp handler parameter names
_KEY_MAP = {
    "prompt": "input_prompt",
    "negative_prompt": "n_prompt",
    "steps": "sampling_steps",
    "guidance": "guide_scale",
    "frames": "frame_num",
    "reference_images": "input_ref_images",
}


def _build_payload(**kwargs) -> dict:
    """Build a Forge payload, mapping tool params to wan2gp params."""
    payload = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        key = _KEY_MAP.get(k, k)
        payload[key] = v
    return payload


async def _invoke_forge(ctx: Context | None, payload: dict) -> dict:
    """Get ForgeClient from lifespan context and invoke."""
    if ctx is None:
        raise RuntimeError("No MCP context available")
    forge = ctx.lifespan_context.get("forge_client")
    if forge is None:
        raise RuntimeError("Forge client not initialized")
    return await forge.invoke(payload)


async def generate_video(
    prompt: Annotated[str, Field(description="Text description of the video to generate")],
    model: Annotated[str, Field(
        description="Model family: wan/t2v, wan/i2v, hunyuan/t2v, hunyuan/i2v, ltx2",
        json_schema_extra={"enum": ["wan/t2v", "wan/i2v", "hunyuan/t2v", "hunyuan/i2v", "ltx2"]},
    )] = "wan/t2v",
    image_b64: Annotated[str | None, Field(
        description="Base64-encoded input image (required for i2v models)",
    )] = None,
    width: Annotated[int, Field(description="Output width", ge=256, le=1920)] = 768,
    height: Annotated[int, Field(description="Output height", ge=256, le=1920)] = 512,
    frames: Annotated[int, Field(description="Number of frames to generate", ge=8, le=200)] = 81,
    fps: Annotated[int, Field(description="Frames per second", ge=8, le=60)] = 24,
    steps: Annotated[int, Field(description="Denoising steps", ge=1, le=100)] = 30,
    guidance: Annotated[float, Field(description="CFG guidance scale", ge=0.0, le=20.0)] = 5.0,
    seed: Annotated[int, Field(description="Random seed (-1 for random)")] = -1,
    negative_prompt: Annotated[str | None, Field(
        description="Negative prompt (what to avoid)",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate a video from text or image input using GPU models (wan, hunyuan, ltx2)."""
    payload = _build_payload(
        service="wan2gp",
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_b64=image_b64,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )
    return await _invoke_forge(ctx, payload)


async def generate_image(
    prompt: Annotated[str, Field(description="Text description of the image to generate")],
    model: Annotated[str, Field(
        description="Model: flux, flux_schnell, flux2_dev, flux2_klein_4b, qwen-image-edit",
        json_schema_extra={"enum": [
            "flux", "flux_schnell", "flux2_dev", "flux2_klein_4b", "qwen-image-edit",
        ]},
    )] = "flux",
    image_b64: Annotated[str | None, Field(
        description="Base64 input image (for editing models)",
    )] = None,
    width: Annotated[int, Field(description="Output width", ge=256, le=2048)] = 1024,
    height: Annotated[int, Field(description="Output height", ge=256, le=2048)] = 1024,
    steps: Annotated[int, Field(description="Denoising steps", ge=1, le=100)] = 24,
    guidance: Annotated[float, Field(description="CFG guidance scale", ge=0.0, le=20.0)] = 3.5,
    seed: Annotated[int, Field(description="Random seed (-1 for random)")] = -1,
    negative_prompt: Annotated[str | None, Field(
        description="Negative prompt (what to avoid)",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate an image from text using diffusion models (flux, qwen)."""
    payload = _build_payload(
        service="wan2gp",
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_b64=image_b64,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )
    return await _invoke_forge(ctx, payload)


async def generate_3d(
    image_b64: Annotated[str, Field(
        description="Base64-encoded input image to convert to 3D mesh",
    )],
    model: Annotated[str, Field(
        description="Model: trellis or anigen",
        json_schema_extra={"enum": ["trellis", "anigen"]},
    )] = "trellis",
    steps: Annotated[int, Field(description="Generation steps", ge=1, le=200)] = 50,
    ctx: Context | None = None,
) -> dict:
    """Convert an image to a 3D mesh model using GPU (trellis, anigen)."""
    payload = _build_payload(
        service="wan2gp",
        model=model,
        image_b64=image_b64,
        steps=steps,
    )
    return await _invoke_forge(ctx, payload)


async def generate_audio(
    prompt: Annotated[str, Field(
        description="Text for TTS or sound effect description",
    )],
    model: Annotated[str, Field(
        description="Audio model: moss-soundeffect, kokoro, espeak, vibevoice_cpp_gpu, vibevoice_cpp_cpu",
        json_schema_extra={"enum": [
            "moss-soundeffect", "kokoro", "espeak",
            "vibevoice_cpp_gpu", "vibevoice_cpp_cpu",
        ]},
    )] = "moss-soundeffect",
    voice: Annotated[str | None, Field(description="Voice name (for TTS models)")] = None,
    language: Annotated[str | None, Field(description="Language code (e.g. en, zh)")] = None,
    duration_seconds: Annotated[float | None, Field(
        description="Target duration in seconds (for music/SFX)",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate audio: speech (TTS), sound effects, or music via GPU/CPU models."""
    payload = _build_payload(
        service="wan2gp",
        model=model,
        text=prompt,
        voice=voice,
        language=language,
        duration_seconds=duration_seconds,
    )
    return await _invoke_forge(ctx, payload)
