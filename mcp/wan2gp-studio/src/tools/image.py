"""Image generation MCP tool — unified interface for image models.

Provides sensible defaults per model with optional parameter override.
All models route through the Wan2GP forge adapter via /v1/run.
"""
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

# ── Model presets ──────────────────────────────────────────────────────────────

_MODEL_PRESETS: dict[str, dict] = {
    # ── Z-Image family ──────────────────────────────────────────────────────
    "z_image": {
        "label": "Z-Image Turbo",
        "quality": "turbo",
        "width": 1024, "height": 1024,
        "sampling_steps": 8, "guide_scale": 0.0,
        "description": "Z-Image Turbo 6B — distilled, 8 steps, no CFG. Best for photorealism & speed.",
    },
    "z_image_base": {
        "label": "Z-Image Base",
        "quality": "standard",
        "width": 1024, "height": 1024,
        "sampling_steps": 50, "guide_scale": 4.0,
        "negative_prompt": "blurry, low quality, deformed, bad anatomy, extra fingers, watermark, cropped",
        "description": "Z-Image Base 6B — full model, 50 steps, CFG 4.0. Best for creative work, fine-tuning, max diversity.",
    },
    # ── Anima (anime/illustration) ──────────────────────────────────────────
    "anima_base": {
        "label": "Anima",
        "quality": "standard",
        "width": 1024, "height": 1024,
        "sampling_steps": 30, "guide_scale": 4.0,
        "description": "Anima Base 2B — anime/illustration focused, Cosmos architecture. 30 steps, CFG 4.0.",
    },
    # ── Flux 1 family ───────────────────────────────────────────────────────
    "flux": {
        "label": "Flux 1 Dev",
        "width": 1280, "height": 720,
        "description": "FLUX.1 Dev 12B — full rectified flow transformer.",
    },
    "flux_schnell": {
        "label": "Flux 1 Schnell",
        "width": 1280, "height": 720,
        "sampling_steps": 4, "guide_scale": 1.0,
        "description": "FLUX.1 Schnell 12B — distilled, 4 steps.",
    },
    "flux_chroma": {
        "label": "Flux Chroma HD",
        "width": 1280, "height": 720,
        "sampling_steps": 20, "guide_scale": 3.0,
        "description": "FLUX.1 Chroma 1 HD 8.9B — strong base for finetuning.",
    },
    "flux_chroma_radiance": {
        "label": "Flux Chroma Radiance",
        "width": 1280, "height": 720,
        "sampling_steps": 20, "guide_scale": 3.0,
        "description": "FLUX.1 Chroma Radiance 8.9B — improved base for finetuning.",
    },
    # ── Flux 2 family ───────────────────────────────────────────────────────
    "flux2_dev": {
        "label": "Flux 2 Dev",
        "width": 1024, "height": 1024,
        "sampling_steps": 30, "embedded_guidance_scale": 4,
        "description": "FLUX.2 Dev 32B — latest rectified flow transformer.",
    },
    "flux2_klein_4b": {
        "label": "Flux 2 Klein 4B",
        "width": 1024, "height": 1024,
        "sampling_steps": 4, "embedded_guidance_scale": 1,
        "description": "FLUX.2 Klein 4B — distilled, 4 steps, fast.",
    },
    "flux2_klein_9b": {
        "label": "Flux 2 Klein 9B",
        "width": 1024, "height": 1024,
        "sampling_steps": 4, "embedded_guidance_scale": 1,
        "description": "FLUX.2 Klein 9B — distilled, 4 steps, higher quality.",
    },
    "flux2_klein_base_4b": {
        "label": "Flux 2 Klein Base 4B",
        "width": 1024, "height": 1024,
        "description": "FLUX.2 Klein Base 4B — full model for finetuning.",
    },
    "flux2_klein_base_9b": {
        "label": "Flux 2 Klein Base 9B",
        "width": 1024, "height": 1024,
        "description": "FLUX.2 Klein Base 9B — full model for finetuning.",
    },
    # ── Qwen Image family ──────────────────────────────────────────────────
    "qwen_image_20B": {
        "label": "Qwen Image 20B",
        "width": 1328, "height": 1328,
        "description": "Qwen Image 20B — excellent long text rendering in images.",
    },
    "qwen_image_2512_20B": {
        "label": "Qwen Image 2512 20B",
        "width": 1328, "height": 1328,
        "description": "Qwen Image 2512 — enhanced realism, finer details, improved text rendering.",
    },
    # ── HiDream family ──────────────────────────────────────────────────────
    "hidream_o1": {
        "label": "HiDream O1 Full",
        "width": 1920, "height": 1088,
        "sampling_steps": 50, "guide_scale": 5.0,
        "description": "HiDream O1 Image Full 10B — unified text+pixel token space.",
    },
    "hidream_o1_dev": {
        "label": "HiDream O1 Dev",
        "width": 1920, "height": 1088,
        "sampling_steps": 28, "guide_scale": 0.0,
        "description": "HiDream O1 Image Dev 10B — distilled, fewer steps.",
    },
}

_MODEL_CHOICES = list(_MODEL_PRESETS.keys())


def get_model_preset(model: str) -> dict[str, Any]:
    """Get the preset defaults for a specific model.

    Returns the preset configuration including recommended steps, guidance, etc.
    Used by the frontend to pre-fill form fields with model-specific defaults.
    """
    return _MODEL_PRESETS.get(model, {})


async def generate(
    model: Annotated[str, Field(
        description=f"Model to use.",
        enum=_MODEL_CHOICES,
        default="z_image",
    )],
    prompt: Annotated[str, Field(
        description="Text prompt describing the image to generate. 60-200 words for best results.",
        default="",
    )],
    negative_prompt: Annotated[str | None, Field(
        description="Negative prompt for base models. Ignored by turbo/distilled models.",
    )] = None,
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    width: Annotated[int, Field(
        description="Image width in pixels. Must be divisible by 16.",
    )] = 1024,
    height: Annotated[int, Field(
        description="Image height in pixels. Must be divisible by 16.",
    )] = 1024,
    sampling_steps: Annotated[int | None, Field(
        description="Number of denoising steps. Leave empty to use model preset.",
    )] = None,
    guide_scale: Annotated[float | None, Field(
        description="CFG guidance scale. Leave empty to use model preset.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Generate images with model-specific defaults.

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    preset = _MODEL_PRESETS.get(model, {})

    # Build params: start with preset defaults, override with explicit args
    params = {
        "input_prompt": prompt,
        "seed": seed,
    }

    # Apply preset defaults (overridden by explicit args if provided)
    params["quality"] = preset.get("quality", "turbo")
    params["width"] = width if width != 1024 else preset.get("width", 1024)
    params["height"] = height if height != 1024 else preset.get("height", 1024)
    if negative_prompt is not None:
        params["n_prompt"] = negative_prompt
    elif "negative_prompt" in preset:
        params["n_prompt"] = preset["negative_prompt"]
    if sampling_steps is not None:
        params["sampling_steps"] = sampling_steps
    elif "sampling_steps" in preset:
        params["sampling_steps"] = preset["sampling_steps"]
    if guide_scale is not None:
        params["guide_scale"] = guide_scale
    elif "guide_scale" in preset:
        params["guide_scale"] = preset["guide_scale"]
    for extra_key in ("steps", "guidance", "resolution", "embedded_guidance_scale"):
        if extra_key in preset:
            params[extra_key] = preset[extra_key]

    payload = {"service": "wan2gp", "model": model, **params}
    return await client.invoke(payload)
