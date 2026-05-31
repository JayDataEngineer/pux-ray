"""VNCCS workflow MCP tools — character sheet and pose edit pipelines.

Both route through the Forge's DAG pipeline runner at /v1/run
with the pipeline key set to the VNCCS workflow name.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field


async def char_sheet(
    prompt: Annotated[str, Field(
        description="Text description of the character. Include appearance, clothing, style.",
    )],
    quality: Annotated[str, Field(
        description="Generation quality: 'turbo' (4-step, fast) or 'standard' (50-step, detailed).",
        enum=["turbo", "standard"],
        default="turbo",
    )] = "turbo",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    negative_prompt: Annotated[str, Field(
        description="Things to avoid in the generated image.",
    )] = "",
    ctx: Context | None = None,
) -> dict:
    """Generate a character base sheet from a text description.

    Two-stage pipeline:
      1. Z-Image generates the base character
      2. QWEN-Image-Edit refines details

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    params: dict[str, Any] = {"prompt": prompt, "seed": seed}
    if quality:
        params["quality"] = quality
    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    return await client.invoke({"pipeline": "vnccs/char-sheet", "params": params})


async def pose_edit(
    character_image_b64: Annotated[str, Field(
        description="Base64-encoded character image to re-pose.",
    )],
    rotations: Annotated[str, Field(
        description="JSON string of joint rotations: {\"joint_name\": [rx, ry, rz], ...}. "
                    "Leave empty for default T-pose.",
    )] = "",
    model_rotation_y: Annotated[float, Field(
        description="Camera angle in degrees: 0=front, 90=right, 180=back, 270=left.",
    )] = 0.0,
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Apply a body mesh pose to a character image.

    Three-stage pipeline:
      1. BodyMesh renders joint rotations into a 3D skeleton image (CPU)
      2. DWPose extracts a skeleton overlay (CPU)
      3. QWEN-Image-Edit composites mesh + character + skeleton → posed character

    The character keeps their identity and clothing while matching the target pose.

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    params: dict[str, Any] = {
        "character_image_b64": character_image_b64,
        "seed": seed,
    }
    if rotations:
        import json as _json
        try:
            params["rotations"] = _json.loads(rotations)
        except _json.JSONDecodeError:
            return {"status": "error", "error": "rotations must be valid JSON"}
    params["model_rotation_y"] = model_rotation_y

    return await client.invoke({"pipeline": "vnccs/pose-edit", "params": params})
