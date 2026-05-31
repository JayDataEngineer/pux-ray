"""VNCCS workflow MCP tools — character sheet, pose edit, and clone pipelines.

All route through the Forge's DAG pipeline runner at /v1/run
with the pipeline key set to the VNCCS workflow name.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field


async def char_sheet(
    prompt: Annotated[str, Field(
        description="Text description override. If provided, this is used as the "
                    "full prompt instead of building one from the character attributes.",
    )] = "",
    image_b64: Annotated[str | None, Field(
        description="Base64-encoded starting character image. If provided, skips SD "
                    "base generation and refines this image directly into a sheet.",
    )] = None,
    reference_image_b64: Annotated[str | None, Field(
        description="Base64-encoded body reference template image for the sheet layout. "
                    "Overrides the default character sheet template.",
    )] = None,
    model: Annotated[str, Field(
        description="SD model for base character generation. 'z_image' for SD distilled "
                    "(fast), 'z_image_base' for full quality SD, 'flux_schnell' for Flux.",
        enum=["z_image", "z_image_base", "flux_schnell", "flux_dev", "anima_base"],
    )] = "z_image",
    nsfw: Annotated[bool, Field(
        description="Enable NSFW/nude mode. When true, character is generated "
                    "naked. When false (default), wears underwear.",
    )] = False,
    background_color: Annotated[str, Field(
        description="Background color for the character sheet (e.g. 'green', 'white', 'transparent').",
    )] = "green",
    aesthetics: Annotated[str, Field(
        description="Aesthetic quality tags. E.g. 'masterpiece,best quality,amazing quality'.",
    )] = "masterpiece,best quality,amazing quality",
    sex: Annotated[str, Field(
        description="Character sex.",
        enum=["female", "male"],
    )] = "female",
    age: Annotated[int, Field(
        description="Character age (0-120). Drives body type descriptors and LoRA strength.",
    )] = 18,
    race: Annotated[str, Field(
        description="Character race/ethnicity (e.g. 'human', 'elf', 'cat').",
    )] = "human",
    eyes: Annotated[str, Field(
        description="Eye description (e.g. 'blue eyes', 'red eyes', 'green eyes').",
    )] = "blue eyes",
    hair: Annotated[str, Field(
        description="Hair description (e.g. 'black long', 'short blonde', 'red curly').",
    )] = "black long",
    face: Annotated[str, Field(
        description="Face features (e.g. 'freckles', 'sharp', 'oval').",
    )] = "",
    body: Annotated[str, Field(
        description="Body type (e.g. 'medium breasts', 'slim', 'athletic', 'curvy').",
    )] = "medium breasts",
    skin_color: Annotated[str, Field(
        description="Skin color (e.g. 'white', 'tan', 'dark', 'pale').",
    )] = "",
    additional_details: Annotated[str, Field(
        description="Any additional character details not covered by other fields.",
    )] = "",
    lora_prompt: Annotated[str, Field(
        description="Additional LoRA trigger words to append to the prompt.",
    )] = "",
    quality: Annotated[str, Field(
        description="Generation quality: 'turbo' (fast, 4-step QWEN) or 'standard' (full quality).",
        enum=["turbo", "standard"],
        default="turbo",
    )] = "turbo",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    negative_prompt: Annotated[str, Field(
        description="Override negative prompt. If empty, a default is built from character attributes.",
    )] = "",
    ctx: Context | None = None,
) -> dict:
    """Generate a character base sheet matching the VNCCS CharacterCreator workflow.

    Builds the prompt exactly like the original ComfyUI CharacterCreator node
    from structured attributes (sex, age, race, eyes, hair, body, nsfw, etc).

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    params: dict[str, Any] = {"seed": seed}
    if prompt:
        params["prompt"] = prompt
    if image_b64:
        params["image_b64"] = image_b64
    if reference_image_b64:
        params["reference_image_b64"] = reference_image_b64
    for k in ("nsfw", "model", "background_color", "aesthetics", "sex", "age", "race",
              "eyes", "hair", "face", "body", "skin_color", "additional_details",
              "lora_prompt", "quality"):
        params[k] = locals()[k]
    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    return await client.invoke({"pipeline": "vnccs/char-sheet", "params": params})


async def pose_edit(
    character_image_b64: Annotated[str, Field(
        description="Base64-encoded character image to re-pose. The character's identity "
                    "and clothing are preserved while matching the target pose.",
    )],
    model_rotation_y: Annotated[float, Field(
        description="Camera angle in degrees: 0=front, 90=right, 180=back, 270=left.",
    )] = 0.0,
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Re-pose a character image using BodyMesh + DWPose + QWEN.

    Three-stage VNCCS pipeline:
      1. BodyMesh renders the target pose as a 3D skeleton image (CPU)
      2. DWPose extracts a skeleton overlay from the mesh (CPU)
      3. QWEN-Image-Edit composites mesh + character + skeleton → posed character

    Uses the VNCCS PoseStudio V2 LoRA for high-quality pose preservation.
    Default pose is a front-facing T-pose. For custom poses, use the kimodo
    motion service to generate joint rotation data.

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
    params["model_rotation_y"] = model_rotation_y

    return await client.invoke({"pipeline": "vnccs/pose-edit", "params": params})


async def clone_character(
    reference_image_b64: Annotated[str, Field(
        description="Base64-encoded reference character image to clone/modify.",
    )],
    sex: Annotated[str, Field(
        description="Target character sex.",
        enum=["female", "male"],
    )] = "female",
    age: Annotated[int, Field(
        description="Target character age. Drives body type descriptors.",
    )] = 18,
    race: Annotated[str, Field(
        description="Target character race/ethnicity.",
    )] = "human",
    eyes: Annotated[str, Field(
        description="Target eye description.",
    )] = "blue eyes",
    hair: Annotated[str, Field(
        description="Target hair description.",
    )] = "black long",
    face: Annotated[str, Field(
        description="Target face features.",
    )] = "",
    body: Annotated[str, Field(
        description="Target body type.",
    )] = "medium breasts",
    skin_color: Annotated[str, Field(
        description="Target skin color.",
    )] = "",
    additional_details: Annotated[str, Field(
        description="Any additional changes to apply to the cloned character.",
    )] = "",
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Clone an existing character with modified attributes.

    Takes a reference character image and re-renders it with new
    character attributes (age, hair, eyes, body, etc) using QWEN-Image-Edit.
    The character's core identity is preserved while applying the changes.

    Corresponds to the VNCCS Step1.1 Clone workflow.

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    client = ctx.lifespan_context.get("forge_client")
    if client is None:
        raise RuntimeError("API client not initialized")

    character_def: dict[str, Any] = {}
    for k in ("sex", "age", "race", "eyes", "hair", "face", "body", "skin_color", "additional_details"):
        v = locals()[k]
        if v:
            character_def[k] = v

    params: dict[str, Any] = {
        "reference_image_b64": reference_image_b64,
        "character_def": character_def,
        "seed": seed,
    }

    return await client.invoke({"pipeline": "vnccs/clone", "params": params})
