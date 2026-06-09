"""VNCCS workflow MCP tools — character sheet, pose edit, and clone pipelines.

char_sheet and clone_character route through the Forge pipeline runner.
pose_edit routes through the DAG workflow engine (vnccs_pose_edit.yaml).
"""
from __future__ import annotations

import asyncio
import base64
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
        description="Generation quality: 'turbo' (fast, 8-step QWEN) or 'standard' (full quality 20-step).",
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

    1:1 match of VN_Step1_QWEN_CharSheetGenerator_v1 ComfyUI workflow:
      1. CharacterCreator -> prompt from structured attributes (sex, age, race, etc)
      2. VNCCS_PoseGenerator -> 12-pose openpose grid reference image
      3. SD base generation -> turbo (8-step) or standard (20-step)
      4. QWEN refinement -> 4-step image edit with pose grid + poser_helper_v2 LoRA
      5. Face detailer -> DWPose face crop -> 20-step QWEN refine -> composite back

    Returns base64-encoded sheet image + optional face crop.
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
        description="Base64-encoded character image to re-pose. The character's "
                    "identity and clothing are preserved while matching the target pose.",
    )],
    pose_image_b64: Annotated[str, Field(
        description="Base64-encoded pose reference image (image1 in QWEN). "
                    "Generate one with Kimodo (motion tab) or upload a reference photo. "
                    "DWPose extracts the skeleton from this image.",
    )],
    prompt: Annotated[str, Field(
        description="QWEN prompt text. The original VNCCS prompt_template is "
                    "'Draw character from image2\\n<lighting>\\n<user_prompt>'. "
                    "Include any lighting or custom instructions here.",
    )] = "Draw character from image2",
    lighting_prompt: Annotated[str, Field(
        description="Lighting description appended after the main prompt. "
                    "In the original VNCCS workflow, VNCCS_PoseStudio auto-generated "
                    "this from the mesh camera angle (e.g. 'soft ambient lighting from "
                    "above-left, warm rim light'). Manually provide or leave empty.",
    )] = "",
    user_prompt: Annotated[str, Field(
        description="Custom user text appended to the prompt. "
                    "Matches the <user_prompt> placeholder in the VNCCS prompt_template.",
    )] = "",
    lora_name: Annotated[str, Field(
        description="VNCCS PoseStudio LoRA filename. Original workflow uses "
                    "VNCCS_PoseStudioQIE2511_V1 or V2 from MIUProject/VNCCS_PoseStudio. "
                    "Selected by VNCCS_ModelSelector + LoraLoaderModelOnly nodes.",
    )] = "VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
    lora_strength: Annotated[float, Field(
        description="PoseStudio LoRA strength (0-2). Original LoraLoaderModelOnly "
                    "node default is 1.0.",
    )] = 1.0,
    sampling_steps: Annotated[int, Field(
        description="QWEN sampling steps. Original KSampler node uses 4 "
                    "(lightning LoRA mode).",
    )] = 4,
    guide_scale: Annotated[float, Field(
        description="CFG guidance scale. Original KSampler node uses 1.0.",
    )] = 1.0,
    seed: Annotated[int, Field(
        description="Random seed for reproducibility. -1 for random.",
    )] = -1,
    ctx: Context | None = None,
) -> dict:
    """Re-pose a character using the VNCCS Pose Studio QWEN workflow.

    Routes through the DAG workflow engine (vnccs_pose_edit.yaml), NOT through
    a custom ComfyUI port. The DAG orchestrates:
      1. DWPose extracts skeleton overlay from the pose reference image
      2. QWEN-Image-Edit with VNCCS PoseStudio LoRA generates the posed character
         image1 = pose reference (from Kimodo or upload)
         image2 = character to re-pose
         image3 = DWPose skeleton overlay

    Matches the original VNCCS_Utils Pose Studio QWEN ComfyUI workflow:
      - VNCCS_ModelSelector + LoraLoaderModelOnly -> LoRA selection + strength
      - VNCCS_PoseStudio -> replaced by Kimodo for pose reference generation
      - DWPreprocessor -> skeleton extraction
      - QWEN KSampler -> 4-step lightning generation

    Returns base64-encoded image data.
    """
    if ctx is None:
        raise RuntimeError("No MCP context available")
    wf = ctx.lifespan_context.get("workflow_client")
    if wf is None:
        raise RuntimeError("Workflow client not initialized")

    # Assemble prompt from pieces (matches VNCCS prompt_template)
    assembled_prompt = prompt
    if lighting_prompt:
        assembled_prompt += f"\n{lighting_prompt}"
    if user_prompt:
        assembled_prompt += f"\n{user_prompt}"

    dag_inputs: dict[str, Any] = {
        "character_image": character_image_b64,
        "pose_image": pose_image_b64,
        "prompt": assembled_prompt,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
        "sampling_steps": sampling_steps,
        "guide_scale": guide_scale,
        "seed": seed,
    }

    # Run through DAG engine
    run = await wf.run_and_wait("vnccs_pose_edit", dag_inputs)

    if run.get("status") != "completed":
        # Collect step errors for diagnostics
        step_errors = []
        for sid, ss in run.get("step_states", {}).items():
            if isinstance(ss, dict) and ss.get("status") == "failed":
                step_errors.append(f"{sid}: {ss.get('error', 'unknown')}")
        error_detail = "; ".join(step_errors) if step_errors else run.get("error", "Workflow failed")
        return {"status": "error", "error": error_detail}

    # Fetch the final artifact from pose_transfer step
    data = await wf.get_artifact_data(
        "vnccs_pose_edit", run["run_id"], "pose_transfer", "output",
    )

    return {
        "status": "ok",
        "data": base64.b64encode(data).decode(),
        "media_type": "image/png",
    }


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
