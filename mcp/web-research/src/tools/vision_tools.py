"""Vision Analysis Tools

Tools for analyzing images using the Florence-2 vision model.
- analyze_image: Analyze an image by URL
"""

from typing import Annotated

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field


async def analyze_image(
    imageSource: Annotated[str, Field(
        description="Remote URL to the image (supports PNG, JPG, JPEG)",
    )],
    prompt: Annotated[str, Field(
        description=(
            "Detailed text prompt. If the task is **front-end code replication**, "
            "the prompt you provide must be: \"Describe in detail the layout structure, "
            "color style, main components, and interactive elements of the website in "
            "this image to facilitate subsequent code generation by the model.\" + your "
            "additional requirements. For **other tasks**, the prompt you provide must "
            "clearly describe what to analyze, extract, or understand from the image."
        ),
    )],
    ctx: Context | None = None,
) -> dict:
    """Analyze an image using advanced AI vision models with comprehensive understanding capabilities.

    Only supports remote URL.

    Args:
        imageSource: Remote URL to the image (supports PNG, JPG, JPEG).
        prompt: Detailed text prompt describing what to analyze or extract.

    Returns:
        Dictionary with analysis results from the vision model.
    """
    if ctx:
        await ctx.info(f"Analyzing image: {imageSource[:80]}...")

    vision_svc = ctx.lifespan_context.get("vision_service")
    if not vision_svc:
        raise ToolError("Vision service not available")

    result = await vision_svc.analyze(
        image_url=imageSource,
        task="<MORE_DETAILED_CAPTION>",
        text_input=prompt,
    )

    if not result.get("success"):
        error = result.get("error", "Unknown error")
        if ctx:
            await ctx.error(f"Image analysis failed: {error}")
        raise ToolError(f"Image analysis failed: {error}")

    if ctx:
        await ctx.info("Image analysis complete")

    return result
