"""LLM tools — chat completions and model configuration.

- chat: Send messages to the LLM and get a response
- llm_configure: Set model, engine, system prompt, and session defaults
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field


def _forge(ctx: Context) -> Any:
    fc = ctx.lifespan_context.get("forge_client") if ctx else None
    if fc is None:
        raise RuntimeError("Forge client not available")
    return fc


async def chat(
    messages: Annotated[list[dict[str, str]], Field(
        description="Chat messages in OpenAI format: [{role:'system'|'user'|'assistant', content:'...'}]. "
                    "At least one user message is required.",
    )],
    model: Annotated[str | None, Field(
        description="Model override. Uses the current LLM default if omitted.",
    )] = None,
    temperature: Annotated[float | None, Field(
        description="Sampling temperature 0.0–2.0. Higher = more creative.",
    )] = None,
    max_tokens: Annotated[int | None, Field(
        description="Max tokens to generate in the response.",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Send messages to the LLM and get a completion.

    Routes through llama.cpp on GPU. Returns {choices: [{message: {role, content}}], usage: {...}}.
    """
    forge = _forge(ctx)

    payload: dict[str, Any] = {
        "service": "llm",
        "messages": messages,
    }
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return await forge.invoke(payload)


async def llm_configure(
    model: Annotated[str | None, Field(
        description="Model to load (e.g. 'qwen3.6-27b-q5_k_s-32k', 'llama-3.3-70b').",
    )] = None,
    system_prompt: Annotated[str | None, Field(
        description="Set the default system prompt for chat sessions.",
    )] = None,
    context_length: Annotated[int | None, Field(
        description="Context window size in tokens (e.g. 8192, 32768).",
    )] = None,
    gpu_layers: Annotated[int | None, Field(
        description="Number of layers to offload to GPU (-1 = all).",
    )] = None,
    ctx: Context | None = None,
) -> dict:
    """Configure the LLM engine: change model, system prompt, or hardware settings.

    Changes persist until the next configure call or server restart.
    Returns the new configuration state.
    """
    forge = _forge(ctx)

    payload: dict[str, Any] = {
        "service": "llm",
        "action": "configure",
    }
    if model:
        payload["model"] = model
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if context_length is not None:
        payload["context_length"] = context_length
    if gpu_layers is not None:
        payload["gpu_layers"] = gpu_layers

    return await forge.invoke(payload)
