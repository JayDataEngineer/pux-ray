"""Workflow MCP tools — step-by-step pipeline execution.

Tools for listing specs, creating manual runs, executing individual steps,
approving interactive steps, and rerunning failed steps.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from ..workflow_client import WorkflowClient


def _wf(ctx: Context) -> WorkflowClient:
    client = ctx.lifespan_context.get("workflow_client")
    if client is None:
        raise RuntimeError("Workflow client not initialized")
    return client


async def workflow_list_specs(
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List available workflow specs with step counts and descriptions."""
    try:
        result = await _wf(ctx).list_specs()
        return result.get("data", [])
    except Exception as e:
        from loguru import logger
        logger.error("workflow_list_specs failed: {}", e)
        return []


async def workflow_get_spec(
    spec_name: Annotated[str, Field(description="Workflow spec name (e.g. 'video_editor')")],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get workflow spec details: inputs schema, steps, dependencies, parameters."""
    return await _wf(ctx).get_spec(spec_name)


async def workflow_start_run(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    inputs: Annotated[dict[str, Any] | None, Field(
        description="Input values for the workflow. Only provide what you have — "
                    "missing inputs default to spec defaults. Common: character_prompt, "
                    "scene_prompt, seed.",
    )] = None,
    manual: Annotated[bool, Field(
        description="Manual mode: create run without auto-executing. Execute steps one at a time.",
    )] = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Start a new workflow run. Manual mode (default) lets you execute steps individually."""
    return await _wf(ctx).start_run(spec_name, inputs or {}, manual=manual)


async def workflow_get_run(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    run_id: Annotated[str, Field(description="Run ID returned by workflow_start_run")],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get run status: step states, artifacts, errors, timing."""
    return await _wf(ctx).get_run(spec_name, run_id)


async def workflow_cancel_run(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    run_id: Annotated[str, Field(description="Run ID to cancel")],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Cancel a running workflow."""
    return await _wf(ctx).cancel_run(spec_name, run_id)


async def workflow_execute_step(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    run_id: Annotated[str, Field(description="Run ID")],
    step_id: Annotated[str, Field(
        description="Step to execute (e.g. 'generate_character', 'mesh_pose', 'scene_compose'). "
                    "Dependencies must be completed first.",
    )],
    params: Annotated[dict[str, Any] | None, Field(
        description="Optional param overrides: model, prompt, steps, seed, etc.",
        default=None,
    )] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute a single pipeline step in isolation. No downstream cascade.

    Check workflow_get_run first to verify dependencies are completed.
    Returns: {run_id, step_id, status, duration_ms, outputs}.
    """
    return await _wf(ctx).execute_step(spec_name, run_id, step_id, params)


async def workflow_approve_step(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    run_id: Annotated[str, Field(description="Run ID")],
    step_id: Annotated[str, Field(description="Step waiting for approval")],
    data: Annotated[dict[str, Any] | None, Field(
        description="Approval data. For file uploads: {file_data: base64, name: filename, "
                    "media_type: mime}. For review approval: empty dict.",
        default=None,
    )] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Approve a step waiting for user input (file upload or review confirmation)."""
    return await _wf(ctx).approve_step(spec_name, run_id, step_id, data)


async def workflow_rerun_step(
    spec_name: Annotated[str, Field(description="Workflow spec name")],
    run_id: Annotated[str, Field(description="Run ID")],
    step_id: Annotated[str, Field(description="Step to rerun from")],
    params: Annotated[dict[str, Any] | None, Field(
        description="Optional new params for the rerun.",
        default=None,
    )] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Rerun from a specific step, invalidating all downstream steps."""
    return await _wf(ctx).rerun_step(spec_name, run_id, step_id, params)
