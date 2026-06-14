"""MCP server for DAG workflow operations.

Tools:
  list_workflows    — List available workflow specs
  get_workflow      — Get details of a specific workflow
  start_workflow    — Start a new workflow run
  get_run_status    — Get status of a running/completed workflow
  list_runs         — List recent workflow runs
  cancel_run        — Cancel a running workflow
  get_artifact      — Download an artifact from a completed step
  get_run_events    — Get SSE events for a run (for streaming)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
WORKFLOW_SPECS_DIR = Path(__file__).resolve().parents[2] / "config" / "workflows"


# ─── MCP Tool Definitions ─────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "list_workflows",
        "description": "List all available workflow specifications (DAG templates). "
                       "Returns name, description, and required inputs for each.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_workflow",
        "description": "Get detailed information about a specific workflow, including "
                       "all steps, inputs, and their descriptions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workflow spec name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "start_workflow",
        "description": "Start a new workflow run. Returns run_id and initial status. "
                       "The workflow executes asynchronously — use get_run_status to poll.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow": {"type": "string", "description": "Workflow spec name"},
                "inputs": {"type": "object", "description": "Input parameters for the workflow"},
                "skip_review": {"type": "boolean", "default": True,
                                "description": "Skip review pauses (for API calls)"},
            },
            "required": ["workflow", "inputs"],
        },
    },
    {
        "name": "get_run_status",
        "description": "Get the current status of a workflow run, including "
                       "step-by-step progress, elapsed time, and any errors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Workflow run ID"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "list_runs",
        "description": "List recent workflow runs with their statuses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "status": {"type": "string", "description": "Filter by status (running, completed, failed)"},
            },
        },
    },
    {
        "name": "cancel_run",
        "description": "Cancel a running workflow.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "list_native_models",
        "description": "List all models available through the native diffusers service, "
                       "including pipeline class, default steps, and task type.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_image",
        "description": "Quick image generation using native diffusers. "
                       "Shortcut that starts a native_generate workflow.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Generation prompt"},
                "model": {"type": "string", "default": "z-image-turbo",
                          "description": "Model name (z-image-turbo, flux-schnell, anima, etc.)"},
                "seed": {"type": "integer", "default": -1},
                "steps": {"type": "integer", "default": -1},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
            },
            "required": ["prompt"],
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────────

import httpx


async def list_workflows() -> str:
    """List all available workflow specs from config/workflows/."""
    import yaml

    workflows = []
    for path in sorted(WORKFLOW_SPECS_DIR.glob("*.yaml")):
        try:
            with open(path) as f:
                spec = yaml.safe_load(f)
            workflows.append({
                "name": spec.get("name", path.stem),
                "description": spec.get("description", ""),
                "inputs": list(spec.get("inputs", {}).keys()),
                "steps": [s.get("id", "?") for s in spec.get("steps", [])],
            })
        except Exception as e:
            workflows.append({"name": path.stem, "error": str(e)})

    return json.dumps({"workflows": workflows}, indent=2)


async def get_workflow(name: str) -> str:
    """Get detailed info about a specific workflow."""
    import yaml

    for ext in (".yaml", ".yml"):
        path = WORKFLOW_SPECS_DIR / f"{name}{ext}"
        if path.exists():
            with open(path) as f:
                spec = yaml.safe_load(f)
            return json.dumps(spec, indent=2, default=str)

    return json.dumps({"error": f"Workflow '{name}' not found"})


async def start_workflow(workflow: str, inputs: dict, skip_review: bool = True) -> str:
    """Start a workflow run through the gateway."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{GATEWAY_URL}/wf-internal/start",
            json={
                "spec": workflow,
                "inputs": inputs,
                "skip_review": skip_review,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return json.dumps({
                "run_id": data.get("run_id"),
                "status": data.get("status", "started"),
                "workflow": workflow,
            }, indent=2)
        return json.dumps({"error": f"Gateway returned {resp.status_code}: {resp.text}"})


async def get_run_status(run_id: str) -> str:
    """Get status of a workflow run."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{GATEWAY_URL}/wf-internal/runs/{run_id}/status")
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
        return json.dumps({"error": f"Run '{run_id}' not found"})


async def list_runs(limit: int = 10, status: str | None = None) -> str:
    """List recent workflow runs."""
    async with httpx.AsyncClient(timeout=30) as client:
        params = {"limit": limit}
        if status:
            params["status"] = status
        resp = await client.get(f"{GATEWAY_URL}/wf-internal/runs", params=params)
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
        return json.dumps({"error": f"Gateway returned {resp.status_code}"})


async def cancel_run(run_id: str) -> str:
    """Cancel a running workflow."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{GATEWAY_URL}/wf-internal/runs/{run_id}/cancel")
        return json.dumps({"run_id": run_id, "cancelled": resp.status_code == 200})


async def list_native_models() -> str:
    """List all models in the native diffusers registry."""
    # Import directly (no gateway call needed)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from services.native.models import MODELS

    models = []
    for name, cfg in MODELS.items():
        models.append({
            "name": name,
            "pipeline": cfg.pipeline_class,
            "task": cfg.task,
            "default_steps": cfg.default_steps,
            "license": cfg.license,
            "notes": cfg.notes,
        })

    return json.dumps({"models": models}, indent=2)


async def generate_image(prompt: str, model: str = "z-image-turbo",
                         seed: int = -1, steps: int = -1,
                         width: int = 1024, height: int = 1024) -> str:
    """Quick generation — starts native_generate workflow."""
    inputs = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
    }
    if seed >= 0:
        inputs["seed"] = seed
    if steps > 0:
        inputs["steps"] = steps

    return await start_workflow("native_generate", inputs)


# ─── Dispatch ──────────────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "list_workflows": lambda args: list_workflows(),
    "get_workflow": lambda args: get_workflow(args["name"]),
    "start_workflow": lambda args: start_workflow(args["workflow"], args.get("inputs", {}), args.get("skip_review", True)),
    "get_run_status": lambda args: get_run_status(args["run_id"]),
    "list_runs": lambda args: list_runs(args.get("limit", 10), args.get("status")),
    "cancel_run": lambda args: cancel_run(args["run_id"]),
    "list_native_models": lambda args: list_native_models(),
    "generate_image": lambda args: generate_image(
        args["prompt"], args.get("model", "z-image-turbo"),
        args.get("seed", -1), args.get("steps", -1),
        args.get("width", 1024), args.get("height", 1024),
    ),
}


async def handle_tool_call(name: str, arguments: dict) -> str:
    """Dispatch a tool call to its handler."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return await handler(arguments)
    except Exception as e:
        logger.exception("Tool '%s' failed", name)
        return json.dumps({"error": str(e)})


# ─── MCP Server Entry Point ────────────────────────────────────────────────────

def run_server():
    """Run the MCP server (stdio mode)."""
    import asyncio
    import select

    logger.info("DAG MCP server starting (stdio mode)")

    while True:
        # Read line from stdin
        line = sys.stdin.readline()
        if not line:
            break

        try:
            msg = json.loads(line)
            method = msg.get("method", "")
            msg_id = msg.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "dag-mcp", "version": "1.0"},
                    },
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": TOOLS},
                }
            elif method == "tools/call":
                tool_name = msg.get("params", {}).get("name")
                tool_args = msg.get("params", {}).get("arguments", {})
                result_text = asyncio.run(handle_tool_call(tool_name, tool_args))
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                    },
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.exception("Error handling message")
            if msg_id:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32603, "message": str(e)},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    run_server()
