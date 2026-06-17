"""MCP server for the inference pool system.

Exposes the 4-tier priority fallback pool system as MCP tools so the web-ui
(and any MCP-aware client) can:
  - discover which models are available and where they live
  - resolve a model to its pool chain (primary + fallbacks)
  - inspect optimizations (FP8, Cache-DiT, TeaCache) per model
  - check pool/container health
  - start/stop pools (admin)

Architecture role (per design):
  - Inference pools (config/inference_pools.yaml) define the dockers.
  - Workflows (config/workflows/*.yaml) reference models by name; the DAG
    engine uses PoolManager to dispatch each step to the right pool.
  - This MCP server exposes the pool system to the web-ui so users can
    browse, resolve, and manage pools from a chat / tool interface.

Runs as a standalone FastMCP server (mirrors mcp/dag/server.py).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

from services.inference import PoolManager
from services.inference.launcher import PoolLauncher


def _manager() -> PoolManager:
    """Fresh manager per call (cheap — reads YAML only)."""
    return PoolManager.from_yaml()


def _launcher() -> PoolLauncher:
    return PoolLauncher(_manager())


# ─── Tool schemas (MCP-style) ────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_inference_pools",
        "description": "List all 4 inference pools (Tier A/B/C/D) with their "
                       "framework, port, VRAM budget, and models served.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_inference_models",
        "description": "List every model the pool system knows how to route, "
                       "with its primary pool, fallback chain, and launch script.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "resolve_model",
        "description": "Resolve a model name to its priority-ordered pool chain. "
                       "Returns each pool's tier, framework, optimization config, "
                       "and benchmark if known. Use this before invoking a model "
                       "to see where it lives and what fallbacks exist.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string",
                          "description": "Model name (e.g. qwen-image-edit, z-image)"},
            },
            "required": ["model"],
        },
    },
    {
        "name": "get_pool_status",
        "description": "Get docker container + health status for one pool "
                       "(or all pools if no name given).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool": {"type": "string",
                         "description": "Optional pool name (moss, omni-vllm, etc.)"},
            },
        },
    },
    {
        "name": "get_model_optimization",
        "description": "Inspect the optimization config for a specific model: "
                       "quantization, Cache-DiT, TaylorSeer, TeaCache, VAE tiling, "
                       "and any benchmark numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
            },
            "required": ["model"],
        },
    },
    {
        "name": "start_inference_pool",
        "description": "Start an inference pool's docker container. If a model is "
                       "given, uses that model's launch script (e.g. the FP8 Qwen "
                       "patch). Admin-only — gate behind user permission in prod.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool": {"type": "string", "description": "Pool name"},
                "model": {"type": "string",
                          "description": "Optional model name (selects launch script)"},
            },
            "required": ["pool"],
        },
    },
    {
        "name": "stop_inference_pool",
        "description": "Stop and remove an inference pool's container. Admin-only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool": {"type": "string"},
            },
            "required": ["pool"],
        },
    },
]


# ─── Tool implementations ────────────────────────────────────────────────────

async def list_inference_pools() -> str:
    mgr = _manager()
    pools = []
    for p in mgr.pools():
        pools.append({
            "name": p.name,
            "tier": p.tier,
            "priority": p.priority,
            "framework": p.framework,
            "port": p.port,
            "vram_mb": p.vram_mb,
            "models": p.models,
            "description": p.description,
        })
    return json.dumps({"pools": pools, "total": len(pools)}, indent=2)


async def list_inference_models() -> str:
    mgr = _manager()
    models = []
    for m in mgr.models():
        targets = mgr.resolve(m)
        models.append({
            "model": m,
            "primary_pool": targets[0].pool.name if targets else None,
            "primary_tier": targets[0].pool.tier if targets else None,
            "fallback_pools": [t.pool.name for t in targets[1:]],
            "launch_script": (targets[0].launcher.script
                              if targets and targets[0].launcher else None),
        })
    return json.dumps({"models": models, "total": len(models)}, indent=2)


async def resolve_model(model: str) -> str:
    mgr = _manager()
    targets = mgr.resolve(model)
    if not targets:
        return json.dumps({"error": f"No route for model '{model}'"}, indent=2)
    chain = []
    for t in targets:
        entry = {
            "pool": t.pool.name,
            "tier": t.pool.tier,
            "framework": t.pool.framework,
            "port": t.pool.port,
            "is_primary": t.is_primary,
            "fallback_index": t.fallback_index,
            "base_url": t.pool.base_url,
        }
        if t.launcher:
            entry["launcher"] = {
                "script": t.launcher.script,
                "patch": t.launcher.patch,
                "model_dir": t.launcher.model_dir,
                "variant": t.launcher.variant,
                "optimization": _opt_dict(t.launcher.optimization) if t.launcher.optimization else None,
                "benchmark": t.launcher.benchmark,
                "api_endpoints": t.launcher.api,
            }
        chain.append(entry)
    return json.dumps({"model": model, "resolution_chain": chain}, indent=2)


async def get_pool_status(pool: str | None = None) -> str:
    launcher = _launcher()
    if pool:
        return json.dumps(launcher.status(pool), indent=2)
    return json.dumps({"pools": launcher.status_all()}, indent=2)


async def get_model_optimization(model: str) -> str:
    mgr = _manager()
    return json.dumps(mgr.optimization_summary(model), indent=2)


async def start_inference_pool(pool: str, model: str | None = None) -> str:
    launcher = _launcher()
    result = launcher.start(pool, model=model)
    return json.dumps({
        "pool": result.pool,
        "container": result.container,
        "port": result.port,
        "healthy": result.healthy,
        "elapsed_s": result.elapsed_s,
        "message": result.message,
        "model_loaded": result.model_loaded,
    }, indent=2)


async def stop_inference_pool(pool: str) -> str:
    launcher = _launcher()
    ok = launcher.stop(pool)
    return json.dumps({"pool": pool, "stopped": ok}, indent=2)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _opt_dict(opt) -> dict[str, Any]:
    """Serialize an Optimization dataclass to a dict, dropping empty values."""
    from dataclasses import asdict
    d = asdict(opt)
    return {k: v for k, v in d.items() if v not in (None, False, [], {})}


# ─── Entry point (FastMCP) ───────────────────────────────────────────────────

def _build_app():
    """Build a FastMCP-style app if fastmcp is available, else return None."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        return None
    app = FastMCP("inference-pools")
    app.add_tool(list_inference_pools)
    app.add_tool(list_inference_models)
    app.add_tool(resolve_model)
    app.add_tool(get_pool_status)
    app.add_tool(get_model_optimization)
    app.add_tool(start_inference_pool)
    app.add_tool(stop_inference_pool)
    return app


if __name__ == "__main__":
    app = _build_app()
    if app is None:
        print("fastmcp not installed — TOOLS and async handlers are still importable.",
              file=sys.stderr)
        sys.exit(1)
    app.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))
