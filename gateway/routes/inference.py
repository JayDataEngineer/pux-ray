"""Inference pool HTTP routes — exposes the 4-tier pool system to the web-ui.

These routes mirror the MCP tools in mcp/inference/server.py. The web-ui
can hit either interface:
  - HTTP /v1/inference/*      — direct REST calls from frontend JS
  - MCP   inference tools     — model-driven discovery from a chat client

Both interfaces consume the same source of truth (config/inference_pools.yaml)
and the same Python modules (services.inference.manager / launcher / dispatch).

Routes:
  GET  /v1/inference/pools                            — list all pools
  GET  /v1/inference/pools/{pool_name}                — single pool status
  GET  /v1/inference/models                           — list all routable models
  GET  /v1/inference/models/{model}/resolve           — resolution chain
  GET  /v1/inference/models/{model}/optimization       — optimization summary
  GET  /v1/inference/resolve/{model}                  — alias for resolve
  POST /v1/inference/pools/{pool_name}/start           — start a pool
  POST /v1/inference/pools/{pool_name}/stop            — stop a pool

Admin routes (start/stop) should be gated by an auth layer in production —
they shell out to docker. Today they go through the same API key middleware
as the rest of the gateway.
"""
from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from services.inference import PoolManager
from services.inference.launcher import PoolLauncher

logger = logging.getLogger(__name__)


# ─── Module-level singletons ─────────────────────────────────────────────────
# The manager is read-only and cheap to construct (just YAML parse), but
# reuse the same instance for the lifetime of the process. The launcher
# is similarly stateless — it just shells out to docker on demand.

def _manager() -> PoolManager:
    return PoolManager.from_yaml()


def _launcher() -> PoolLauncher:
    return PoolLauncher(_manager())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _opt_dict(opt) -> dict[str, Any] | None:
    """Serialize an Optimization dataclass, dropping empty values."""
    if opt is None:
        return None
    from dataclasses import asdict
    d = asdict(opt)
    return {k: v for k, v in d.items() if v not in (None, False, [], {})}


def _resolution_entry(target) -> dict[str, Any]:
    """One hop in the resolution chain, as JSON."""
    entry: dict[str, Any] = {
        "pool": target.pool.name,
        "tier": target.pool.tier,
        "framework": target.pool.framework,
        "port": target.pool.port,
        "base_url": target.pool.base_url,
        "is_primary": target.is_primary,
        "fallback_index": target.fallback_index,
    }
    if target.launcher:
        entry["launcher"] = {
            "script": target.launcher.script,
            "patch": target.launcher.patch,
            "model_dir": target.launcher.model_dir,
            "variant": target.launcher.variant,
            "api_endpoints": target.launcher.api,
            "optimization": _opt_dict(target.launcher.optimization),
            "benchmark": target.launcher.benchmark,
        }
    return entry


# ─── Route handlers ──────────────────────────────────────────────────────────

async def list_pools(request: Request) -> JSONResponse:
    """GET /v1/inference/pools — all pools in priority order."""
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
    return JSONResponse({"object": "list", "total": len(pools), "data": pools})


async def get_pool(request: Request) -> JSONResponse:
    """GET /v1/inference/pools/{pool_name} — single pool status."""
    pool_name = request.path_params.get("pool_name", "")
    launcher = _launcher()
    status = launcher.status(pool_name)
    if "error" in status:
        return JSONResponse(status, status_code=404)
    return JSONResponse(status)


async def list_models(request: Request) -> JSONResponse:
    """GET /v1/inference/models — every routable model."""
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
    return JSONResponse({"object": "list", "total": len(models), "data": models})


async def resolve_model(request: Request) -> JSONResponse:
    """GET /v1/inference/models/{model}/resolve — ordered resolution chain."""
    model = request.path_params.get("model", "")
    mgr = _manager()
    targets = mgr.resolve(model)
    if not targets:
        return JSONResponse(
            {"error": f"No route for model '{model}'"},
            status_code=404,
        )
    chain = [_resolution_entry(t) for t in targets]
    return JSONResponse({"model": model, "resolution_chain": chain})


async def get_optimization(request: Request) -> JSONResponse:
    """GET /v1/inference/models/{model}/optimization — opt config + benchmark."""
    model = request.path_params.get("model", "")
    mgr = _manager()
    summary = mgr.optimization_summary(model)
    if not summary.get("served", True):
        return JSONResponse(summary, status_code=404)
    return JSONResponse(summary)


async def start_pool(request: Request) -> JSONResponse:
    """POST /v1/inference/pools/{pool_name}/start — start a pool's container.

    Optional body: {"model": "qwen-image-edit"} to use a specific model's
    launch script (e.g. the FP8 patch for Qwen-Image-Edit).
    """
    pool_name = request.path_params.get("pool_name", "")
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}
    model = body.get("model")
    launcher = _launcher()
    result = launcher.start(pool_name, model=model)
    return JSONResponse({
        "pool": result.pool,
        "container": result.container,
        "port": result.port,
        "healthy": result.healthy,
        "elapsed_s": result.elapsed_s,
        "message": result.message,
        "model_loaded": result.model_loaded,
    }, status_code=200 if result.healthy else 503)


async def stop_pool(request: Request) -> JSONResponse:
    """POST /v1/inference/pools/{pool_name}/stop — stop and remove a container."""
    pool_name = request.path_params.get("pool_name", "")
    launcher = _launcher()
    ok = launcher.stop(pool_name)
    return JSONResponse(
        {"pool": pool_name, "stopped": ok},
        status_code=200 if ok else 404,
    )
