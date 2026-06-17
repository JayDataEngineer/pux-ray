"""Dispatch bridge — route workflow steps to inference pools.

This module sits between the DAG/workflow engine and the inference pool
system. It takes a (service, model) pair from a workflow step and produces
an ordered list of HTTP endpoints to try, plus the request shape each pool
expects.

Usage from the workflow engine:

    from services.inference.dispatch import resolve_step
    plan = resolve_step(service="forge", model="qwen-image-edit")
    for hop in plan:
        if hop.healthy:
            response = hop.client.post(hop.url, json=hop.payload(body))
            if response.ok:
                return response.json()
    raise RuntimeError("no pool could serve the request")

Key design points:
  - The workflow spec's ``service:`` field (e.g. "forge", "native") is the
    LOGICAL service. The pool system is the PHYSICAL backend. Multiple
    logical services may map into the same pool (e.g. "native" + "wan2gp"
    both resolve to omni-vllm for DiT models).
  - ``model:`` is the canonical model name; resolution walks the pool
    chain from PoolManager.resolve().
  - All hops share the same TNAP-style body; per-pool request shaping lives
    in the launcher's ``api`` field (POST /v1/images/generations, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from services.inference.config import Pool
from services.inference.manager import PoolManager, ResolvedTarget
from services.inference.launcher import is_healthy

logger = logging.getLogger(__name__)


# ─── Logical → physical service mapping ──────────────────────────────────────
# Workflow specs use abstract service names; we collapse them into the model
# name for routing. The pool that owns the model handles the actual call.
SERVICE_TO_MODEL_HINT: dict[str, str] = {
    # Legacy "forge" / "native" services become a no-op — model name drives routing.
    "forge": "",
    "native": "",
    "wan2gp": "",
    # Services where the service name IS the model name (Tier A specialized).
    "moss": "moss_tts",
    "moss_soundeffect": "moss_soundeffect",
    "diarization": "diarization",
    "comfyui": "comfyui",
    "llm": "llama",
    "llama-bee": "llama-bee",
}


# ─── Dispatch plan ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DispatchHop:
    """A single candidate HTTP endpoint for fulfilling a workflow step."""
    pool: Pool
    target: ResolvedTarget
    url: str                    # full URL including action path
    action: str                 # "generate" | "edit" | "transcribe" | ...
    method: str                 # "POST" | "GET"
    healthy: bool

    def payload(self, body: dict[str, Any]) -> dict[str, Any]:
        """Wrap a workflow step body into the pool's TNAP envelope.

        Most pools accept OpenAI-compatible / TNAP-shaped bodies directly.
        Per-pool quirks (e.g. ComfyUI's /prompt JSON graph) are handled by
        the pool's adapter in services/<area>/, not here.
        """
        return {
            "service": self.pool.name,
            "model": self.target.model,
            "tier": self.pool.tier,
            **body,
        }


class DispatchPlan(list[DispatchHop]):
    """Ordered list of hops. Truthy if at least one healthy hop exists."""
    @property
    def first_healthy(self) -> DispatchHop | None:
        for hop in self:
            if hop.healthy:
                return hop
        return None


# ─── Resolver ────────────────────────────────────────────────────────────────

def _pick_action(target: ResolvedTarget, fallback: str = "generate") -> tuple[str, str]:
    """Return (action_name, http_path) from the launcher's api: map."""
    if target.launcher and target.launcher.api:
        # Pick first declared action (typically "generate" or "edit").
        action, endpoint = next(iter(target.launcher.api.items()))
        # Endpoint looks like "POST /v1/images/generations"
        try:
            method, path = endpoint.split(" ", 1)
        except ValueError:
            method, path = "POST", endpoint
        return action, path
    return fallback, "/v1/generate"


def resolve_step(service: str | None, model: str,
                 action: str = "generate",
                 manager: PoolManager | None = None) -> DispatchPlan:
    """Resolve a workflow (service, model) step to an ordered dispatch plan.

    Walks the pool chain from PoolManager.resolve(). Each pool that actually
    serves the model becomes a DispatchHop with its URL + action path. Use
    the plan to make HTTP calls in priority order, falling through on
    connection errors or unhealthy pools.
    """
    mgr = manager or PoolManager.from_yaml()

    # Resolve model name. If service itself implies a model (Tier A), use it.
    canonical = model
    if (not canonical or canonical == "auto") and service in SERVICE_TO_MODEL_HINT:
        canonical = SERVICE_TO_MODEL_HINT[service]
    if not canonical:
        raise ValueError(
            f"Cannot resolve step: service={service!r}, model={model!r}. "
            "No model name and service is not a Tier A specialization."
        )

    targets = mgr.resolve(canonical)
    if not targets:
        raise ValueError(
            f"No pool serves model {canonical!r} "
            f"(looked in {len(mgr.pools())} pools)"
        )

    plan: DispatchPlan = DispatchPlan()
    for target in targets:
        picked_action, path = _pick_action(target, fallback=action)
        url = f"{target.pool.base_url}{path}"
        method = "POST"
        if target.launcher and target.launcher.api:
            endpoint = target.launcher.api.get(picked_action, "")
            if endpoint.startswith("GET "):
                method = "GET"
        plan.append(DispatchHop(
            pool=target.pool,
            target=target,
            url=url,
            action=picked_action,
            method=method,
            healthy=is_healthy(target.pool),
        ))
    return plan


# ─── Convenience: synchronous call with auto-fallback ────────────────────────

def call(service: str | None, model: str, body: dict[str, Any],
         *, timeout: float = 600.0,
         manager: PoolManager | None = None) -> dict[str, Any]:
    """Synchronous dispatch with automatic fallback.

    Tries each hop in priority order. The first pool that returns a 2xx
    response wins. Connection errors / timeouts advance to the next hop.
    """
    plan = resolve_step(service, model, manager=manager)
    last_error: Exception | None = None
    for hop in plan:
        if not hop.healthy:
            logger.debug("Skipping unhealthy pool %s for %s",
                         hop.pool.name, model)
            continue
        try:
            with httpx.Client(timeout=timeout) as client:
                if hop.method == "POST":
                    resp = client.post(hop.url, json=hop.payload(body))
                else:
                    resp = client.get(hop.url, params=body)
                if resp.is_success:
                    return resp.json()
                logger.info("Pool %s returned %d for %s: %s",
                            hop.pool.name, resp.status_code, model,
                            resp.text[:200])
        except Exception as e:
            last_error = e
            logger.info("Pool %s failed for %s: %s",
                        hop.pool.name, model, e)
    raise RuntimeError(
        f"All pools failed for model={model!r} service={service!r}. "
        f"Last error: {last_error}"
    )
