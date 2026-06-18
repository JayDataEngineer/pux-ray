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
    logical services may map into the same pool (e.g. "forge" + "native"
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
    # Legacy services — model name drives routing.
    "forge": "",
    "native": "",

    # ── Tier A: MOSS (Python server, migrating to OpenMOSS C++) ─────────────
    "moss": "moss-tts",
    "moss_tts": "moss-tts",
    "moss-tts": "moss-tts",
    "moss_ttsd": "moss-ttsd",
    "moss-ttsd": "moss-ttsd",
    "moss_soundeffect": "moss-soundeffect",
    "moss-soundeffect": "moss-soundeffect",
    "moss_soundeffect_v2": "moss-soundeffect-v2",
    "moss-soundeffect-v2": "moss-soundeffect-v2",
    "moss_tts_realtime": "moss-tts-realtime",
    "moss-tts-realtime": "moss-tts-realtime",
    "moss_tts_local_transformer": "moss-tts-local-transformer",
    "moss-tts-local-transformer": "moss-tts-local-transformer",
    "moss_voicegenerator": "moss-voicegenerator",
    "moss-voicegenerator": "moss-voicegenerator",
    "moss_tts_nano": "moss-tts-nano",
    "moss-tts-nano": "moss-tts-nano",

    # ── Tier A: Diarization / ASR / Whisper (all via CrispASR) ─────────────
    "diarization": "diarization",
    "diarization-base": "diarization-base",
    "diarization-turbo": "diarization-turbo",
    "asr": "diarization",                     # generic "asr" service → base
    "whisper": "diarization",                 # CrispASR handles whisper requests
    "faster-whisper": "diarization",          # legacy alias → diarization
    "faster_whisper": "diarization",

    # ── Tier A: LLM / ComfyUI ───────────────────────────────────────────────
    "comfyui": "comfyui",
    "llm": "llama",
    "llama": "llama",
    "llama-bee": "llama-bee",

    # ── Tier A: ACE-Step C++ (acestep.cpp, /lm → /synth two-step) ────────────
    "ace-step": "ace-step",
    "ace-step-turbo": "ace-step-turbo",
    "ace_step": "ace-step",
    "ace_step_turbo": "ace-step-turbo",

    # ── Tier B: vLLM-Omni DiT models ────────────────────────────────────────
    "qwen-image-edit": "qwen-image-edit",
    "qwen-image-edit-turbo": "qwen-image-edit-turbo",
    "wan-vace": "wan-vace",
    "wan-vace-turbo": "wan-vace-turbo",
    "z-image": "z-image",
    "z-image-turbo": "z-image",              # turbo alias → z-image (omni-vllm primary)
    "z-image-base": "z-image-base",          # base: same pipeline, non-distilled weights (omni-vllm → sglang fallback)
    "cosmos": "cosmos",

    # ── Tier C: SGLang ───────────────────────────────────────────────────────
    "ideogram4": "ideogram4",
    "ltx-video": "ltx-video",

    # ── Tier D: Diffusers catch-all ──────────────────────────────────────────
    "kimodo": "kimodo",
    "kokoro": "kokoro",
    "see-through": "see-through",
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

def _pick_action(target: ResolvedTarget, preferred: str | None = None,
                 fallback: str = "generate") -> tuple[str, str]:
    """Return (action_name, http_path) from the launcher's api: map.

    Honors ``preferred`` if it's declared in the api map; otherwise picks
    the first declared action.
    """
    if target.launcher and target.launcher.api:
        api = target.launcher.api
        # Prefer the requested action if it exists.
        action = preferred if (preferred and preferred in api) else None
        if action is None:
            # Fall back to first declared action.
            action = next(iter(api.keys()))
        endpoint = api[action]
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

    ``action`` is the PREFERRED action key from the launcher's api: map
    (e.g. "edit" vs "generate"). If the preferred action isn't declared
    for a pool, the first declared action is used instead — so a pool
    that only declares "generate" can still serve an "edit" workflow step
    via its generate endpoint.
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
        picked_action, path = _pick_action(target, preferred=action, fallback=action)
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
