"""Pool-dispatched step executor — calls inference pools via the dispatch bridge.

This is the generic pool client. It uses services.inference.dispatch to turn
a (service, model) step into an ordered list of HTTP endpoints, then walks
them in priority order until one returns 2xx. Connection errors / unhealthy
pools advance to the next hop automatically.

Step type: ``pool``

Workflow params:
  _service    — logical service name (forge, native, wan2gp, moss, ...)
  _model      — canonical model name (qwen-image-edit, z-image, ...)
  _action     — optional action key from the launcher's api: map
                (default: first declared action, usually "generate")
  _method     — optional HTTP method override (default: from api endpoint)
  body        — request body (merged into the TNAP envelope)

All other params become the request body. Binary file paths in the body are
detected and base64-encoded for OpenAI-compatible endpoints.

Used for pools where the request shape is JSON-and-go (SGLang, most Tier A
specialized services, the diffusers catch-all). Image/video gen via Omni has
multipart form-data specifics handled by the dedicated img_edit / vace /
ltx_video step executors, which still resolve URLs via dispatch.resolve_step
but craft their own payloads.
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from . import StepExecutor, StepContext, StepResult
from services.inference.dispatch import resolve_step, DispatchHop

logger = logging.getLogger(__name__)

# Params that may hold a file path → base64 for JSON bodies.
_B64_PARAMS = {"image_b64", "reference_image", "reference_images",
               "video_path", "audio_path", "mask_b64", "image_b64_2"}

# Long enough for the slowest valid path (BF16 + layerwise offload on a
# 14B model can be ~5 min). Pools that respond faster will return sooner.
DEFAULT_TIMEOUT = 600.0


class PoolStepExecutor(StepExecutor):
    """Resolve a (service, model) to a pool chain and dispatch with fallback."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()

        service = params.pop("_service", None) or params.pop("service", None)
        model = params.pop("_model", None) or params.pop("model", None)
        action = params.pop("_action", params.pop("action", "generate"))
        method_override = params.pop("_method", None)

        if not model and service:
            # Let dispatch's SERVICE_TO_MODEL_HINT resolve Tier A services.
            pass
        if not model and not service:
            raise ValueError(
                "Pool step needs at least one of 'service' or 'model'"
            )

        # Drop None values (unresolved template placeholders).
        body = {k: v for k, v in params.items() if v is not None}

        # Encode file-path params as base64 for JSON bodies.
        body = await self._encode_file_params(body, context)

        plan = resolve_step(service=service, model=model, action=action)
        if not plan.first_healthy:
            healthy_pools = [h.pool.name for h in plan if h.healthy]
            unhealthy = [f"{h.pool.name}({h.pool.health_url})"
                         for h in plan if not h.healthy]
            raise RuntimeError(
                f"No healthy pool for model={model!r} service={service!r}. "
                f"Candidates (priority order): {unhealthy or 'none declared'}"
            )

        logger.info("Pool dispatch %s via %s → %d hop(s), first healthy: %s",
                    model, service or "(none)", len(plan),
                    plan.first_healthy.pool.name)

        last_error: Exception | None = None
        last_status: int = 0
        last_body_snippet: str = ""
        for hop in plan:
            if not hop.healthy:
                logger.debug("Skipping unhealthy pool %s for %s",
                             hop.pool.name, model)
                continue
            method = (method_override or hop.method).upper()
            payload = hop.payload(body)
            url = hop.url
            try:
                timeout = context.config.get("pool_timeout_s", DEFAULT_TIMEOUT)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if method == "GET":
                        resp = await client.get(url, params=payload)
                    else:
                        resp = await client.request(method, url, json=payload)
                if resp.is_success:
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    return await self._build_result(
                        resp, hop, elapsed_ms, context
                    )
                logger.info(
                    "Pool %s returned %d for %s — trying next hop",
                    hop.pool.name, resp.status_code, model
                )
                last_status = resp.status_code
                last_body_snippet = resp.text[:300]
            except Exception as e:
                logger.info(
                    "Pool %s raised for %s: %s — trying next hop",
                    hop.pool.name, model, e
                )
                last_error = e

        raise RuntimeError(
            f"All pools failed for model={model!r} service={service!r}. "
            f"Last HTTP {last_status}: {last_body_snippet or 'no body'}. "
            f"Last exception: {last_error!r}."
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _encode_file_params(self, body: dict, context: StepContext) -> dict:
        """Translate artifact file paths into base64 for JSON bodies.

        Same heuristic as the Forge executor — short strings starting with
        '/' that exist as files get base64-encoded. Lists of paths too.
        """
        encoded: dict[str, Any] = {}
        for key, value in body.items():
            if key not in _B64_PARAMS:
                encoded[key] = value
                continue
            if isinstance(value, list):
                encoded[key] = [self._maybe_b64(v) for v in value]
            else:
                encoded[key] = self._maybe_b64(value)
        return encoded

    @staticmethod
    def _maybe_b64(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if not (len(value) < 4096 and value.startswith("/")):
            return value
        try:
            path = Path(value)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
        except OSError:
            pass
        return value

    async def _build_result(self, resp: httpx.Response, hop: DispatchHop,
                            elapsed_ms: int, context: StepContext) -> StepResult:
        """Parse an HTTP response into a StepResult.

        Handles three common response shapes:
          - Raw bytes (video/mp4) — store directly
          - OpenAI-style {data:[{b64_json:...}]} — decode and store
          - Arbitrary JSON — return as-is in outputs
        """
        content_type = resp.headers.get("content-type", "")
        metadata = {
            "pool": hop.pool.name,
            "tier": hop.pool.tier,
            "elapsed_ms": elapsed_ms,
            "http_status": resp.status_code,
        }

        # Raw bytes (e.g. video/mp4 from Omni /v1/videos/sync)
        if "application/octet-stream" in content_type or "video/" in content_type:
            data = resp.content
            return StepResult(
                data=data,
                media_type=content_type or "application/octet-stream",
                metadata=metadata,
            )

        # JSON response
        try:
            result = resp.json()
        except Exception:
            # Not JSON, not raw bytes — return as binary
            return StepResult(
                data=resp.content,
                media_type=content_type or "application/octet-stream",
                metadata=metadata,
            )

        # OpenAI DALL-E compatible: {data: [{b64_json: "..."}]}
        data_field = result.get("data")
        if isinstance(data_field, list) and data_field:
            first = data_field[0] if isinstance(data_field[0], dict) else {}
            b64 = first.get("b64_json") if isinstance(first, dict) else None
            if b64:
                return StepResult(
                    data=base64.b64decode(b64),
                    media_type="image/png",
                    metadata={**metadata, "model": result.get("model")},
                )
            # URL-style response — caller fetches separately
            url = first.get("url") if isinstance(first, dict) else None
            if url:
                return StepResult(
                    outputs={"url": url, **{k: v for k, v in result.items()
                                           if k != "data"}},
                    metadata=metadata,
                )

        # Generic JSON — surface all fields as outputs (skip bloated fields)
        _SKIP = {"data"}
        outputs = {k: str(v) for k, v in result.items()
                   if k not in _SKIP and isinstance(v, (str, int, float, bool))}
        return StepResult(outputs=outputs, metadata=metadata)
