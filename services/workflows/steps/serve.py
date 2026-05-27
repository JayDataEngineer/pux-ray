"""Serve step executor — calls CPU services through Ray Serve handles.

Used for services like Kokoro TTS, eSpeak, and Faster-Whisper that run
as independent Ray Serve deployments (no GPU, no Forge needed).
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from ray import serve

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)


class ServeStepExecutor(StepExecutor):
    """Execute a CPU service call via Ray Serve handle."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        service = params.pop("_service", params.pop("service", ""))

        if not service:
            raise ValueError("Serve step missing 'service' param")

        # Resolve artifact file paths to base64 if needed
        resolved = await self._prepare_params(params, context)

        # Look up the deployment name from the service registry
        deployment = _resolve_deployment(service)

        handle = serve.get_deployment_handle(deployment, deployment)
        result = await handle.remote(resolved)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(f"Serve service '{service}' failed: {result.get('error')}")

        outputs = await self._store_outputs(result, context)
        return StepResult(outputs=outputs, duration_ms=elapsed_ms)

    async def _prepare_params(self, params: dict, context: StepContext) -> dict:
        """Resolve artifact references to the format the service expects."""
        from pathlib import Path as P
        resolved = {}
        for key, value in params.items():
            if isinstance(value, (str, Path)):
                p = P(value)
                if p.exists() and p.suffix in (".png", ".jpg", ".jpeg", ".wav", ".mp3", ".mp4"):
                    resolved[key] = base64.b64encode(p.read_bytes()).decode()
                    continue
            resolved[key] = value
        return resolved

    async def _store_outputs(self, result: dict | bytes, context: StepContext) -> dict[str, str]:
        """Store the service result as file artifacts."""
        outputs: dict[str, str] = {}

        if isinstance(result, bytes):
            # Raw bytes response
            artifact = await context.artifacts.store(
                context.run_id, context.step_id, "output", result, "application/octet-stream"
            )
            outputs["output"] = str(artifact.file_path)
            return outputs

        if isinstance(result, dict):
            data = result.get("data")
            media_type = result.get("media_type", "application/octet-stream")

            if data and isinstance(data, str):
                artifact = await context.artifacts.store(
                    context.run_id,
                    context.step_id,
                    "output",
                    base64.b64decode(data),
                    media_type,
                )
                outputs["output"] = str(artifact.file_path)

        return outputs


def _resolve_deployment(service: str) -> str:
    """Map service name to Ray Serve deployment name.

    Checks the SERVICE_REGISTRY first, falls back to the service name.
    """
    try:
        from services.registry import SERVICE_REGISTRY
        entry = SERVICE_REGISTRY.get(service)
        if entry and hasattr(entry, "deployment"):
            return entry.deployment
    except (ImportError, AttributeError):
        pass
    return service
