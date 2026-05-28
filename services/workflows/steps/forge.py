"""Forge step executor — calls GPU services through the Forge via Ray handle.

Translates artifact file paths into the format each service expects.
For backward compatibility, services that expect image_b64 get the file
read and base64-encoded. New services can accept file_path directly.
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

# Params that commonly hold artifact references → need file → base64 conversion
_B64_PARAMS = {"image_b64", "reference_image", "reference_images", "video_path", "audio_path"}


class ForgeStepExecutor(StepExecutor):
    """Execute a GPU service call through the Forge."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()
        service = params.pop("_service", params.pop("service", ""))
        model = params.pop("_model", params.pop("model", None))

        if not service:
            raise ValueError("Forge step missing 'service' param")

        # Resolve artifact file paths → format the service expects
        resolved = await self._prepare_params(params, context)

        # Drop None values — unresolved template placeholders (e.g. missing inputs.seed)
        resolved = {k: v for k, v in resolved.items() if v is not None}

        # Call Forge via Ray handle
        forge = serve.get_deployment_handle("forge", "forge")
        result = await forge.invoke.remote(service, resolved, model)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if result.get("status") != "ok":
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Forge service '{service}' failed: {error}")

        # Store output as artifact
        outputs = await self._store_outputs(result, context)
        return StepResult(outputs=outputs, duration_ms=elapsed_ms)

    async def _prepare_params(self, params: dict, context: StepContext) -> dict:
        """Translate file-path artifact references to the format services expect."""
        resolved = {}
        for key, value in params.items():
            if key in _B64_PARAMS and isinstance(value, (str, Path)):
                path = Path(value)
                if path.exists():
                    resolved[key] = base64.b64encode(path.read_bytes()).decode()
                else:
                    resolved[key] = value
            elif key in _B64_PARAMS and isinstance(value, list):
                encoded = []
                for v in value:
                    p = Path(v) if isinstance(v, (str, Path)) else v
                    if isinstance(p, Path) and p.exists():
                        encoded.append(base64.b64encode(p.read_bytes()).decode())
                    else:
                        encoded.append(v)
                resolved[key] = encoded
            else:
                resolved[key] = value
        return resolved

    async def _store_outputs(self, result: dict, context: StepContext) -> dict[str, str]:
        """Store the service result as file artifacts."""
        outputs: dict[str, str] = {}
        data = result.get("data")
        media_type = result.get("media_type", "application/octet-stream")

        if data and isinstance(data, str):
            # Base64-encoded output from the service
            artifact = await context.artifacts.store(
                run_id=context.run_id,
                step_id=context.step_id,
                name="output",
                data=base64.b64decode(data),
                media_type=media_type,
            )
            outputs["output"] = str(artifact.file_path)

        # Store any additional metadata
        for k, v in result.items():
            if k not in ("status", "data", "media_type") and isinstance(v, (str, int, float, bool)):
                outputs[k] = str(v)

        return outputs
