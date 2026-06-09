"""Python function step executor — wraps legacy workflow functions.

Provides a migration path for procedural workflows (loops, conditionals,
dynamic parameter construction) that don't decompose cleanly into static DAGs.

Usage in YAML:
  steps:
    run_emotions:
      type: python
      function: services.workflows.vnccs:emotions
      depends_on: [char_sheet]
      params:
        sheet_image_b64: "{{ char_sheet.outputs.image }}"
        emotions_list: "{{ inputs.emotions }}"
      outputs:
        results: { media_type: application/json }

The function is imported by dotted path and called with the resolved params.
If the function is async, it's awaited. The return value is stored as the
step's output. For functions returning base64 data, the output is decoded
and stored as an artifact file.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from . import StepContext, StepExecutor, StepResult

logger = logging.getLogger(__name__)

# Functions that return base64-encoded data in a "data" key
_B64_KEYS = {"data", "image_b64", "video_b64", "audio_b64"}


def _import_function(dotted_path: str) -> Any:
    """Import a function from a dotted path like 'module.path:function'."""
    if ":" in dotted_path:
        module_path, func_name = dotted_path.rsplit(":", 1)
    elif "." in dotted_path:
        module_path, func_name = dotted_path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid function path: {dotted_path}")

    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(f"Function '{func_name}' not found in '{module_path}'")
    return fn


class PythonStepExecutor(StepExecutor):
    """Execute a legacy Python workflow function as a step."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        import asyncio
        import base64
        import time

        func_path = params.get("function") or params.get("_function")
        if not func_path:
            raise ValueError("Python step requires 'function' param with dotted path")

        fn = _import_function(func_path)

        # Build kwargs from params (exclude internal keys)
        kwargs = {
            k: v for k, v in params.items()
            if not k.startswith("_") and k != "function"
        }

        t0 = time.monotonic()
        import inspect
        if inspect.iscoroutinefunction(fn):
            result = await fn(**kwargs)
        else:
            result = fn(**kwargs)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Process result into outputs + artifacts
        outputs = {}
        metadata = {}

        if isinstance(result, dict):
            # Store non-data fields as metadata
            for k, v in result.items():
                if k in ("status", "total", "model", "message"):
                    metadata[k] = v

            # Check for base64 data to store as artifact
            for key in _B64_KEYS:
                if key in result and isinstance(result[key], str):
                    try:
                        data = base64.b64decode(result[key])
                        ref = await context.artifacts.store(
                            context.run_id, context.step_id,
                            "output", data,
                            result.get("media_type", "application/octet-stream"),
                        )
                        outputs["output"] = str(ref.file_path)
                    except Exception:
                        outputs["output"] = result[key]
                    break
            else:
                # No base64 data — store the whole result as JSON artifact
                import json
                ref = await context.artifacts.store(
                    context.run_id, context.step_id,
                    "output.json",
                    json.dumps(result, default=str).encode(),
                    "application/json",
                )
                outputs["output"] = str(ref.file_path)

        elif isinstance(result, (str, bytes)):
            data = result if isinstance(result, bytes) else result.encode()
            ref = await context.artifacts.store(
                context.run_id, context.step_id, "output", data,
                "application/octet-stream",
            )
            outputs["output"] = str(ref.file_path)

        return StepResult(outputs=outputs, duration_ms=elapsed_ms, metadata=metadata)


def _ext_for_media_type(media_type: str) -> str:
    mapping = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
        "video/mp4": ".mp4", "video/webm": ".webm",
        "audio/wav": ".wav", "audio/mp3": ".mp3", "audio/ogg": ".ogg",
        "application/json": ".json",
    }
    return mapping.get(media_type, ".bin")
