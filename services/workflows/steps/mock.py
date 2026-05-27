"""Mock step executor — returns a dummy artifact for testing.

Used by _test_spec.yaml and unit tests. Produces a small text file as output
so the rest of the engine pipeline (template resolution, artifact storage,
state transitions) can be exercised without real GPU services.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import StepExecutor, StepContext, StepResult


class MockStepExecutor(StepExecutor):
    """Return a placeholder artifact for testing."""

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        t0 = time.monotonic()

        data = f"mock output for step {context.step_id}".encode()
        artifact = await context.artifacts.store(
            context.run_id,
            context.step_id,
            "output",
            data,
            "application/octet-stream",
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return StepResult(
            outputs={"output": str(artifact.file_path)},
            duration_ms=elapsed_ms,
        )
