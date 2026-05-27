"""External wait step — blocks until user provides data (e.g., Kimodo upload).

Emits a step_waiting SSE event and blocks on an asyncio.Event. The frontend
POSTs file data to the approve endpoint, which stores the artifact and
signals the event to resume execution.
"""
from __future__ import annotations

import logging
from typing import Any

from . import StepExecutor, StepContext, StepResult

logger = logging.getLogger(__name__)


class ExternalWaitStep(StepExecutor):
    """Wait for external user input (file upload, tool output).

    This executor does NOT execute anything itself. It signals that the
    step is waiting for input and returns immediately. The engine handles
    the asyncio.Event blocking in _execute_step().

    The actual data upload happens through the approve_step() API endpoint,
    which stores the artifact and signals the event.
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        # The engine's _execute_step() handles the actual waiting.
        # This executor just validates params and returns metadata.
        description = params.get("description", "Waiting for external input")
        accepted_types = params.get("accepted_types", ["application/octet-stream"])

        logger.info(
            "Step %s waiting for external input: %s (accepted: %s)",
            context.step_id, description, accepted_types,
        )

        return StepResult(
            outputs={},
            metadata={
                "waiting": True,
                "description": description,
                "accepted_types": accepted_types,
            },
        )
