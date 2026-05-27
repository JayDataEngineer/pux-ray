"""Step executor base class and registry."""
from __future__ import annotations

from typing import Any, Callable


class StepContext:
    """Context provided to every step executor."""

    __slots__ = ("run_id", "step_id", "artifacts", "config", "emit_event")

    def __init__(
        self,
        run_id: str,
        step_id: str,
        artifacts: Any,
        config: dict | None = None,
        emit_event: Callable | None = None,
    ):
        self.run_id = run_id
        self.step_id = step_id
        self.artifacts = artifacts
        self.config = config or {}
        self.emit_event = emit_event  # async callable for SSE events


class StepResult:
    """Returned by a step executor after successful execution."""

    __slots__ = ("outputs", "duration_ms", "metadata")

    def __init__(
        self,
        outputs: dict[str, Any] | None = None,
        duration_ms: int = 0,
        metadata: dict | None = None,
    ):
        self.outputs = outputs or {}
        self.duration_ms = duration_ms
        self.metadata = metadata or {}


class StepExecutor:
    """Base class for step executors.

    Subclass and implement execute(). Register with StepRegistry.
    """

    async def execute(self, params: dict, context: StepContext) -> StepResult:
        raise NotImplementedError

    def validate_params(self, params: dict) -> None:
        pass


class StepRegistry:
    """Maps step type strings to StepExecutor subclasses."""

    def __init__(self) -> None:
        self._types: dict[str, type[StepExecutor]] = {}

    def register(self, step_type: str, executor_cls: type[StepExecutor]) -> None:
        self._types[step_type] = executor_cls

    def get(self, step_type: str) -> StepExecutor:
        cls = self._types.get(step_type)
        if cls is None:
            raise ValueError(f"Unknown step type: {step_type}")
        return cls()

    def available_types(self) -> list[str]:
        return sorted(self._types.keys())
