"""Pipeline executor — multi-step inference DAG with output chaining.

Accepts a pipeline spec (list of steps with dependencies), resolves output
references between steps, and executes sequentially. Yields SSE events for
progress streaming.

No Ray dependencies — pure async logic. The caller provides a dispatch
function that routes to the actual service backends.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r"\{(\w+(?:\.\w+)*)\}")


@dataclass
class PipelineStep:
    name: str
    service: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PipelineSpec:
    steps: list[PipelineStep]

    @classmethod
    def from_dict(cls, data: dict) -> PipelineSpec:
        raw_steps = data.get("steps", [])
        if not raw_steps:
            raise ValueError("Pipeline must have at least one step")
        steps = []
        for s in raw_steps:
            name = s.get("name")
            service = s.get("service")
            if not name:
                raise ValueError("Every step must have a 'name'")
            if not service:
                raise ValueError(f"Step '{name}' must have a 'service'")
            steps.append(PipelineStep(
                name=name,
                service=service,
                params=s.get("params", {}),
                depends_on=s.get("depends_on", []),
            ))
        spec = cls(steps=steps)
        spec.validate()
        return spec

    def validate(self):
        names = set()
        for step in self.steps:
            if step.name in names:
                raise ValueError(f"Duplicate step name: {step.name}")
            names.add(step.name)

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in names:
                    raise ValueError(
                        f"Step '{step.name}' depends on '{dep}', which doesn't exist"
                    )

        # Cycle detection via topological sort (Kahn's algorithm)
        in_degree = {s.name: len(s.depends_on) for s in self.steps}
        queue = [s.name for s in self.steps if in_degree[s.name] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for s in self.steps:
                if node in s.depends_on:
                    in_degree[s.name] -= 1
                    if in_degree[s.name] == 0:
                        queue.append(s.name)
        if visited != len(self.steps):
            raise ValueError("Pipeline has cyclic dependencies")

    def execution_order(self) -> list[PipelineStep]:
        """Return steps in topological order."""
        in_degree = {s.name: len(s.depends_on) for s in self.steps}
        step_map = {s.name: s for s in self.steps}
        queue = [s.name for s in self.steps if in_degree[s.name] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(step_map[node])
            for s in self.steps:
                if node in s.depends_on:
                    in_degree[s.name] -= 1
                    if in_degree[s.name] == 0:
                        queue.append(s.name)
        return order


def resolve_refs(params: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Replace {stepname.output.field} references with actual values.

    Walks every string value in params. If it matches a reference pattern,
    resolves it by walking into the results dict via dot-path.
    Non-string values are left as-is.
    """
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            resolved[key] = _resolve_value(value, results)
        elif isinstance(value, dict):
            resolved[key] = resolve_refs(value, results)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_value(v, results) if isinstance(v, str) else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def _resolve_value(value: str, results: dict[str, Any]) -> Any:
    """Resolve a single string value that may contain {step.path} refs."""
    refs = _REF_PATTERN.findall(value)
    if not refs:
        return value

    if len(refs) == 1 and value == f"{{{refs[0]}}}":
        # Exact match — return the resolved value directly (preserves type)
        return _walk(results, refs[0])

    # Template with surrounding text — substitute as strings
    def replacer(m):
        path = m.group(1)
        resolved = _walk(results, path)
        return str(resolved) if resolved is not None else m.group(0)
    return _REF_PATTERN.sub(replacer, value)


def _walk(data: dict, path: str) -> Any:
    """Walk into a nested dict using a dot-separated path like 'output.data'."""
    # First segment is the step name
    parts = path.split(".")
    current = data.get(parts[0])
    for part in parts[1:]:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


# Type alias for the dispatch function
DispatchFn = Callable[[str, dict], Awaitable[dict]]


async def execute_pipeline(
    spec: PipelineSpec,
    dispatch: DispatchFn,
) -> list[dict]:
    """Execute a pipeline spec sequentially, resolving refs between steps.

    Args:
        spec: Validated PipelineSpec.
        dispatch: Async function(service_name, params) -> result_dict.

    Returns:
        List of SSE event dicts in order.
    """
    events: list[dict] = []
    results: dict[str, Any] = {}
    order = spec.execution_order()

    events.append({"event": "pipeline_started", "total_steps": len(order)})

    for step in order:
        t0 = time.time()
        events.append({
            "event": "step_started",
            "step": step.name,
            "service": step.service,
        })

        try:
            resolved_params = resolve_refs(step.params, results)
            result = await dispatch(step.service, resolved_params)
            elapsed = round(time.time() - t0, 2)
            results[step.name] = result

            events.append({
                "event": "step_completed",
                "step": step.name,
                "elapsed_s": elapsed,
                "output_keys": list(result.keys()) if isinstance(result, dict) else [],
            })
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            logger.exception("Pipeline step '%s' failed", step.name)
            events.append({
                "event": "step_error",
                "step": step.name,
                "error": str(e),
                "elapsed_s": elapsed,
            })
            events.append({
                "event": "pipeline_error",
                "failed_step": step.name,
                "error": str(e),
            })
            return events

    events.append({"event": "pipeline_completed", "results": results})
    return events
