"""Workflow specification — Pydantic models + YAML loader.

Parses declarative workflow specs from config/workflows/*.yaml into validated
Python objects. Computes parallel execution groups via topological sort.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SPEC_DIR = Path(__file__).resolve().parents[2] / "config" / "workflows"

_TEMPLATE_PATTERN = re.compile(r"\{\{(.+?)\}\}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class InputSpec(BaseModel):
    type: str = "string"
    required: bool = True
    default: Any = None
    description: str = ""
    enum: list[str] | None = None


class OutputSpec(BaseModel):
    media_type: str


class StepSpec(BaseModel):
    id: str = Field(..., alias="id")
    type: str
    service: str | None = None
    model: str | None = None
    method: str | None = None
    function: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    interaction: str | None = None  # required | optional | None

    class Config:
        populate_by_name = True


class WorkflowSpec(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    inputs: dict[str, InputSpec]
    steps: list[StepSpec]


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_spec(name: str, spec_dir: Path | None = None) -> WorkflowSpec:
    """Load and validate a workflow spec from YAML.

    Searches for {name}.yaml or {name}.yml in spec_dir (default: config/workflows/).
    """
    base = spec_dir or _SPEC_DIR
    for ext in (".yaml", ".yml"):
        path = base / f"{name}{ext}"
        if path.exists():
            return _parse_file(path)
    raise FileNotFoundError(f"No workflow spec '{name}' in {base}")


def list_specs(spec_dir: Path | None = None) -> list[str]:
    """Return available workflow spec names."""
    base = spec_dir or _SPEC_DIR
    names = []
    for p in sorted(base.glob("*.yaml")):
        names.append(p.stem)
    for p in sorted(base.glob("*.yml")):
        if p.stem not in names:
            names.append(p.stem)
    return names


def _parse_file(path: Path) -> WorkflowSpec:
    with open(path) as f:
        raw = yaml.safe_load(f)

    steps = []
    for step_id, step_data in raw.get("steps", {}).items():
        steps.append(StepSpec(id=step_id, **step_data))

    spec = WorkflowSpec(
        name=raw["name"],
        version=raw.get("version", "1.0"),
        description=raw.get("description", ""),
        inputs=raw.get("inputs", {}),
        steps=steps,
    )
    _validate(spec)
    return spec


def _validate(spec: WorkflowSpec) -> None:
    """Validate spec integrity: unique IDs, valid deps, no cycles, template refs."""
    ids = {s.id for s in spec.steps}
    if len(ids) != len(spec.steps):
        seen = set()
        for s in spec.steps:
            if s.id in seen:
                raise ValueError(f"Duplicate step id: {s.id}")
            seen.add(s.id)

    step_outputs: dict[str, set[str]] = {}
    for s in spec.steps:
        step_outputs[s.id] = set(s.outputs.keys())

    for s in spec.steps:
        for dep in s.depends_on:
            if dep not in ids:
                raise ValueError(
                    f"Step '{s.id}' depends on '{dep}', which doesn't exist"
                )

    _validate_template_refs(spec, step_outputs)

    # Cycle detection via Kahn's algorithm
    _execution_plan(spec)


def _validate_template_refs(
    spec: WorkflowSpec, step_outputs: dict[str, set[str]]
) -> None:
    """Verify all {{ }} references point to real inputs or step outputs."""
    for step in spec.steps:
        for ref in _collect_refs(step.params):
            parts = ref.strip().split(".")
            if parts[0] == "inputs" and len(parts) == 2:
                if parts[1] not in spec.inputs:
                    raise ValueError(
                        f"Step '{step.id}' references inputs.{parts[1]}, "
                        f"which is not defined"
                    )
            elif len(parts) >= 3 and parts[1] == "outputs":
                dep_id = parts[0]
                if dep_id not in step_outputs:
                    raise ValueError(
                        f"Step '{step.id}' references {dep_id}.outputs, "
                        f"but step '{dep_id}' doesn't exist"
                    )
                if len(parts) >= 3:
                    out_name = parts[2]
                    if out_name not in step_outputs[dep_id]:
                        raise ValueError(
                            f"Step '{step.id}' references {dep_id}.outputs.{out_name}, "
                            f"but step '{dep_id}' doesn't declare that output "
                            f"(available: {sorted(step_outputs[dep_id])})"
                        )
                if dep_id not in step.depends_on:
                    raise ValueError(
                        f"Step '{step.id}' references {dep_id}.outputs but "
                        f"doesn't declare depends_on: [{dep_id}]"
                    )


def _collect_refs(params: dict[str, Any]) -> list[str]:
    """Extract all {{ }} references from params dict."""
    refs: list[str] = []
    _collect_refs_walk(params, refs)
    return refs


def _collect_refs_walk(obj: Any, refs: list[str]) -> None:
    if isinstance(obj, str):
        for m in _TEMPLATE_PATTERN.finditer(obj):
            refs.append(m.group(1))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_refs_walk(v, refs)
    elif isinstance(obj, list):
        for v in obj:
            _collect_refs_walk(v, refs)


# ---------------------------------------------------------------------------
# Execution planning
# ---------------------------------------------------------------------------

def execution_plan(spec: WorkflowSpec) -> list[list[StepSpec]]:
    """Return parallel groups. Each inner list contains steps that can run
    concurrently. Groups run sequentially."""
    return _execution_plan(spec)


def _execution_plan(spec: WorkflowSpec) -> list[list[StepSpec]]:
    in_degree: dict[str, int] = {s.id: len(s.depends_on) for s in spec.steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in spec.steps}
    step_map: dict[str, StepSpec] = {s.id: s for s in spec.steps}

    for s in spec.steps:
        for dep in s.depends_on:
            dependents[dep].append(s.id)

    groups: list[list[StepSpec]] = []
    ready = [sid for sid, deg in in_degree.items() if deg == 0]
    visited = 0

    while ready:
        group = [step_map[sid] for sid in sorted(ready)]
        groups.append(group)
        visited += len(ready)
        next_ready = []
        for sid in ready:
            for child in dependents[sid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ready.append(child)
        ready = next_ready

    if visited != len(spec.steps):
        raise ValueError("Workflow has cyclic dependencies")

    return groups


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

def resolve_templates(
    params: dict[str, Any],
    inputs: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    """Resolve {{ }} template expressions in params.

    Supported references:
      - {{ inputs.field_name }}  → value from user inputs
      - {{ step_id.outputs.name }}  → file path from a previous step's artifact
    """
    context = {"inputs": inputs, **artifact_paths}
    return _walk_params(params, context)


def _walk_params(params: Any, context: dict) -> Any:
    if isinstance(params, str):
        return _resolve_str(params, context)
    if isinstance(params, dict):
        return {k: _walk_params(v, context) for k, v in params.items()}
    if isinstance(params, list):
        return [_walk_params(v, context) for v in params]
    return params


def _resolve_str(value: str, context: dict) -> Any:
    refs = _TEMPLATE_PATTERN.findall(value)
    if not refs:
        return value

    if len(refs) == 1 and value.strip() == "{{" + refs[0] + "}}":
        return _walk(context, refs[0].strip())

    def replacer(m):
        resolved = _walk(context, m.group(1).strip())
        return str(resolved) if resolved is not None else m.group(0)

    return _TEMPLATE_PATTERN.sub(replacer, value)


def _walk(data: dict, path: str) -> Any:
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current
