"""Workflow state management — run state persistence with asyncio locks.

Stores WorkflowRun state as JSON on PVC. Uses asyncio.Lock per run_id to
prevent race conditions when concurrent requests modify the same run
(e.g., a cancel arriving while a step completes).
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE = Path("/models/workflows")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepState:
    step_id: str
    status: str = "pending"  # pending | running | waiting_input | completed | failed | skipped
    outputs: dict[str, str] = field(default_factory=dict)  # name → artifact key
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    interaction: str | None = None  # required | optional | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkflowRun:
    run_id: str
    spec_name: str
    status: str = "pending"  # pending | running | waiting_input | completed | failed | cancelled
    inputs: dict[str, Any] = field(default_factory=dict)
    step_states: dict[str, StepState] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)  # "{step}.{name}" → ArtifactRef dict
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spec_name": self.spec_name,
            "status": self.status,
            "inputs": self.inputs,
            "step_states": {k: v.to_dict() for k, v in self.step_states.items()},
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowRun:
        step_states = {}
        for k, v in d.get("step_states", {}).items():
            step_states[k] = StepState.from_dict(v) if isinstance(v, dict) else v
        return cls(
            run_id=d["run_id"],
            spec_name=d["spec_name"],
            status=d.get("status", "pending"),
            inputs=d.get("inputs", {}),
            step_states=step_states,
            artifacts=d.get("artifacts", {}),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    def artifact_paths(self) -> dict[str, Any]:
        """Return nested dict for template resolution.

        Returns structure like:
          { "generate_base": { "outputs": { "image": "/path/to/file.png" } } }

        Template expressions like {{ generate_base.outputs.image }} walk this
        via dot-path: context["generate_base"]["outputs"]["image"].
        """
        nested: dict[str, Any] = {}
        for key, ref in self.artifacts.items():
            if isinstance(ref, dict) and "file_path" in ref:
                # key is "{step_id}.{name}"
                parts = key.split(".", 1)
                if len(parts) == 2:
                    step_id, name = parts
                    step_node = nested.setdefault(step_id, {})
                    outputs_node = step_node.setdefault("outputs", {})
                    outputs_node[name] = ref["file_path"]
        return nested


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------

class RunStore:
    """File-based run state persistence with per-run asyncio locks."""

    def __init__(self, base_dir: Path = _DEFAULT_BASE):
        self.base_dir = base_dir
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_creation = asyncio.Lock()

    async def _get_lock(self, run_id: str) -> asyncio.Lock:
        async with self._lock_creation:
            if run_id not in self._locks:
                self._locks[run_id] = asyncio.Lock()
            return self._locks[run_id]

    def _state_path(self, run_id: str) -> Path:
        return self.base_dir / run_id / "state.json"

    async def create(self, run: WorkflowRun) -> None:
        """Create a new run state file. Writes initial state to disk."""
        run.created_at = _now_iso()
        run.updated_at = run.created_at
        path = self._state_path(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = await self._get_lock(run.run_id)
        async with lock:
            _atomic_write(path, json.dumps(run.to_dict(), indent=2))

    async def save(self, run: WorkflowRun) -> None:
        """Persist current run state. Acquires per-run lock."""
        run.updated_at = _now_iso()
        path = self._state_path(run.run_id)
        lock = await self._get_lock(run.run_id)
        async with lock:
            _atomic_write(path, json.dumps(run.to_dict(), indent=2))

    async def load(self, run_id: str) -> WorkflowRun | None:
        """Load run state from disk."""
        path = self._state_path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return WorkflowRun.from_dict(data)

    async def load_locked(self, run_id: str) -> WorkflowRun | None:
        """Load with lock held (for read-modify-write patterns)."""
        lock = await self._get_lock(run_id)
        async with lock:
            return await self.load(run_id)

    def list_runs(self, spec_name: str | None = None) -> list[str]:
        """List run IDs, optionally filtered by spec name."""
        if not self.base_dir.exists():
            return []
        runs = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and (d / "state.json").exists():
                if spec_name:
                    try:
                        data = json.loads((d / "state.json").read_text())
                        if data.get("spec_name") == spec_name:
                            runs.append(d.name)
                    except (json.JSONDecodeError, KeyError):
                        continue
                else:
                    runs.append(d.name)
        return runs

    async def update_step(self, run_id: str, step_id: str, **updates: Any) -> WorkflowRun | None:
        """Atomic update of a single step's state. Acquires per-run lock."""
        lock = await self._get_lock(run_id)
        async with lock:
            run = await self.load(run_id)
            if run is None:
                return None
            ss = run.step_states.get(step_id)
            if ss is None:
                return run
            for k, v in updates.items():
                if hasattr(ss, k):
                    setattr(ss, k, v)
            run.updated_at = _now_iso()
            _atomic_write(self._state_path(run_id), json.dumps(run.to_dict(), indent=2))
            return run

    def cleanup_lock(self, run_id: str) -> None:
        """Remove in-memory lock for a completed run."""
        self._locks.pop(run_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, content: str) -> None:
    """Write to temp file then rename — prevents partial writes on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w") as f:
            f.write(content)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
