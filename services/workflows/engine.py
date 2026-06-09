"""Workflow engine — Ray Serve deployment that orchestrates DAG execution.

CPU-only deployment (num_gpus=0) isolated from GPU crashes. Manages run state,
resolves template expressions, dispatches steps to executors, and streams
SSE events for frontend consumption.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from ray import serve

from .artifacts import ArtifactStore
from .spec import WorkflowSpec, StepSpec, load_spec, execution_plan, resolve_templates
from .state import WorkflowRun, StepState, RunStore
from .steps import StepContext, StepResult, StepRegistry
from .steps.forge import ForgeStepExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

def _event(event_type: str, **data: Any) -> dict[str, Any]:
    return {"event": event_type, **data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Engine deployment
# ---------------------------------------------------------------------------

@serve.deployment(
    name="workflow_engine",
    max_ongoing_requests=10,
    ray_actor_options={"num_gpus": 0},
)
class WorkflowEngine:
    """Orchestrates workflow execution. CPU-only, survives GPU crashes."""

    def __init__(self):
        self.artifacts = ArtifactStore()
        self.state_store = RunStore()
        self.registry = StepRegistry()
        self._running: dict[str, asyncio.Task] = {}
        self._wait_events: dict[str, asyncio.Event] = {}
        self._register_executors()

    def _register_executors(self) -> None:
        from .steps.serve import ServeStepExecutor
        from .steps.compose import ComposeStepExecutor
        from .steps.transform import TransformStepExecutor
        from .steps.external import ExternalWaitStep
        from .steps.python import PythonStepExecutor
        from .steps.mock import MockStepExecutor
        from .steps.ltx_video import LTXGenerateStep, LTXSpatialUpscaleStep

        self.registry.register("forge", ForgeStepExecutor)
        self.registry.register("serve", ServeStepExecutor)
        self.registry.register("compose", ComposeStepExecutor)
        self.registry.register("transform", TransformStepExecutor)
        self.registry.register("external_wait", ExternalWaitStep)
        self.registry.register("python", PythonStepExecutor)
        self.registry.register("mock", MockStepExecutor)
        self.registry.register("ltx_generate", LTXGenerateStep)
        self.registry.register("ltx_upscale", LTXSpatialUpscaleStep)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def start_run(self, spec_name: str, inputs: dict, manual: bool = False,
                        skip_review: bool = False) -> dict:
        """Create a new workflow run. If manual=True, don't auto-execute steps.
        If skip_review=True, skip review pauses (for programmatic API calls)."""
        spec = load_spec(spec_name)
        self._validate_inputs(spec, inputs)

        run_id = uuid.uuid4().hex[:12]
        run = WorkflowRun(
            run_id=run_id,
            spec_name=spec_name,
            inputs=inputs,
            step_states={s.id: StepState(step_id=s.id) for s in spec.steps},
        )
        if skip_review:
            self._skip_review_runs: set[str] = getattr(self, '_skip_review_runs', set())
            self._skip_review_runs.add(run_id)
        await self.state_store.create(run)

        if not manual:
            # Start execution in background
            task = asyncio.create_task(self._run_workflow(run_id))
            self._running[run_id] = task

        status = "pending" if manual else "running"
        return {"run_id": run_id, "status": status, "spec": spec_name}

    async def get_run(self, run_id: str) -> dict | None:
        run = await self.state_store.load(run_id)
        return run.to_dict() if run else None

    async def cancel_run(self, run_id: str) -> dict:
        task = self._running.get(run_id)
        if task and not task.done():
            task.cancel()
            run = await self.state_store.load(run_id)
            if run:
                run.status = "cancelled"
                await self.state_store.save(run)
            return {"run_id": run_id, "status": "cancelled"}
        return {"run_id": run_id, "status": "not_running"}

    async def approve_step(self, run_id: str, step_id: str, data: dict) -> dict:
        """Provide user input for a waiting step (e.g., Kimodo upload)."""
        run = await self.state_store.load(run_id)
        if not run:
            return {"error": "Run not found"}

        ss = run.step_states.get(step_id)
        if not ss or ss.status != "waiting_input":
            return {"error": f"Step '{step_id}' is not waiting for input"}

        # Store uploaded data as artifact
        file_data = data.get("file_data")
        file_name = data.get("name", "upload")
        media_type = data.get("media_type", "application/octet-stream")

        if file_data:
            import base64
            if isinstance(file_data, str):
                file_data = base64.b64decode(file_data)
            artifact = await self.artifacts.store(
                run_id, step_id, file_name, file_data, media_type
            )
            run.artifacts[f"{step_id}.{file_name}"] = artifact.to_dict()
            ss.outputs[file_name] = str(artifact.file_path)

        ss.status = "completed"
        await self.state_store.save(run)

        # Signal the waiting step to resume
        event_key = f"{run_id}.{step_id}"
        evt = self._wait_events.get(event_key)
        if evt:
            evt.set()

        return {"status": "ok", "step_id": step_id}

    async def rerun_from(self, run_id: str, step_id: str, new_params: dict | None = None) -> dict:
        """Re-execute from a specific step, invalidating downstream artifacts."""
        run = await self.state_store.load(run_id)
        if not run:
            return {"error": "Run not found"}

        spec = load_spec(run.spec_name)
        plan = execution_plan(spec)

        # Find steps to invalidate (the target step + everything downstream)
        to_invalidate = self._downstream_steps(spec, step_id)
        to_invalidate.add(step_id)

        for sid in to_invalidate:
            ss = run.step_states.get(sid)
            if ss:
                ss.status = "pending"
                ss.error = None
                ss.outputs = {}
                ss.duration_ms = None
            # Remove artifacts for invalidated steps
            keys_to_remove = [k for k in run.artifacts if k.startswith(f"{sid}.")]
            for k in keys_to_remove:
                del run.artifacts[k]

        run.status = "running"
        await self.state_store.save(run)

        # Start re-execution
        task = asyncio.create_task(self._run_workflow(run_id, from_step=step_id))
        self._running[run_id] = task

        return {"run_id": run_id, "status": "running", "from_step": step_id}

    async def execute_single_step(self, run_id: str, step_id: str, params_override: dict | None = None) -> dict:
        """Execute a single step in isolation without affecting downstream steps.

        Used by the video editor frontend when the user clicks "regenerate"
        on one specific step (e.g., re-generate the character without
        re-running the entire pipeline).
        """
        run = await self.state_store.load(run_id)
        if not run:
            return {"error": "Run not found"}

        spec = load_spec(run.spec_name)
        step = next((s for s in spec.steps if s.id == step_id), None)
        if not step:
            return {"error": f"Step '{step_id}' not found in spec"}

        # Check dependencies are satisfied
        for dep in step.depends_on:
            dep_state = run.step_states.get(dep)
            if not dep_state or dep_state.status != "completed":
                return {"error": f"Dependency '{dep}' not yet completed"}

        # Apply param overrides if provided
        if params_override:
            step = StepSpec(
                id=step.id,
                type=step.type,
                service=step.service or params_override.get("service"),
                model=params_override.get("model", step.model),
                method=step.method,
                depends_on=step.depends_on,
                params={**step.params, **params_override.get("params", {})},
                outputs=step.outputs,
                interaction=step.interaction,
            )

        # Reset this step's state
        run.step_states[step_id] = StepState(step_id=step_id)
        # Remove old artifacts for this step
        keys_to_remove = [k for k in run.artifacts if k.startswith(f"{step_id}.")]
        for k in keys_to_remove:
            del run.artifacts[k]
        await self.state_store.save(run)

        # Execute just this step (skip review pause so it completes immediately)
        try:
            await self._execute_step(run, step, skip_review=True)
            run = await self.state_store.load(run_id)
            ss = run.step_states.get(step_id)

            # Update run status if all steps completed
            all_done = all(
                run.step_states.get(s.id).status in ("completed", "skipped")
                for s in spec.steps
                if run.step_states.get(s.id)
            )
            if all_done:
                run.status = "completed"
                await self.state_store.save(run)

            return {
                "run_id": run_id,
                "step_id": step_id,
                "status": ss.status if ss else "unknown",
                "duration_ms": ss.duration_ms if ss else None,
                "outputs": ss.outputs if ss else {},
            }
        except Exception as e:
            return {"run_id": run_id, "step_id": step_id, "status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # SSE event streaming
    # ------------------------------------------------------------------

    async def stream_events(self, run_id: str) -> AsyncIterator[dict]:
        """Yield SSE events for a running workflow.

        Polls state changes and yields events. In production, this would
        use a proper pub/sub (Redis, Ray Actor), but polling state.json
        works for Phase 1.
        """
        yield _event("connected", run_id=run_id)

        last_status = {}
        while True:
            run = await self.state_store.load(run_id)
            if not run:
                yield _event("error", message="Run not found")
                return

            # Check for state changes since last poll
            for sid, ss in run.step_states.items():
                prev = last_status.get(sid)
                curr = ss.status
                if prev != curr:
                    if curr == "running":
                        yield _event("step_started", step_id=sid)
                    elif curr == "completed":
                        yield _event(
                            "step_completed",
                            step_id=sid,
                            duration_ms=ss.duration_ms,
                            outputs=ss.outputs,
                        )
                    elif curr == "failed":
                        yield _event("step_failed", step_id=sid, error=ss.error)
                    elif curr == "waiting_input":
                        yield _event("step_waiting", step_id=sid, message="Waiting for user input")
                    last_status[sid] = curr

            if run.status in ("completed", "failed", "cancelled"):
                yield _event(
                    f"workflow_{run.status}",
                    run_id=run_id,
                    artifacts=list(run.artifacts.keys()),
                )
                return

            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run_workflow(self, run_id: str, from_step: str | None = None) -> None:
        """Execute a workflow run."""
        run = await self.state_store.load(run_id)
        if not run:
            logger.error("Run %s not found", run_id)
            return

        spec = load_spec(run.spec_name)
        plan = execution_plan(spec)

        run.status = "running"
        await self.state_store.save(run)

        try:
            skip = from_step is not None
            for group in plan:
                # Skip groups until we reach the from_step
                if skip:
                    step_ids = {s.id for s in group}
                    if from_step not in step_ids:
                        continue
                    # Found the group containing from_step — execute only from that step onward
                    skip = False
                    # Execute this group (all steps in it, since parallel group)
                    await self._execute_group(run, group)
                else:
                    await self._execute_group(run, group)

            run = await self.state_store.load(run_id)
            if run:
                all_done = all(
                    ss.status in ("completed", "skipped")
                    for ss in run.step_states.values()
                )
                if all_done:
                    run.status = "completed"
                    await self.state_store.save(run)

        except asyncio.CancelledError:
            logger.info("Run %s cancelled", run_id)
        except Exception as e:
            logger.exception("Run %s failed", run_id)
            run = await self.state_store.load(run_id)
            if run:
                run.status = "failed"
                await self.state_store.save(run)
        finally:
            self._running.pop(run_id, None)
            self._cleanup_wait_events(run_id)
            _skip_set = getattr(self, '_skip_review_runs', set())
            _skip_set.discard(run_id)

    async def _execute_group(self, run: WorkflowRun, group: list[StepSpec]) -> None:
        """Execute a group of steps concurrently via asyncio.gather.

        Steps within a group have no dependencies on each other. GPU steps
        will naturally serialize through the Forge's VRAM lock; CPU steps
        run freely in parallel.
        """
        pending = []
        for step in group:
            ss = run.step_states.get(step.id)
            if ss and ss.status == "completed":
                continue  # Already done (e.g., from previous partial run)
            pending.append(step)

        if not pending:
            return

        if len(pending) == 1:
            await self._execute_step(run, pending[0])
            return

        # Run all pending steps concurrently
        tasks = [self._execute_step(run, step) for step in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Propagate first error (group fails if any step fails)
        for step, result in zip(pending, results):
            if isinstance(result, Exception):
                raise result

    async def _execute_step(self, run: WorkflowRun, step: StepSpec, *, skip_review: bool = False) -> None:
        """Execute a single step: resolve params → pick executor → run → store."""
        run_id = run.run_id

        await self.state_store.update_step(run_id, step.id, status="running", started_at=_now())

        try:
            executor = self.registry.get(step.type)

            # Resolve {{ }} template expressions
            params = dict(step.params)
            if step.service:
                params["_service"] = step.service
            if step.model:
                params["_model"] = step.model
            if step.method:
                params["_method"] = step.method
            if step.function:
                params["_function"] = step.function

            # Reload run to get latest artifact paths
            run = await self.state_store.load(run_id)
            resolved = resolve_templates(params, run.inputs, run.artifact_paths())

            context = StepContext(
                run_id=run_id,
                step_id=step.id,
                artifacts=self.artifacts,
            )

            # Handle interaction steps
            if step.type == "external_wait" or step.interaction == "required":
                await self.state_store.update_step(run_id, step.id, status="waiting_input")
                # Wait for approve_step() to signal
                event_key = f"{run_id}.{step.id}"
                evt = asyncio.Event()
                self._wait_events[event_key] = evt
                await evt.wait()
                # Step was approved, mark completed
                run = await self.state_store.load(run_id)
                return

            if step.interaction == "optional":
                # Run normally but tag as interactive so frontend can offer re-run
                await self.state_store.update_step(
                    run_id, step.id,
                    status="running",
                    error=None,  # clear any previous metadata key
                )

            t0 = time.monotonic()
            result: StepResult = await executor.execute(resolved, context)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            # Record artifact refs from executor outputs into run state
            run = await self.state_store.load(run_id)
            for name, path in result.outputs.items():
                if not isinstance(path, (str, Path)):
                    continue
                try:
                    artifact_path = Path(path)
                    if artifact_path.exists():
                        ref = await self.artifacts.store_from_file(
                            run_id, step.id, name, artifact_path,
                        )
                        run.artifacts[f"{step.id}.{name}"] = ref.to_dict()
                except OSError:
                    continue  # Not a valid file path (e.g., base64 data)
            await self.state_store.save(run)

            update_kwargs = dict(
                status="completed",
                outputs=result.outputs,
                duration_ms=elapsed_ms,
                completed_at=_now(),
            )
            if step.interaction:
                update_kwargs["interaction"] = step.interaction
            await self.state_store.update_step(run_id, step.id, **update_kwargs)

            # Review mode: pause after every completed step for user approval.
            # Skip for steps that already manage their own interaction
            # (external_wait steps pause before execution, not after).
            # Also skip when called from execute_single_step (skip_review=True).
            # Also skip when run was started with skip_review flag (API/programmatic).
            _skip_set = getattr(self, '_skip_review_runs', set())
            run_skip = run_id in _skip_set
            if step.type != "external_wait" and not skip_review and not run_skip:
                await self.state_store.update_step(
                    run_id, step.id, status="waiting_input", interaction="review",
                )
                event_key = f"{run_id}.{step.id}"
                evt = asyncio.Event()
                self._wait_events[event_key] = evt
                await evt.wait()
                await self.state_store.update_step(run_id, step.id, status="completed")

        except Exception as e:
            logger.exception("Step %s failed", step.id)
            await self.state_store.update_step(
                run_id, step.id,
                status="failed",
                error=str(e),
                completed_at=_now(),
            )
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, spec: WorkflowSpec, inputs: dict) -> None:
        # Fill in defaults for missing optional inputs
        for name, ispec in spec.inputs.items():
            if name not in inputs and ispec.default is not None:
                inputs[name] = ispec.default
        # Validate required + enum
        for name, ispec in spec.inputs.items():
            if ispec.required and name not in inputs and ispec.default is None:
                raise ValueError(f"Missing required input: {name}")
            if name in inputs and ispec.enum and inputs[name] not in ispec.enum:
                raise ValueError(f"Input '{name}' must be one of {ispec.enum}")

    def _downstream_steps(self, spec: WorkflowSpec, step_id: str) -> set[str]:
        """Find all steps that depend (transitively) on step_id."""
        downstream = set()
        for s in spec.steps:
            if step_id in s.depends_on:
                downstream.add(s.id)
                downstream |= self._downstream_steps(spec, s.id)
        return downstream

    def _cleanup_wait_events(self, run_id: str) -> None:
        keys = [k for k in self._wait_events if k.startswith(f"{run_id}.")]
        for k in keys:
            self._wait_events.pop(k, None)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
