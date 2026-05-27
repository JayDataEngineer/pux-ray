"""Test state persistence and artifact store."""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.workflows.state import WorkflowRun, StepState, RunStore
from services.workflows.artifacts import ArtifactStore


async def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(base_dir=Path(tmp))

        run = WorkflowRun(
            run_id="test123",
            spec_name="character_sheet",
            inputs={"prompt": "cyberpunk warrior"},
            step_states={
                "generate_base": StepState(step_id="generate_base", status="completed"),
                "refine": StepState(step_id="refine", status="pending"),
            },
        )

        await store.create(run)

        loaded = await store.load("test123")
        assert loaded is not None
        assert loaded.run_id == "test123"
        assert loaded.spec_name == "character_sheet"
        assert loaded.step_states["generate_base"].status == "completed"
        assert loaded.step_states["refine"].status == "pending"
        print("PASS: state roundtrip")


async def test_state_lock():
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(base_dir=Path(tmp))
        run = WorkflowRun(run_id="lock_test", spec_name="test")
        await store.create(run)

        # Concurrent updates should serialize via lock
        async def update(n):
            await store.update_step("lock_test", "step_a", status="completed", duration_ms=n * 100)

        # Create the step first
        run = await store.load("lock_test")
        run.step_states["step_a"] = StepState(step_id="step_a")
        await store.save(run)

        await asyncio.gather(update(1), update(2), update(3))

        loaded = await store.load("lock_test")
        assert loaded.step_states["step_a"].status == "completed"
        print("PASS: state lock")


async def test_artifact_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(base_dir=Path(tmp), url_prefix="/v1/workflows")

        ref = await store.store(
            run_id="run1",
            step_id="gen",
            name="output",
            data=b"fake image data",
            media_type="image/png",
        )

        assert ref.run_id == "run1"
        assert ref.step_id == "gen"
        assert ref.media_type == "image/png"
        assert ref.url == "/v1/workflows/runs/run1/artifacts/gen/output.png"
        assert Path(ref.file_path).exists()
        assert Path(ref.file_path).read_bytes() == b"fake image data"

        # Test ref serialization
        d = ref.to_dict()
        ref2 = type(ref).from_dict(d)
        assert ref2.run_id == ref.run_id
        assert ref2.file_path == ref.file_path
        print("PASS: artifact store")


async def test_artifact_paths_nested():
    from services.workflows.state import WorkflowRun, StepState

    run = WorkflowRun(
        run_id="test",
        spec_name="test",
        artifacts={
            "generate_base.output": {
                "file_path": "/mnt/data/runs/test/gen/output.png",
                "media_type": "image/png",
                "run_id": "test",
                "step_id": "generate_base",
                "name": "output",
                "url": "/v1/workflows/test/artifacts/gen/output.png",
                "size_bytes": 100,
                "created_at": "2026-01-01T00:00:00Z",
            }
        },
    )

    paths = run.artifact_paths()
    assert paths == {"generate_base": {"outputs": {"output": "/mnt/data/runs/test/gen/output.png"}}}
    print("PASS: artifact_paths nested structure")


async def main():
    await test_state_roundtrip()
    await test_state_lock()
    await test_artifact_store()
    await test_artifact_paths_nested()
    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
