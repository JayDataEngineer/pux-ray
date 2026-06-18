"""Phase 2 integration tests — parallel execution, step executors."""
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.workflows.spec import load_spec, execution_plan, resolve_templates, WorkflowSpec, StepSpec
from services.workflows.state import WorkflowRun, StepState, RunStore
from services.workflows.artifacts import ArtifactStore
from services.workflows.steps import StepRegistry, StepContext, StepResult, StepExecutor


# ---------------------------------------------------------------------------
# Test parallel execution groups
# ---------------------------------------------------------------------------

def test_parallel_groups():
    """A spec with 3 independent audio steps should have them in one group."""
    spec = WorkflowSpec(
        name="parallel_audio",
        inputs={"prompt": {"type": "string", "required": True}},
        steps=[
            StepSpec(id="voice", type="serve", service="kokoro", depends_on=[]),
            StepSpec(id="sfx", type="forge", service="moss_soundeffect", depends_on=["video"]),
            StepSpec(id="music", type="forge", service="ace_step", depends_on=[]),
            StepSpec(id="video", type="forge", service="z_image", depends_on=[]),
            StepSpec(id="mix", type="compose", method="ffmpeg_mix", depends_on=["voice", "sfx", "music"]),
        ],
    )
    plan = execution_plan(spec)

    # Group 1: video + voice + music (all independent)
    group1_ids = {s.id for s in plan[0]}
    assert "video" in group1_ids
    assert "voice" in group1_ids
    assert "music" in group1_ids

    # Group 2: sfx (depends on video)
    assert plan[1][0].id == "sfx"

    # Group 3: mix (depends on voice, sfx, music)
    assert plan[2][0].id == "mix"

    print(f"PASS: parallel groups — {len(plan)} groups, group1 has {len(plan[0])} parallel steps")


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

def test_registry():
    from services.workflows.steps.serve import ServeStepExecutor
    from services.workflows.steps.compose import ComposeStepExecutor
    from services.workflows.steps.transform import TransformStepExecutor
    from services.workflows.steps.external import ExternalWaitStep
    from services.workflows.steps.forge import ForgeStepExecutor

    registry = StepRegistry()
    registry.register("forge", ForgeStepExecutor)
    registry.register("serve", ServeStepExecutor)
    registry.register("compose", ComposeStepExecutor)
    registry.register("transform", TransformStepExecutor)
    registry.register("external_wait", ExternalWaitStep)

    available = registry.available_types()
    assert "forge" in available
    assert "serve" in available
    assert "compose" in available
    assert "transform" in available
    assert "external_wait" in available

    # Each type returns a fresh executor instance
    exec1 = registry.get("forge")
    exec2 = registry.get("forge")
    assert exec1 is not exec2

    # Unknown type raises
    try:
        registry.get("nonexistent")
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)

    print("PASS: registry")


# ---------------------------------------------------------------------------
# Test compose executor (requires ffmpeg)
# ---------------------------------------------------------------------------

async def test_compose_mix():
    from services.workflows.steps.compose import ComposeStepExecutor

    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(base_dir=Path(tmp))

        # Create two dummy WAV files (valid WAV headers)
        wav1 = Path(tmp) / "track1.wav"
        wav2 = Path(tmp) / "track2.wav"
        _create_silent_wav(wav1, 0.5)
        _create_silent_wav(wav2, 0.5)

        context = StepContext(
            run_id="test",
            step_id="mix",
            artifacts=store,
        )

        executor = ComposeStepExecutor()
        result = await executor.execute(
            {"_method": "ffmpeg_mix", "tracks": [str(wav1), str(wav2)]},
            context,
        )

        assert "output" in result.outputs
        output_path = Path(result.outputs["output"])
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        print(f"PASS: compose mix — output {output_path.stat().st_size} bytes")


async def test_transform_resize():
    from services.workflows.steps.transform import TransformStepExecutor

    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(base_dir=Path(tmp))

        # Create a test image
        from PIL import Image
        img = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
        src = Path(tmp) / "input.png"
        img.save(src)

        context = StepContext(
            run_id="test",
            step_id="resize",
            artifacts=store,
        )

        executor = TransformStepExecutor()
        result = await executor.execute(
            {"_method": "resize", "image": str(src), "width": 512, "height": 512},
            context,
        )

        assert "output" in result.outputs
        output_path = Path(result.outputs["output"])
        assert output_path.exists()
        resized = Image.open(output_path)
        assert resized.size == (512, 512)
        print(f"PASS: transform resize — {resized.size}")


# ---------------------------------------------------------------------------
# Test parallel group execution with engine
# ---------------------------------------------------------------------------

async def test_parallel_engine_execution():
    """Test that the engine executes independent steps concurrently."""
    # Import the undecorated class for testing
    import services.workflows.engine as engine_mod
    # The @serve.deployment wraps the class — get the original via func_or_class
    EngineCls = engine_mod.WorkflowEngine.func_or_class

    with tempfile.TemporaryDirectory() as tmp:
        engine = EngineCls.__new__(EngineCls)
        engine.artifacts = ArtifactStore(base_dir=Path(tmp))
        engine.state_store = RunStore(base_dir=Path(tmp))
        engine.registry = StepRegistry()
        engine._running = {}
        engine._wait_events = {}
        engine._register_executors()

        # Register a fast mock executor for testing
        class MockExecutor(StepExecutor):
            async def execute(self, params, context):
                await asyncio.sleep(0.1)  # Simulate work
                ref = await context.artifacts.store(
                    context.run_id, context.step_id, "output",
                    b"mock data", "application/octet-stream"
                )
                return StepResult(outputs={"output": str(ref.file_path)}, duration_ms=100)

        engine.registry.register("mock", MockExecutor)

        spec = WorkflowSpec(
            name="parallel_test",
            inputs={"prompt": {"type": "string", "required": True}},
            steps=[
                StepSpec(id="a", type="mock", params={"x": "1"}),
                StepSpec(id="b", type="mock", params={"x": "2"}),
                StepSpec(id="c", type="mock", params={"x": "3"}, depends_on=["a", "b"]),
            ],
        )

        run_id = "par_test"
        run = WorkflowRun(
            run_id=run_id,
            spec_name="parallel_test",
            inputs={"prompt": "test"},
            step_states={s.id: StepState(step_id=s.id) for s in spec.steps},
        )
        await engine.state_store.create(run)

        plan = execution_plan(spec)
        assert len(plan) == 2  # [a,b] then [c]

        t0 = time.monotonic()
        # Execute first group (a and b in parallel)
        await engine._execute_group(run, plan[0])
        elapsed = time.monotonic() - t0

        # Two 0.1s steps in parallel should take ~0.1s, not ~0.2s
        assert elapsed < 0.25, f"Parallel group took {elapsed:.2f}s (expected <0.25s)"

        run = await engine.state_store.load(run_id)
        assert run.step_states["a"].status == "completed"
        assert run.step_states["b"].status == "completed"

        print(f"PASS: parallel engine execution — group took {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Test external wait step
# ---------------------------------------------------------------------------

async def test_external_wait():
    from services.workflows.steps.external import ExternalWaitStep

    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(base_dir=Path(tmp))
        context = StepContext(run_id="test", step_id="kimodo", artifacts=store)

        executor = ExternalWaitStep()
        result = await executor.execute(
            {"description": "Upload Kimodo pose", "accepted_types": ["image/png"]},
            context,
        )

        assert result.metadata.get("waiting") is True
        assert result.metadata["description"] == "Upload Kimodo pose"
        print("PASS: external wait step")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_silent_wav(path: Path, duration: float, sample_rate: int = 22050) -> None:
    """Create a minimal valid WAV file with silence."""
    import struct
    n_samples = int(sample_rate * duration)
    data_size = n_samples * 2  # 16-bit mono

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00\x00" * n_samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    test_parallel_groups()
    test_registry()
    await test_compose_mix()
    await test_transform_resize()
    await test_parallel_engine_execution()
    await test_external_wait()
    print("\nAll Phase 2 tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
