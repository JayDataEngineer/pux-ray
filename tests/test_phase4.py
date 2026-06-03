"""Phase 4 integration tests — spec validation, YAML migration, python executor."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.workflows.spec import (
    load_spec, list_specs, execution_plan, resolve_templates,
    _validate, _collect_refs, WorkflowSpec, StepSpec, InputSpec, OutputSpec,
)
from services.workflows.artifacts import ArtifactStore
from services.workflows.steps.python import PythonStepExecutor, _import_function
from services.workflows.steps import StepContext, StepResult


# ── Spec validation tests ─────────────────────────────────────────────────

def test_cycle_detection():
    """Cyclic dependencies should raise ValueError."""
    spec = WorkflowSpec(
        name="cycle_test",
        inputs={},
        steps=[
            StepSpec(id="a", type="forge", depends_on=["b"]),
            StepSpec(id="b", type="forge", depends_on=["c"]),
            StepSpec(id="c", type="forge", depends_on=["a"]),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised ValueError for cycle"
    except ValueError as e:
        assert "cyclic" in str(e).lower()


def test_missing_dependency():
    """Referencing a non-existent step in depends_on should raise."""
    spec = WorkflowSpec(
        name="bad_dep",
        inputs={},
        steps=[
            StepSpec(id="a", type="forge", depends_on=["nonexistent"]),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised ValueError for missing dep"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_duplicate_step_id():
    """Duplicate step IDs should raise."""
    spec = WorkflowSpec(
        name="dup",
        inputs={},
        steps=[
            StepSpec(id="a", type="forge"),
            StepSpec(id="a", type="forge"),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised for duplicate ID"
    except ValueError as e:
        assert "Duplicate" in str(e)


def test_bad_template_ref_undefined_input():
    """Template referencing undefined input should raise."""
    spec = WorkflowSpec(
        name="bad_ref",
        inputs={"prompt": InputSpec(type="string", required=True)},
        steps=[
            StepSpec(
                id="a", type="forge",
                params={"input_prompt": "{{ inputs.nonexistent }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised for undefined input ref"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_bad_template_ref_missing_output():
    """Template referencing a step output that doesn't exist should raise."""
    spec = WorkflowSpec(
        name="bad_output_ref",
        inputs={"prompt": InputSpec(type="string", required=True)},
        steps=[
            StepSpec(
                id="a", type="forge",
                params={"input_prompt": "{{ inputs.prompt }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
            StepSpec(
                id="b", type="forge",
                depends_on=["a"],
                params={"ref": "{{ a.outputs.nonexistent }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised for missing output ref"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_bad_template_ref_missing_dep():
    """Template referencing a step not in depends_on should raise."""
    spec = WorkflowSpec(
        name="missing_dep_ref",
        inputs={"prompt": InputSpec(type="string", required=True)},
        steps=[
            StepSpec(
                id="a", type="forge",
                params={"input_prompt": "{{ inputs.prompt }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
            StepSpec(
                id="b", type="forge",
                # depends_on missing "a" but references a.outputs
                params={"ref": "{{ a.outputs.image }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised for missing dependency"
    except ValueError as e:
        assert "depends_on" in str(e).lower()


def test_valid_template_refs():
    """Valid template references should pass validation."""
    spec = WorkflowSpec(
        name="good_refs",
        inputs={"prompt": InputSpec(type="string", required=True)},
        steps=[
            StepSpec(
                id="a", type="forge",
                params={"input_prompt": "{{ inputs.prompt }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
            StepSpec(
                id="b", type="forge",
                depends_on=["a"],
                params={"ref": "{{ a.outputs.image }}"},
                outputs={"image": OutputSpec(media_type="image/png")},
            ),
        ],
    )
    _validate(spec)  # Should not raise
    print("PASS: valid template references pass validation")


# ── Migrated YAML spec tests ──────────────────────────────────────────────

MIGRATED_SPECS = [
    "character_sheet",
    "tech_noir_generate",
    "tech_noir_sheet",
    "tech_noir_video",
    "tech_noir_trellis",
    "tech_noir_face_detailer",
    "tech_noir_motion_npz",
    "tech_noir_outfit",
    "tech_noir_state",
    "vnccs_pose_edit",
    "wdc_ltx_fflf_2stage",
    "wdc_ltx_audio",
]


def test_all_migrated_specs_load():
    """All migrated YAML specs should load and validate without errors."""
    loaded = 0
    for spec_name in MIGRATED_SPECS:
        spec = load_spec(spec_name)
        assert spec.name == spec_name, f"Name mismatch: {spec.name} != {spec_name}"
        assert len(spec.steps) >= 1, f"{spec_name} has no steps"

        # Verify execution plan is valid
        plan = execution_plan(spec)
        assert len(plan) >= 1, f"{spec_name} has empty execution plan"

        # Verify all steps are reachable in the plan
        plan_steps = {s.id for group in plan for s in group}
        spec_steps = {s.id for s in spec.steps}
        assert plan_steps == spec_steps, (
            f"{spec_name}: plan steps {plan_steps} != spec steps {spec_steps}"
        )
        loaded += 1

    print(f"PASS: {loaded} migrated specs load and validate")


def test_multi_step_specs_have_correct_deps():
    """Multi-step specs should have correct dependency chains."""
    # character_sheet: generate_base → refine
    spec = load_spec("character_sheet")
    plan = execution_plan(spec)
    assert len(plan) == 2, f"character_sheet should have 2 groups, got {len(plan)}"

    # vnccs_pose_edit: render_mesh → extract_skeleton → pose_transfer
    spec = load_spec("vnccs_pose_edit")
    plan = execution_plan(spec)
    assert len(plan) == 3, f"vnccs_pose_edit should have 3 groups, got {len(plan)}"
    assert plan[0][0].id == "render_mesh"
    assert plan[1][0].id == "extract_skeleton"
    assert plan[2][0].id == "pose_transfer"

    # tech_noir_face_detailer: detect → refine → composite
    spec = load_spec("tech_noir_face_detailer")
    plan = execution_plan(spec)
    assert len(plan) == 3
    assert plan[0][0].id == "detect_face"
    assert plan[1][0].id == "refine_face"
    assert plan[2][0].id == "composite"

    print("PASS: multi-step specs have correct dependency chains")


def test_yaml_specs_not_in_migration_map():
    """YAML-only specs (video_editor, _test_spec) should also be loadable."""
    spec = load_spec("video_editor")
    assert len(spec.steps) == 11

    spec = load_spec("_test_spec")
    assert len(spec.steps) == 2

    print("PASS: non-migrated YAML specs still load correctly")


# ── Python step executor tests ────────────────────────────────────────────

def test_import_function():
    """_import_function should resolve dotted paths."""
    fn = _import_function("services.workflows.vnccs:char_sheet")
    assert callable(fn)
    assert fn.__name__ == "char_sheet"

    # dot notation
    fn2 = _import_function("services.workflows.vnccs.char_sheet")
    assert fn2 is fn

    print("PASS: function import by dotted path")


def test_import_function_errors():
    """_import_function should raise on bad paths."""
    try:
        _import_function("nonexistent_module:function")
        assert False
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        _import_function("services.workflows.vnccs:nonexistent_function")
        assert False
    except ValueError as e:
        assert "nonexistent_function" in str(e)

    print("PASS: bad function imports raise errors")


async def test_python_step_executor():
    """PythonStepExecutor should call a function and store result as artifact."""
    executor = PythonStepExecutor()

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = ArtifactStore(base_dir=Path(tmp))
        context = StepContext(
            run_id="test_run",
            step_id="test_step",
            artifacts=artifacts,
        )

        # Use a simple built-in function
        params = {
            "function": "json:dumps",
            "obj": {"test": True},
        }
        result = await executor.execute(params, context)
        assert isinstance(result, StepResult)
        assert "output" in result.outputs
        assert result.duration_ms >= 0

        # Verify artifact was stored
        output_path = Path(result.outputs["output"])
        assert output_path.exists()
        content = output_path.read_text()
        assert '"test"' in content

    print("PASS: PythonStepExecutor stores function result as artifact")


# ── Collect refs tests ────────────────────────────────────────────────────

def test_collect_refs():
    """_collect_refs should extract all {{ }} references from nested params."""
    params = {
        "prompt": "{{ inputs.character_prompt }}",
        "nested": {
            "ref": "{{ generate_base.outputs.image }}",
            "list": ["{{ inputs.seed }}", "plain string"],
        },
    }
    refs = [r.strip() for r in _collect_refs(params)]
    assert len(refs) == 3
    assert "inputs.character_prompt" in refs
    assert "generate_base.outputs.image" in refs
    assert "inputs.seed" in refs

    print("PASS: _collect_refs extracts template references from nested params")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    test_cycle_detection()
    test_missing_dependency()
    test_duplicate_step_id()
    test_bad_template_ref_undefined_input()
    test_bad_template_ref_missing_output()
    test_bad_template_ref_missing_dep()
    test_valid_template_refs()
    test_all_migrated_specs_load()
    test_multi_step_specs_have_correct_deps()
    test_yaml_specs_not_in_migration_map()
    test_import_function()
    test_import_function_errors()

    # Async tests
    asyncio.run(test_python_step_executor())

    test_collect_refs()

    print("\nAll Phase 4 tests passed.")


if __name__ == "__main__":
    main()
