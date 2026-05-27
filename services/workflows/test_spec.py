"""Test workflow spec parser and execution planning."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.workflows.spec import load_spec, execution_plan, resolve_templates


def test_character_sheet_spec():
    spec_dir = Path(__file__).resolve().parents[2] / "config" / "workflows"
    spec = load_spec("character_sheet", spec_dir)

    assert spec.name == "character_sheet"
    assert len(spec.steps) == 2
    assert spec.steps[0].id == "generate_base"
    assert spec.steps[1].id == "refine"
    assert spec.steps[1].depends_on == ["generate_base"]
    print("PASS: spec parsing")


def test_execution_plan():
    spec_dir = Path(__file__).resolve().parents[2] / "config" / "workflows"
    spec = load_spec("character_sheet", spec_dir)
    plan = execution_plan(spec)

    assert len(plan) == 2
    # Group 1: generate_base (no deps)
    assert len(plan[0]) == 1
    assert plan[0][0].id == "generate_base"
    # Group 2: refine (depends on generate_base)
    assert len(plan[1]) == 1
    assert plan[1][0].id == "refine"
    print("PASS: execution plan")


def test_template_resolution():
    inputs = {"prompt": "cyberpunk warrior", "seed": 42, "quality": "turbo", "negative_prompt": "bad"}
    artifact_paths = {"generate_base": {"outputs": {"image": "/mnt/data/workflows/abc123/generate_base/output.png"}}}

    params = {
        "input_prompt": "{{ inputs.prompt }}",
        "seed": "{{ inputs.seed }}",
        "image_b64": "{{ generate_base.outputs.image }}",
    }

    resolved = resolve_templates(params, inputs, artifact_paths)

    assert resolved["input_prompt"] == "cyberpunk warrior"
    assert resolved["seed"] == 42
    assert resolved["image_b64"] == "/mnt/data/workflows/abc123/generate_base/output.png"
    print("PASS: template resolution")


def test_cycle_detection():
    from services.workflows.spec import _validate, WorkflowSpec, StepSpec
    spec = WorkflowSpec(
        name="cycle_test",
        inputs={},
        steps=[
            StepSpec(id="a", type="forge", depends_on=["b"]),
            StepSpec(id="b", type="forge", depends_on=["a"]),
        ],
    )
    try:
        _validate(spec)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cyclic" in str(e).lower()
    print("PASS: cycle detection")


if __name__ == "__main__":
    test_character_sheet_spec()
    test_execution_plan()
    test_template_resolution()
    test_cycle_detection()
    print("\nAll tests passed.")
