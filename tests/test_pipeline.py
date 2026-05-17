"""Tests for pipeline executor — spec validation, ref resolution, execution."""
from __future__ import annotations

import pytest

from gateway.pipeline import (
    PipelineSpec,
    PipelineStep,
    resolve_refs,
    execute_pipeline,
)


# ── Spec Validation ──────────────────────────────────────────────────────


class TestPipelineSpec:
    def test_basic_spec(self):
        data = {"steps": [
            {"name": "gen", "service": "kokoro", "params": {"text": "hello"}},
        ]}
        spec = PipelineSpec.from_dict(data)
        assert len(spec.steps) == 1
        assert spec.steps[0].name == "gen"

    def test_two_step_spec(self):
        data = {"steps": [
            {"name": "gen", "service": "ace_step", "params": {"prompt": "piano"}},
            {"name": "to3d", "service": "trellis", "depends_on": ["gen"],
             "params": {"image_b64": "{gen.output.data}"}},
        ]}
        spec = PipelineSpec.from_dict(data)
        assert len(spec.steps) == 2
        order = spec.execution_order()
        assert order[0].name == "gen"
        assert order[1].name == "to3d"

    def test_empty_steps_rejected(self):
        with pytest.raises(ValueError, match="at least one step"):
            PipelineSpec.from_dict({"steps": []})

    def test_missing_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            PipelineSpec.from_dict({"steps": [{"service": "kokoro"}]})

    def test_missing_service_rejected(self):
        with pytest.raises(ValueError, match="service"):
            PipelineSpec.from_dict({"steps": [{"name": "x"}]})

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            PipelineSpec.from_dict({"steps": [
                {"name": "x", "service": "a"},
                {"name": "x", "service": "b"},
            ]})

    def test_missing_dep_rejected(self):
        with pytest.raises(ValueError, match="doesn't exist"):
            PipelineSpec.from_dict({"steps": [
                {"name": "x", "service": "a", "depends_on": ["missing"]},
            ]})

    def test_cycle_rejected(self):
        with pytest.raises(ValueError, match="cyclic"):
            PipelineSpec.from_dict({"steps": [
                {"name": "a", "service": "s1", "depends_on": ["b"]},
                {"name": "b", "service": "s2", "depends_on": ["a"]},
            ]})

    def test_execution_order_topological(self):
        data = {"steps": [
            {"name": "c", "service": "s3", "depends_on": ["a"]},
            {"name": "a", "service": "s1"},
            {"name": "b", "service": "s2", "depends_on": ["a"]},
        ]}
        spec = PipelineSpec.from_dict(data)
        order = spec.execution_order()
        names = [s.name for s in order]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")


# ── Reference Resolution ────────────────────────────────────────────────


class TestResolveRefs:
    def test_no_refs(self):
        params = {"text": "hello", "seed": 42}
        assert resolve_refs(params, {}) == params

    def test_simple_ref(self):
        params = {"image_b64": "{gen.output.data}"}
        results = {"gen": {"output": {"data": "base64string"}}}
        resolved = resolve_refs(params, results)
        assert resolved["image_b64"] == "base64string"

    def test_nested_dict_refs(self):
        params = {"input": {"audio": "{step1.output.content}"}}
        results = {"step1": {"output": {"content": "wavdata"}}}
        resolved = resolve_refs(params, results)
        assert resolved["input"]["audio"] == "wavdata"

    def test_list_refs(self):
        params = {"files": ["{a.output.x}", "{b.output.y}"]}
        results = {"a": {"output": {"x": "X"}}, "b": {"output": {"y": "Y"}}}
        resolved = resolve_refs(params, results)
        assert resolved["files"] == ["X", "Y"]

    def test_template_ref(self):
        params = {"prompt": "generate from {gen.output.seed}"}
        results = {"gen": {"output": {"seed": 42}}}
        resolved = resolve_refs(params, results)
        assert resolved["prompt"] == "generate from 42"

    def test_missing_ref_returns_none(self):
        params = {"x": "{missing.output.data}"}
        resolved = resolve_refs(params, {})
        assert resolved["x"] is None

    def test_preserves_type_on_exact_match(self):
        params = {"value": "{s.output.num}"}
        results = {"s": {"output": {"num": 42}}}
        resolved = resolve_refs(params, results)
        assert resolved["value"] == 42
        assert isinstance(resolved["value"], int)


# ── Pipeline Execution ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_single_step():
    spec = PipelineSpec.from_dict({"steps": [
        {"name": "sfx", "service": "moss_soundeffect", "params": {"prompt": "rain"}},
    ]})

    async def mock_dispatch(service, params):
        return {"output": {"content": "audio_data"}}

    events = await execute_pipeline(spec, mock_dispatch)
    event_types = [e["event"] for e in events]
    assert "pipeline_started" in event_types
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "pipeline_completed" in event_types


@pytest.mark.asyncio
async def test_execute_chained_steps():
    spec = PipelineSpec.from_dict({"steps": [
        {"name": "gen", "service": "ace_step", "params": {"prompt": "piano"}},
        {"name": "convert", "service": "trellis", "depends_on": ["gen"],
         "params": {"input_data": "{gen.output.data}"}},
    ]})

    call_log = []

    async def mock_dispatch(service, params):
        call_log.append((service, dict(params)))
        return {"output": {"data": f"{service}_result"}}

    events = await execute_pipeline(spec, mock_dispatch)

    # Second step should have received resolved ref
    assert call_log[1][1]["input_data"] == "ace_step_result"

    event_types = [e["event"] for e in events]
    assert "pipeline_completed" in event_types


@pytest.mark.asyncio
async def test_execute_error_stops_pipeline():
    spec = PipelineSpec.from_dict({"steps": [
        {"name": "fail", "service": "bad_service", "params": {}},
        {"name": "never", "service": "other", "depends_on": ["fail"], "params": {}},
    ]})

    async def mock_dispatch(service, params):
        raise RuntimeError("service crashed")

    events = await execute_pipeline(spec, mock_dispatch)
    event_types = [e["event"] for e in events]
    assert "step_error" in event_types
    assert "pipeline_error" in event_types
    # Second step should never start
    assert "pipeline_completed" not in event_types
