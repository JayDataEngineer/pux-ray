"""Tests for registry/validate.py — IaC validation gate.

Proves the validator catches each class of config error that caused
production failures:
1. YAML syntax errors (colons in unquoted descriptions)
2. VRAM budget exceeded (model + draft > GPU capacity)
3. Cross-reference mismatches (LOAD_KWARGS != DEFAULT_MODEL)
4. Missing TNAP fields (messages not extractable)
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from registry.validate import (
    ValidationFail,
    check_cross_references,
    check_description_syntax,
    check_tnap_fields,
    check_vram_budget,
    check_yaml_syntax,
    run_all,
    VRAM_USABLE_GB,
)


def _make_registry(tmp_path: Path, data: dict) -> Path:
    reg = tmp_path / "model_registry.yaml"
    with open(reg, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return reg


# ── YAML Syntax ──────────────────────────────────────────────────────

class TestYAMLSyntax:

    def test_valid_yaml_passes(self, tmp_path):
        reg = _make_registry(tmp_path, {"llm": {"model-a": {"path": "a.gguf", "size_gb": 1}}})
        with patch("registry.validate._REGISTRY_PATH", reg):
            assert check_yaml_syntax() == []

    def test_broken_yaml_fails(self, tmp_path):
        reg = tmp_path / "model_registry.yaml"
        reg.write_text('llm:\n  model-a:\n    path: a.gguf\n    description: This has a colon: and breaks\n')
        with patch("registry.validate._REGISTRY_PATH", reg):
            failures = check_yaml_syntax()
            assert len(failures) == 1
            assert failures[0].check == "yaml_syntax"


# ── VRAM Budget ──────────────────────────────────────────────────────

class TestVRAMBudget:

    def test_model_fits(self, tmp_path):
        data = {
            "llm": {
                "small-model": {
                    "path": "small.gguf",
                    "size_gb": 10,
                    "vram_estimate_gb": 11,
                    "device": "gpu",
                }
            }
        }
        reg = _make_registry(tmp_path, data)
        # Also need a master_router with this model in LOAD_KWARGS
        router_src = textwrap.dedent('''
            LOAD_KWARGS = {
                "llm": {"model_name": "small-model"},
            }
        ''')
        router_path = tmp_path / "services" / "creative" / "master_router.py"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text(router_src)

        deploy_path = tmp_path / "services" / "llm" / "deployment.py"
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        deploy_path.write_text('DEFAULT_MODEL = "small-model"\n')

        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_vram_budget(data)
            assert failures == []

    def test_model_with_draft_exceeds_vram(self, tmp_path):
        data = {
            "llm": {
                "big-model": {
                    "path": "big.gguf",
                    "size_gb": 20,
                    "vram_estimate_gb": 21,
                    "device": "gpu",
                    "spec_draft_model": "llm/draft-q4.gguf",
                },
                "draft-q4": {
                    "path": "llm/draft-q4.gguf",
                    "size_gb": 2.0,
                    "device": "gpu",
                },
            }
        }
        reg = _make_registry(tmp_path, data)
        router_path = tmp_path / "services" / "creative" / "master_router.py"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text('LOAD_KWARGS = {"llm": {"model_name": "big-model"}}')

        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_vram_budget(data)
            assert len(failures) == 1
            assert failures[0].check == "vram_budget"
            # 21 + 2 = 23 > 22 usable
            assert "23.0GB" in failures[0].message

    def test_non_gpu_model_skipped(self, tmp_path):
        data = {
            "tts": {
                "cpu-model": {
                    "path": "cpu.bin",
                    "size_gb": 1,
                    "device": "cpu",
                }
            }
        }
        reg = _make_registry(tmp_path, data)
        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            assert check_vram_budget(data) == []


# ── Cross-References ─────────────────────────────────────────────────

class TestCrossReferences:

    def test_matching_model_names_pass(self, tmp_path):
        data = {"llm": {"qwen-q5": {"path": "qwen.gguf", "size_gb": 18}}}
        reg = _make_registry(tmp_path, data)

        router_path = tmp_path / "services" / "creative" / "master_router.py"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text(
            'LOAD_KWARGS = {"llm": {"model_name": "qwen-q5"}}'
        )
        deploy_path = tmp_path / "services" / "llm" / "deployment.py"
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        deploy_path.write_text('DEFAULT_MODEL = "qwen-q5"\n')

        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            assert check_cross_references(data) == []

    def test_mismatched_model_names_fail(self, tmp_path):
        data = {"llm": {"qwen-q5": {"path": "qwen-q5.gguf", "size_gb": 18}}}
        reg = _make_registry(tmp_path, data)

        router_path = tmp_path / "services" / "creative" / "master_router.py"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text(
            'LOAD_KWARGS = {"llm": {"model_name": "qwen-q6"}}'
        )
        deploy_path = tmp_path / "services" / "llm" / "deployment.py"
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        deploy_path.write_text('DEFAULT_MODEL = "qwen-q5"\n')

        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_cross_references(data)
            # Should get: 1 for LOAD_KWARGS referencing missing model, 1 for mismatch
            assert len(failures) >= 1
            checks = [f.check for f in failures]
            assert "cross_reference" in checks

    def test_load_kwargs_references_missing_model(self, tmp_path):
        data = {"llm": {"model-a": {"path": "a.gguf", "size_gb": 10}}}
        reg = _make_registry(tmp_path, data)

        router_path = tmp_path / "services" / "creative" / "master_router.py"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        router_path.write_text(
            'LOAD_KWARGS = {"llm": {"model_name": "nonexistent-model"}}'
        )

        with patch("registry.validate._REGISTRY_PATH", reg), \
             patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_cross_references(data)
            assert any("nonexistent-model" in f.message for f in failures)


# ── TNAP Fields ──────────────────────────────────────────────────────

class TestTNAPFields:

    def test_tnap_with_messages_passes(self, tmp_path):
        base_src = textwrap.dedent('''
            class TNAPInput(BaseModel):
                messages: Optional[list[dict]] = None
                stream: Optional[bool] = None

            class Other:
                pass

            def _extract_input(self, inp):
                if inp.messages is not None:
                    result["messages"] = inp.messages

            def other_method():
                pass
        ''')
        base_path = tmp_path / "services" / "base.py"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(base_src)

        with patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_tnap_fields({})
            assert failures == []

    def test_tnap_without_messages_fails(self, tmp_path):
        base_src = textwrap.dedent('''
            class TNAPInput(BaseModel):
                prompt: Optional[str] = None

            class Other:
                pass

            def _extract_input(self, inp):
                if inp.prompt:
                    result["prompt"] = inp.prompt

            def other_method():
                pass
        ''')
        base_path = tmp_path / "services" / "base.py"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(base_src)

        with patch("registry.validate._PROJECT_ROOT", tmp_path):
            failures = check_tnap_fields({})
            assert len(failures) >= 1
            assert any("messages" in f.message for f in failures)


# ── Description YAML Safety ─────────────────────────────────────────

class TestDescriptionYAMLSafety:

    def test_quoted_description_passes(self, tmp_path):
        reg = tmp_path / "model_registry.yaml"
        reg.write_text('llm:\n  model-a:\n    description: "Has a colon: safe"\n')
        with patch("registry.validate._REGISTRY_PATH", reg):
            assert check_description_syntax({}) == []

    def test_unquoted_colon_in_description_fails(self, tmp_path):
        reg = tmp_path / "model_registry.yaml"
        reg.write_text('llm:\n  model-a:\n    description: OOM: this breaks\n')
        with patch("registry.validate._REGISTRY_PATH", reg):
            failures = check_description_syntax({})
            assert len(failures) == 1
            assert "unquoted" in failures[0].message


# ── Integration: run_all against real config ─────────────────────────

class TestRealConfig:
    """Validates the actual project config passes all checks."""

    def test_real_config_passes(self):
        failures = run_all()
        assert failures == [], (
            f"Config validation failures:\n" +
            "\n".join(f"  {f}" for f in failures)
        )
