"""Tests for TNAP protocol — Pydantic models and BaseGPUDeployment helpers.

Validates handle_request(), handle_response(), handle_error(), and _extract_input()
on BaseGPUDeployment directly (no subclass needed, no Ray cluster).
"""
from __future__ import annotations

import base64

import pytest


# ─── Pydantic Model Tests ──────────────────────────────────────────────────


class TestTNAPModels:

    def test_tnap_input_defaults_all_none(self):
        from services.base import TNAPInput
        inp = TNAPInput()
        assert inp.prompt is None
        assert inp.text is None
        assert inp.image_b64 is None
        assert inp.audio_b64 is None
        assert inp.video_b64 is None
        assert inp.model is None
        assert inp.voice is None
        assert inp.seed is None
        assert inp.steps is None
        assert inp.guidance is None
        assert inp.messages is None
        assert inp.stream is None

    def test_tnap_input_accepts_all_fields(self):
        from services.base import TNAPInput
        inp = TNAPInput(
            prompt="test", text="hello", image_b64="aW1n",
            audio_b64="YXVk", video_b64="dmlk", model="m1",
            voice="en", seed=42, steps=20, guidance=1.5,
            messages=[{"role": "user", "content": "hi"}], stream=True,
        )
        assert inp.prompt == "test"
        assert inp.messages == [{"role": "user", "content": "hi"}]

    def test_tnap_request_defaults(self):
        from services.base import TNAPRequest
        req = TNAPRequest()
        assert req.action == "generate"
        assert req.input.prompt is None
        assert req.config is None

    def test_tnap_request_with_config(self):
        from services.base import TNAPRequest
        req = TNAPRequest(config={"precision": "fp16", "quantization": "8bit", "low_resource": True})
        assert req.config.precision == "fp16"
        assert req.config.quantization == "8bit"
        assert req.config.low_resource is True

    def test_tnap_response_defaults(self):
        from services.base import TNAPResponse
        resp = TNAPResponse()
        assert resp.status == "success"
        assert resp.output.content == ""
        assert resp.metrics.latency_ms == 0
        assert resp.error is None


# ─── handle_request Tests ──────────────────────────────────────────────────


class TestHandleRequest:

    def test_minimal_body(self, base_deployment):
        tnap, extracted = base_deployment.handle_request({"action": "generate"})
        assert tnap.action == "generate"
        assert extracted == {}

    def test_extracts_prompt(self, base_deployment):
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"prompt": "a cat"}}
        )
        assert extracted["prompt"] == "a cat"

    def test_extracts_text(self, base_deployment):
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"text": "hello"}}
        )
        assert extracted["text"] == "hello"

    def test_extracts_image_b64_and_decodes(self, base_deployment):
        raw = b"\x89PNG data"
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"image_b64": base64.b64encode(raw).decode()}}
        )
        assert extracted["image"] == raw

    def test_extracts_audio_b64_and_decodes(self, base_deployment, sample_wav_b64, sample_wav_bytes):
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"audio_b64": sample_wav_b64}}
        )
        assert extracted["audio"] == sample_wav_bytes

    def test_extracts_video_b64_and_decodes(self, base_deployment):
        raw = b"video data"
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"video_b64": base64.b64encode(raw).decode()}}
        )
        assert extracted["video"] == raw

    def test_extracts_scalar_fields(self, base_deployment):
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {
                "model": "m1", "voice": "en", "seed": 42,
                "steps": 20, "guidance": 1.5,
            }}
        )
        assert extracted["model"] == "m1"
        assert extracted["voice"] == "en"
        assert extracted["seed"] == 42
        assert extracted["steps"] == 20
        assert extracted["guidance"] == 1.5

    def test_extracts_messages_list(self, base_deployment):
        msgs = [{"role": "user", "content": "hello"}]
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"messages": msgs}}
        )
        assert extracted["messages"] == msgs

    def test_extracts_stream_bool(self, base_deployment):
        _, extracted = base_deployment.handle_request(
            {"action": "generate", "input": {"stream": True}}
        )
        assert extracted["stream"] is True

    def test_config_applied_to_inference_config(self, base_deployment):
        base_deployment.handle_request({
            "action": "generate",
            "config": {"precision": "fp16", "quantization": "4bit", "low_resource": True},
        })
        assert base_deployment.config.precision == "fp16"
        assert base_deployment.config.quantization == "4bit"
        assert base_deployment.config.low_resource is True

    def test_empty_input_returns_empty_dict(self, base_deployment):
        _, extracted = base_deployment.handle_request({"action": "generate"})
        assert extracted == {}

    def test_config_none_leaves_config_unchanged(self, base_deployment):
        base_deployment.config.precision = "fp32"
        base_deployment.handle_request({"action": "generate"})
        assert base_deployment.config.precision == "fp32"


# ─── handle_response Tests ─────────────────────────────────────────────────


class TestHandleResponse:

    def test_bytes_content_base64_encoded(self, base_deployment):
        resp = base_deployment.handle_response(b"audio data", "audio/wav", 100)
        assert resp["status"] == "success"
        assert resp["output"]["content"] == base64.b64encode(b"audio data").decode()
        assert resp["output"]["type"] == "audio/wav"

    def test_string_content_has_text_field(self, base_deployment):
        resp = base_deployment.handle_response("hello world", "text/plain", 50)
        assert resp["output"]["text"] == "hello world"
        assert resp["output"]["content"] == base64.b64encode(b"hello world").decode()

    def test_output_type_preserved(self, base_deployment):
        for mime in ("audio/wav", "image/png", "model/gltf-binary", "application/json"):
            resp = base_deployment.handle_response(b"x", mime, 0)
            assert resp["output"]["type"] == mime

    def test_url_included_when_provided(self, base_deployment):
        resp = base_deployment.handle_response(b"x", "text/plain", 0, url="/output/file.txt")
        assert resp["output"]["url"] == "/output/file.txt"

    def test_url_absent_when_not_provided(self, base_deployment):
        resp = base_deployment.handle_response(b"x", "text/plain", 0)
        assert "url" not in resp["output"]

    def test_latency_ms_in_metrics(self, base_deployment):
        resp = base_deployment.handle_response(b"x", "text/plain", 42)
        assert resp["metrics"]["latency_ms"] == 42

    def test_extra_metrics_merged(self, base_deployment):
        resp = base_deployment.handle_response(
            b"x", "text/plain", 0, extra_metrics={"language": "en", "words": 5}
        )
        assert resp["metrics"]["language"] == "en"
        assert resp["metrics"]["words"] == 5

    def test_model_version_in_metrics(self, base_deployment):
        resp = base_deployment.handle_response(b"x", "text/plain", 0)
        assert resp["metrics"]["model_version"] == "test-model"

    def test_vram_zero_when_no_torch(self, base_deployment):
        import sys
        torch_mod = sys.modules.get("torch")
        sys.modules["torch"] = None
        try:
            resp = base_deployment.handle_response(b"x", "text/plain", 0)
            assert resp["metrics"]["vram_used_mb"] == 0
        finally:
            if torch_mod is not None:
                sys.modules["torch"] = torch_mod
            else:
                sys.modules.pop("torch", None)


# ─── handle_error Tests ────────────────────────────────────────────────────


class TestHandleError:

    def test_error_response_structure(self, base_deployment):
        resp = base_deployment.handle_error("something went wrong")
        assert resp["status"] == "error"
        assert resp["output"] == {}
        assert resp["error"] == "something went wrong"
        assert resp["metrics"]["latency_ms"] == 0

    def test_error_with_custom_latency(self, base_deployment):
        resp = base_deployment.handle_error("bad", latency_ms=500)
        assert resp["metrics"]["latency_ms"] == 500
