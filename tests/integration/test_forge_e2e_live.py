"""E2E tests for the Forge — runs against a LIVE deployment.

All tests make real HTTP requests and validate real model loads and inference output.
Requires a running Ray cluster with Forge deployed (ingress at FORGE_URL).

Usage:
  FORGE_URL=http://100.86.69.57:30080 pytest tests/integration/test_forge_e2e_live.py -v
"""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

import pytest

FORGE_URL = os.environ.get("FORGE_URL", "http://100.86.69.57:30080")


def _req(method: str, path: str, data: dict | None = None,
         timeout: int = 30) -> dict:
    url = f"{FORGE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def assert_any(result: dict, key: str, *values: str) -> None:
    """Assert result[key] equals one of the given values."""
    assert result.get(key) in values, (
        f"Expected {key} to be one of {values}, got {result.get(key)!r}"
    )


# ─── Read endpoints ────────────────────────────────────────────────────


class TestReadEndpoints:
    """Endpoints that read state without side effects."""

    def test_health(self):
        r = _req("GET", "/health")
        assert r == {"status": "ok"}

    def test_status_initial_state(self):
        r = _req("GET", "/status")
        assert "vram_free_mb" in r
        assert "vram_total_mb" in r
        assert "vram_allocated_mb" in r
        assert "loaded" in r
        assert "gpu" in r
        assert r["gpu"]["device"] == "NVIDIA GeForce RTX 4090"

    def test_status_vram_invariant(self):
        r = _req("GET", "/status")
        assert r["vram_free_mb"] + r["vram_allocated_mb"] == r["vram_total_mb"]

    def test_list_services(self):
        r = _req("GET", "/v1/services")
        assert isinstance(r, list)
        names = [s["name"] for s in r]
        for required in ("llm", "wan2gp", "comfyui", "kokoro", "espeak", "faster_whisper", "trellis"):
            assert required in names, f"Missing service: {required}"

    def test_service_info_llm(self):
        r = _req("GET", "/v1/services/llm")
        assert r["name"] == "llm"
        assert "label" in r
        assert "category" in r
        assert "default_model" in r

    def test_service_info_not_found(self):
        import urllib
        with pytest.raises(urllib.error.HTTPError, match="404"):
            _req("GET", "/v1/services/nonexistent")

    def test_list_models(self):
        r = _req("GET", "/v1/models")
        assert r["object"] == "list"
        assert len(r["data"]) >= 15
        categories = set(m["category"] for m in r["data"])
        for c in ("llm", "tts", "asr", "audio", "image", "3d", "motion"):
            assert c in categories, f"Missing category: {c}"

    def test_models_filtered(self):
        r = _req("GET", "/v1/models?category=tts")
        for model in r["data"]:
            assert model["category"] == "tts"

    def test_run_catalog(self):
        r = _req("GET", "/v1/run/catalog")
        assert "pipelines" in r
        assert "services" in r
        assert len(r["pipelines"]) > 0


# ─── Model lifecycle: LLM ──────────────────────────────────────────────


class TestLLMLifecycle:
    """Load → infer → release cycle for the LLM service."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        try:
            _req("POST", "/admin/unload", {})
        except Exception:
            pass

    def test_load_llm(self):
        r = _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        assert_any(r, "status", "loaded", "already_loaded")
        assert r.get("service") == "llm"
        if r["status"] == "loaded":
            assert r["vram_used_mb"] > 0
            assert r["vram_free_mb"] < r["vram_used_mb"]

    def test_preload_idempotent(self):
        r1 = _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        assert_any(r1, "status", "loaded", "already_loaded")
        r2 = _req("POST", "/admin/load", {"service": "llm"}, timeout=30)
        assert r2["status"] == "already_loaded"

    def test_llm_inference(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        import time
        time.sleep(3)
        r = _req("POST", "/v1/chat/completions", {
            "model": "qwen3.6-27b-q4_k_xl",
            "messages": [{"role": "user", "content": "reply exactly with: pong"}],
        }, timeout=120)
        content = ""
        if "choices" in r:
            content = r["choices"][0]["message"]["content"]
        elif r.get("status") == "success":
            content = r["data"]["choices"][0]["message"]["content"]
        assert "pong" in content.lower(), f"Bad response: {content}"

    def test_release_llm(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        r = _req("POST", "/admin/unload", {})
        assert r["status"] == "released"
        status = _req("GET", "/status")
        assert status["loaded"] == {}
        assert status["vram_free_mb"] == status["vram_total_mb"]

    def test_reloa_after_release(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        _req("POST", "/admin/unload", {})
        r = _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        assert r["status"] == "loaded"
        _req("POST", "/admin/unload", {})

    def test_infer_on_already_loaded(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        import time
        time.sleep(3)
        r = _req("POST", "/v1/chat/completions", {
            "model": "qwen3.6-27b-q4_k_xl",
            "messages": [{"role": "user", "content": "count to 5: one two"}],
        }, timeout=120)
        content = ""
        if "choices" in r:
            content = r["choices"][0]["message"]["content"]
        elif r.get("status") == "success":
            content = r["data"]["choices"][0]["message"]["content"]
        assert len(content) > 5, f"Empty/short response: {content}"
        _req("POST", "/admin/unload", {})


# ─── VRAM lifecycle ────────────────────────────────────────────────────


class TestVRAMLifecycle:
    """VRAM tracking invariant through load/unload cycles."""

    def _vram_sum(self, status: dict) -> int:
        return status["vram_free_mb"] + status["vram_allocated_mb"]

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        try:
            _req("POST", "/admin/unload", {})
        except Exception:
            pass

    def test_invariant_after_load_unload(self):
        before = _req("GET", "/status")
        assert self._vram_sum(before) == before["vram_total_mb"]

        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        during = _req("GET", "/status")
        assert self._vram_sum(during) == during["vram_total_mb"]

        _req("POST", "/admin/unload", {})
        after = _req("GET", "/status")
        assert self._vram_sum(after) == after["vram_total_mb"]
        assert after["vram_free_mb"] == after["vram_total_mb"]

    def test_release_not_loaded_is_noop(self):
        r = _req("POST", "/admin/unload", {})
        assert r["status"] == "released"

    def test_vram_never_negative(self):
        status = _req("GET", "/status")
        assert status["vram_free_mb"] >= 0
        assert status["vram_allocated_mb"] >= 0


# ─── LLM configure ─────────────────────────────────────────────────────


class TestLLMConfigure:
    """LLM configuration endpoint."""

    @pytest.fixture(autouse=True)
    def _release(self):
        yield
        try:
            _req("POST", "/admin/unload", {})
        except Exception:
            pass

    def test_configure_after_load(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        import time
        time.sleep(3)
        r = _req("POST", "/v1/llm/configure", {
            "model": "qwen3.6-27b-q5_k_s-32k",
            "startup_overrides": {"n_gpu_layers": 40},
        }, timeout=30)
        assert_any(r, "status", "ok", "success")

    def test_configure_default_model(self):
        _req("POST", "/admin/load", {"service": "llm"}, timeout=60)
        import time
        time.sleep(3)
        r = _req("POST", "/v1/llm/configure", {
            "model": "qwen3.6-27b-q5_k_s-32k",
            "engine": "beellama",
        }, timeout=30)
        assert_any(r, "status", "ok", "success")
        assert r.get("changed") is not None
