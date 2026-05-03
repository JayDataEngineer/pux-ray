"""E2E test suite for Tech Noir Ray infrastructure.

Tests all services: LLM, TTS, ASR, VRAM swapping.
Run with: pytest tests/integration/test_full_pipeline.py -v

Prerequisites:
- Ray cluster running: bash scripts/start_cluster.sh
- Services deployed: .venv/bin/python -m scripts.deploy_services
- GPU available (RTX 4090 or similar)
"""

from __future__ import annotations

import os
import subprocess
import time
import pytest
import httpx

BASE_URL = os.environ.get("RAY_BASE_URL", "http://localhost:8000")
TIMEOUT = 300


@pytest.fixture(scope="session")
def client():
    """HTTP client for the Ray Serve API."""
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="session")
def ensure_cluster():
    """Ensure Ray cluster is running."""
    result = subprocess.run(
        [".venv/bin/ray", "status"], capture_output=True, text=True, timeout=5,
    )
    if "node" not in result.stdout:
        pytest.skip("Ray cluster not running. Start with: bash scripts/start_cluster.sh")


@pytest.fixture(scope="session")
def ensure_served(ensure_cluster):
    """Ensure at least one service is deployed."""
    try:
        resp = httpx.get(f"{BASE_URL}/-/routes", timeout=5)
        if resp.status_code != 200 or not resp.json():
            pytest.skip("Services not deployed. Run: .venv/bin/python -m scripts.deploy_services")
    except httpx.ConnectError:
        pytest.skip("Ray Serve not reachable. Is the cluster running?")


def get_vram_free_mb() -> int:
    """Get free VRAM in MB via torch.cuda (Ray-native)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            return int(total - reserved)
    except Exception:
        pass
    return 0


# =============================================================================
# LLM Tests
# =============================================================================

class TestLLM:
    def test_chat_simple(self, client, ensure_served):
        """Simple single-turn chat via direct LLM endpoint."""
        resp = client.post("/llm/", json={
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "model": "qwen3.5-2b-ud-q4_k_xl",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert "4" in content

    def test_chat_multiturn(self, client, ensure_served):
        """Multi-turn conversation - verify the API accepts message arrays."""
        resp = client.post("/llm/", json={
            "messages": [
                {"role": "user", "content": "My name is TestBot."},
                {"role": "assistant", "content": "Nice to meet you, TestBot!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "model": "qwen3.5-2b-ud-q4_k_xl",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        # Model may put content in reasoning_content due to thinking mode
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        assert len(content) > 0, "Model should produce some output"


# =============================================================================
# CPU TTS Tests
# =============================================================================

class TestCPU_TTS:
    def test_espeak(self, client, ensure_served):
        """eSpeak should produce audio on CPU."""
        resp = client.post("/tts/espeak", json={
            "input": "Hello, this is a test.",
            "voice": "en",
        })
        assert resp.status_code == 200
        assert len(resp.content) > 1000

    @pytest.mark.skip(reason="Kokoro model needs to be downloaded first")
    def test_kokoro(self, client, ensure_served):
        """Kokoro TTS on CPU."""
        resp = client.post("/tts/kokoro", json={
            "input": "Testing one two three.",
            "voice": "af_bella",
        })
        assert resp.status_code == 200
        assert len(resp.content) > 1000


# =============================================================================
# CPU ASR Tests
# =============================================================================

class TestCPU_ASR:
    def test_whisper_with_tts(self, client, ensure_served):
        """Generate speech with espeak, then transcribe with whisper."""
        # Generate speech
        tts_resp = client.post("/tts/espeak", json={
            "input": "The quick brown fox jumps over the lazy dog.",
        })
        assert tts_resp.status_code == 200

        # Transcribe it
        asr_resp = client.post(
            "/asr/whisper",
            files={"file": ("test.wav", tts_resp.content)},
            data={"model": "tiny"},
        )
        assert asr_resp.status_code == 200
        text = asr_resp.json()["text"].lower()
        # Whisper should get at least some of the words right
        assert any(word in text for word in ["quick", "brown", "fox", "dog"])


# =============================================================================
# VRAM Swap Tests
# =============================================================================

def _await_serve(resp):
    """Resolve a Serve DeploymentResponse synchronously."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(resp)
    finally:
        loop.close()


class TestVRAMSwap:
    def _get_llm_handle(self):
        """Get LLM deployment handle."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import ray
        from ray import serve
        ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
        return serve.get_deployment_handle("llm", "llm")

    def test_load_llm_via_handle(self, ensure_served):
        """Load LLM model, verify VRAM drops, unload, verify VRAM recovers."""
        handle = self._get_llm_handle()

        # Unload any existing model first
        _await_serve(handle.options(method_name="unload_model").remote())
        time.sleep(3)

        vram_before = get_vram_free_mb()

        # Load model via handle
        _await_serve(
            handle.options(method_name="load_model").remote("qwen3.5-2b-ud-q4_k_xl")
        )

        time.sleep(5)
        vram_loaded = get_vram_free_mb()
        assert vram_loaded < vram_before, (
            f"VRAM didn't drop after loading model: {vram_before}MB -> {vram_loaded}MB"
        )

        # Unload
        _await_serve(handle.options(method_name="unload_model").remote())

        time.sleep(5)
        vram_after = get_vram_free_mb()
        assert vram_after >= vram_before - 500, (
            f"VRAM didn't recover after unload: {vram_before}MB -> {vram_after}MB (ghost VRAM?)"
        )

    def test_cpu_tts_during_gpu_load(self, client, ensure_served):
        """CPU TTS should work while GPU is occupied."""
        handle = self._get_llm_handle()

        # Load LLM on GPU
        _await_serve(
            handle.options(method_name="load_model").remote("qwen3.5-2b-ud-q4_k_xl")
        )

        # CPU TTS should still work
        resp = client.post("/tts/espeak", json={"input": "Still working."})
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Cleanup
        _await_serve(handle.options(method_name="unload_model").remote())


# =============================================================================
# ComfyUI Tests
# =============================================================================

class TestComfyUI:
    @pytest.mark.skip(reason="Requires ComfyUI workflow JSON - enable when ready")
    def test_generate_image(self, client, ensure_served):
        """Submit a simple workflow to ComfyUI."""
        workflow = {}
        resp = client.post("/comfyui/prompt", json={"prompt": workflow})
        assert resp.status_code == 200

    @pytest.mark.skip(reason="ComfyUI needs to be loaded on GPU first")
    def test_comfyui_health(self, client, ensure_served):
        """ComfyUI WebUI should be accessible through the proxy."""
        resp = client.get("/comfyui/")
        assert resp.status_code in (200, 302)


# =============================================================================
# 3D Generation Tests
# =============================================================================

class Test3DGeneration:
    @pytest.mark.skip(reason="Requires test image - enable when ready")
    def test_trellis(self, client, ensure_served):
        """Generate 3D mesh from test image."""
        test_image = b""
        resp = client.post(
            "/3d/trellis",
            files={"image": ("test.png", test_image)},
            data={"output_format": "glb", "resolution": "64"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "model/gltf-binary"
        assert len(resp.content) > 1000
