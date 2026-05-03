"""E2E test suite for Tech Noir Ray infrastructure.

Tests all services end-to-end: LLM, TTS, ASR, VRAM swapping,
ComfyUI image generation, TRELLIS 3D, AniGen rigged 3D,
ACE-STEP music, and See-Through decomposition.

Run with: pytest tests/integration/test_full_pipeline.py -v

Prerequisites:
- Ray cluster running: task boot:ray
- Services deployed (happens automatically via boot:ray)
- GPU available (RTX 4090 or similar)
- Models downloaded: task models:pull

Route reference (from deploy_services.py):
  /llm/*               - LLM (llama.cpp)
  /tts/espeak/*         - eSpeak TTS (CPU)
  /tts/kokoro/*         - Kokoro TTS (CPU)
  /tts/index-tts/*      - IndexTTS (GPU)
  /tts/qwen-tts/*       - Qwen3-TTS (GPU)
  /tts/vibevoice/*      - VibeVoice TTS (GPU)
  /tts/gpt-sovits/*     - GPT-SoVITS (GPU)
  /asr/whisper/*        - Faster-Whisper (CPU)
  /asr/vibevoice/*      - VibeVoice ASR (GPU)
  /asr/qwen/*           - Qwen ASR (GPU)
  /comfyui/*            - ComfyUI (GPU, WebUI)
  /3d/trellis/*         - TRELLIS.2 (GPU)
  /3d/anigen/*          - AniGen (GPU)
  /creative/see-through/* - See-Through (GPU)
  /music/ace-step/*     - ACE-STEP music (GPU)
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import time
from pathlib import Path

import pytest
import httpx

BASE_URL = os.environ.get("RAY_BASE_URL", "http://localhost:18800")
TIMEOUT = 300

RAY_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Test image generation utility
# ---------------------------------------------------------------------------

def _make_test_png(width: int = 256, height: int = 256) -> bytes:
    """Generate a minimal valid PNG image (checkerboard pattern)."""
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            if (x // 32 + y // 32) % 2 == 0:
                raw_rows.extend([200, 100, 50])
            else:
                raw_rows.extend([50, 100, 200])

    compressed = zlib.compress(bytes(raw_rows))
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _make_test_wav(duration_s: float = 1.0, sample_rate: int = 22050,
                   freq: int = 440) -> bytes:
    """Generate a minimal WAV file with a sine tone."""
    import math
    num_samples = int(sample_rate * duration_s)
    data = bytearray()
    for i in range(num_samples):
        sample = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
        data.extend(struct.pack("<h", max(-32768, min(32767, sample))))

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                           sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(bytes(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Serve handle helpers
# ---------------------------------------------------------------------------

def _await_serve(resp):
    """Resolve a Serve DeploymentResponse synchronously."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(resp)
    finally:
        loop.close()


def _get_handle(deployment_name: str, app_name: str):
    """Get a Ray Serve deployment handle."""
    import sys
    sys.path.insert(0, str(RAY_ROOT))
    import ray
    from ray import serve
    ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
    return serve.get_deployment_handle(deployment_name, app_name)


def _load_service(deployment_name: str, app_name: str, model_name: str):
    """Load a GPU model via deployment handle. Idempotent."""
    handle = _get_handle(deployment_name, app_name)
    _await_serve(handle.options(method_name="load_model").remote(model_name))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """HTTP client for the Ray Serve API (port 18800)."""
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="session")
def ensure_cluster():
    """Ensure Ray cluster is running."""
    ray_bin = str(RAY_ROOT / ".venv" / "bin" / "ray")
    result = subprocess.run(
        [ray_bin, "status"], capture_output=True, text=True, timeout=5,
    )
    if "node" not in result.stdout:
        pytest.skip("Ray cluster not running. Start with: task boot:ray")


@pytest.fixture(scope="session")
def ensure_served(ensure_cluster):
    """Ensure at least one service is deployed."""
    try:
        resp = httpx.get(f"{BASE_URL}/-/routes", timeout=5)
        if resp.status_code != 200 or not resp.json():
            pytest.skip("Services not deployed. Run: task boot:ray")
    except httpx.ConnectError:
        pytest.skip("Ray Serve not reachable. Is the cluster running?")


@pytest.fixture(scope="session")
def test_png():
    """Reusable test PNG image (256x256 checkerboard)."""
    return _make_test_png(256, 256)


@pytest.fixture(scope="session")
def test_wav():
    """Reusable test WAV audio (1s, 440Hz sine)."""
    return _make_test_wav(1.0, 22050, 440)


@pytest.fixture(scope="session")
def llm_loaded(ensure_served):
    """Load the LLM model before tests that need it."""
    _load_service("llm", "llm", "qwen3.5-2b-ud-q4_k_xl")
    time.sleep(2)  # wait for llama-server to fully start


@pytest.fixture(scope="session")
def trellis_loaded(ensure_served):
    """Load TRELLIS model (stops ComfyUI, starts Docker worker, loads model)."""
    _unload_gpu_service("comfyui", "comfyui")
    _start_docker_worker("trellis")
    _load_service("trellis", "trellis", "trellis")


@pytest.fixture(scope="session")
def anigen_loaded(ensure_served):
    """Load AniGen model (stops TRELLIS worker, starts AniGen worker)."""
    _stop_docker_worker("trellis")
    _start_docker_worker("anigen")
    _load_service("anigen", "anigen", "anigen")


@pytest.fixture(scope="session")
def ace_step_loaded(ensure_served):
    """Load ACE-STEP model (stops Docker workers, unloads GPU services to free VRAM)."""
    _stop_docker_worker("trellis")
    _stop_docker_worker("anigen")
    _unload_gpu_service("llm", "llm")
    _load_service("ace_step", "ace_step", "ace-step")


@pytest.fixture(scope="session")
def see_through_loaded(ensure_served):
    """Load See-Through model (stops Docker workers, unloads ACE-Step)."""
    _stop_docker_worker("anigen")
    _unload_gpu_service("ace_step", "ace_step")
    _load_service("see_through", "see_through", "see-through")


def get_vram_free_mb() -> int:
    """Get free VRAM via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    return int(result.stdout.strip().split("\n")[0].strip())


def _stop_docker_worker(profile: str) -> None:
    """Stop and remove a Docker worker container to free VRAM."""
    compose_file = str(RAY_ROOT / "infra" / "docker" / "compose.workers.yaml")
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "--profile", profile, "down"],
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(2)


def _start_docker_worker(profile: str) -> None:
    """Start a Docker worker container and wait for health check."""
    compose_file = str(RAY_ROOT / "infra" / "docker" / "compose.workers.yaml")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "--profile", profile, "up", "-d"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start {profile} worker: {result.stderr}")

    # Wait for worker to be ready
    port = {"trellis": 18401, "anigen": 18402, "vibevoice": 18403}[profile]
    for _ in range(30):
        import urllib.request
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Docker worker {profile} did not become healthy within 60s")


def _unload_gpu_service(deployment_name: str, app_name: str) -> None:
    """Unload a GPU model via Serve handle to free VRAM."""
    handle = _get_handle(deployment_name, app_name)
    try:
        _await_serve(handle.options(method_name="unload_model").remote())
    except Exception:
        pass
    time.sleep(2)


# =============================================================================
# LLM Tests
# =============================================================================

class TestLLM:
    def test_chat_simple(self, client, llm_loaded):
        """Simple single-turn chat via direct LLM endpoint."""
        resp = client.post("/llm/", json={
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "model": "qwen3.5-2b-ud-q4_k_xl",
        }, timeout=120)
        assert resp.status_code == 200, f"LLM failed: {resp.status_code} {resp.text[:500]}"
        data = resp.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert "4" in content

    def test_chat_multiturn(self, client, llm_loaded):
        """Multi-turn conversation - verify the API accepts message arrays."""
        resp = client.post("/llm/", json={
            "messages": [
                {"role": "user", "content": "My name is TestBot."},
                {"role": "assistant", "content": "Nice to meet you, TestBot!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "model": "qwen3.5-2b-ud-q4_k_xl",
        }, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
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
        assert resp.content[:4] == b"RIFF"
        assert b"WAVE" in resp.content[:12]


# =============================================================================
# CPU ASR Tests
# =============================================================================

class TestCPU_ASR:
    def test_whisper_with_tts(self, client, ensure_served):
        """Generate speech with espeak, then transcribe with whisper."""
        tts_resp = client.post("/tts/espeak", json={
            "input": "The quick brown fox jumps over the lazy dog.",
        })
        assert tts_resp.status_code == 200

        asr_resp = client.post(
            "/asr/whisper",
            files={"file": ("test.wav", tts_resp.content)},
            data={"model": "tiny"},
        )
        assert asr_resp.status_code == 200, (
            f"ASR failed: {asr_resp.status_code} {asr_resp.text[:500]}"
        )
        text = asr_resp.json()["text"].lower()
        assert any(word in text for word in ["quick", "brown", "fox", "dog"])


# =============================================================================
# VRAM Swap Tests
# =============================================================================

class TestVRAMSwap:
    def test_load_llm_via_handle(self, llm_loaded):
        """Load/unload cycle via handle — verify no crash."""
        handle = _get_handle("llm", "llm")

        # Unload — should not crash
        _await_serve(handle.options(method_name="unload_model").remote())

        # Reload — should not crash
        _await_serve(
            handle.options(method_name="load_model").remote("qwen3.5-2b-ud-q4_k_xl")
        )

    def test_cpu_tts_during_gpu_load(self, client, llm_loaded):
        """CPU TTS should work while GPU is occupied."""
        resp = client.post("/tts/espeak", json={"input": "Still working."})
        assert resp.status_code == 200
        assert len(resp.content) > 0


# =============================================================================
# ComfyUI Tests
# =============================================================================

class TestComfyUI:
    """Test ComfyUI workflow execution through Ray Serve proxy.

    ComfyUI is deployed at /comfyui/* on Ray Serve (port 18800).
    It auto-starts on first request and runs as a managed subprocess.
    """

    # Fast txt2img workflow: SDXL base + DMD2 4-step LoRA.
    # DMD2 distillation allows high-quality images in just 4 steps.
    WORKFLOW = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            }
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "loras/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["1", 0],
                "clip": ["1", 1],
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a beautiful sunset over mountains, high quality photo",
                "clip": ["2", 1]
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, distorted",
                "clip": ["2", 1]
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 4,
                "cfg": 1.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "test_e2e",
                "images": ["7", 0]
            }
        }
    }

    def test_comfyui_starts(self, client, ensure_served):
        """ComfyUI should start on first request and return 200."""
        resp = client.get("/comfyui/", timeout=180)
        assert resp.status_code in (200, 302), (
            f"ComfyUI not accessible: {resp.status_code} {resp.text[:200]}"
        )

    def test_generate_image(self, client, ensure_served):
        """Submit a fast txt2img workflow to ComfyUI (SDXL + DMD2 4-step)."""
        # Ensure ComfyUI is running
        client.get("/comfyui/", timeout=180)

        resp = client.post(
            "/comfyui/prompt",
            json={"prompt": self.WORKFLOW},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Workflow submit failed: {resp.status_code} {resp.text[:500]}"
        )
        data = resp.json()
        assert "prompt_id" in data, f"No prompt_id in response: {data}"

        prompt_id = data["prompt_id"]

        # Poll until execution completes (60 iterations × 5s = 5min max)
        last_status = {}
        for _ in range(60):
            time.sleep(5)
            hist_resp = client.get(f"/comfyui/history/{prompt_id}", timeout=30)
            if hist_resp.status_code != 200:
                continue
            history = hist_resp.json()
            if prompt_id in history:
                last_status = history[prompt_id].get("status", {})
                if last_status.get("status_str") == "error":
                    pytest.fail(f"Workflow error: {last_status}")
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    has_images = any("images" in v for v in outputs.values())
                    assert has_images, f"No image outputs: {outputs}"
                    return

        pytest.fail(
            f"Workflow did not complete in time (prompt_id={prompt_id}, "
            f"status={last_status})"
        )


# =============================================================================
# 3D Generation — TRELLIS
# =============================================================================

class TestTRELLIS:
    """Test TRELLIS.2 image-to-3D mesh generation.

    Deployed at /3d/trellis on Ray Serve. Uses HTTPToolMixin — sends
    requests to a Docker worker container at port 18401.
    """

    def test_generate_glb(self, client, trellis_loaded, test_png):
        """Generate a GLB mesh from a test image."""
        resp = client.post(
            "/3d/trellis/",
            files={"image": ("test.png", test_png, "image/png")},
            data={"output_format": "glb", "resolution": "256"},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"TRELLIS failed: {resp.status_code} {resp.text[:500]}"
        )
        assert resp.headers.get("content-type") in (
            "model/gltf-binary", "application/octet-stream"
        ), f"Wrong content type: {resp.headers.get('content-type')}"
        assert len(resp.content) > 100, (
            f"GLB output too small ({len(resp.content)} bytes)"
        )
        # GLB magic: bytes 0-3 = 0x46546C67 ('glTF'), bytes 4-7 = version (uint32 LE)
        assert resp.content[0:4] == b"glTF", "Not a valid GLB file"


# =============================================================================
# 3D Generation — AniGen
# =============================================================================

class TestAniGen:
    """Test AniGen rigged 3D mesh generation.

    Deployed at /3d/anigen on Ray Serve. Uses HTTPToolMixin — sends
    requests to a Docker worker container at port 18402.
    """

    def test_generate_rigged_glb(self, client, anigen_loaded, test_png):
        """Generate a rigged GLB mesh from a test image."""
        resp = client.post(
            "/3d/anigen/",
            files={"image": ("test.png", test_png, "image/png")},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"AniGen failed: {resp.status_code} {resp.text[:500]}"
        )
        assert resp.headers.get("content-type") in (
            "model/gltf-binary", "application/octet-stream"
        ), f"Wrong content type: {resp.headers.get('content-type')}"
        assert len(resp.content) > 100, (
            f"GLB output too small ({len(resp.content)} bytes)"
        )


# =============================================================================
# Music Generation — ACE-STEP
# =============================================================================

class TestACEStep:
    """Test ACE-STEP text-to-music generation.

    Deployed at /music/ace-step on Ray Serve. Uses CLIToolMixin subprocess.
    """

    def test_generate_music(self, client, ace_step_loaded):
        """Generate a short music clip from a text prompt."""
        resp = client.post(
            "/music/ace-step/",
            json={
                "prompt": "A calm ambient piano piece with soft strings",
                "duration": 10,
                "bpm": 80,
                "instrumental": True,
                "audio_format": "wav",
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"ACE-STEP failed: {resp.status_code} {resp.text[:500]}"
        )
        assert len(resp.content) > 1000, (
            f"Audio output too small ({len(resp.content)} bytes)"
        )
        assert resp.content[:4] == b"RIFF", "Output is not a WAV file"
        assert b"WAVE" in resp.content[:12], "Output is not a WAV file"


# =============================================================================
# Layer Decomposition — See-Through
# =============================================================================

class TestSeeThrough:
    """Test See-Through layer decomposition.

    Deployed at /creative/see-through on Ray Serve. Uses CLIToolMixin.
    """

    def test_decompose(self, client, see_through_loaded, test_png):
        """Decompose an image into layers."""
        resp = client.post(
            "/creative/see-through/",
            files={"image": ("test.png", test_png, "image/png")},
            data={"resolution": "640", "save_to_psd": "true"},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"See-Through failed: {resp.status_code} {resp.text[:500]}"
        )
        data = resp.json()
        assert "layers" in data, f"No layers in response: {data}"
        assert isinstance(data["layers"], list), f"layers should be a list: {data}"
        assert "has_psd" in data, f"No has_psd in response: {data}"
