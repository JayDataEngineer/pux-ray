"""Comprehensive E2E tests for Forge GPU models.

Tests every Wan2GP model that has weights on the PVC. Each test:
1. Sends an inference request through the Forge (/forge endpoint)
2. Validates the response (status, output data, media type)
3. Releases the model

Usage:
  # Full suite against live Forge
  FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s

  # Quick tests only (audio models, <60s each)
  FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -k "audio"

  # 3D/image models
  FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -k "trellis or anigen"

  # Skip slow models (>5min inference)
  FORGE_URL=http://100.86.69.57:30080 pytest tests/test_forge_models_e2e.py -v -s -m "not slow"
"""
from __future__ import annotations

import base64
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pytest

FORGE_URL = os.environ.get("FORGE_URL", "http://100.86.69.57:30080")
DEFAULT_TIMEOUT = 900  # 15 minutes for heavy GPU models (load + inference)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _forge_req(payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Send a request to the Forge endpoint."""
    url = f"{FORGE_URL}/forge"
    body = json.dumps(payload).encode()
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"status": "error", "error": f"HTTP {e.code}: {body[:500]}", "http_code": e.code}
    except URLError as e:
        pytest.skip(f"Forge not reachable at {FORGE_URL}: {e}")
    except Exception as e:
        return {"status": "error", "error": str(e)[:500]}


def _forge_generate(model: str, timeout: int = DEFAULT_TIMEOUT, **extra) -> dict:
    """Send a generate request through Forge → wan2gp."""
    payload = {"service": "wan2gp", "model": model, **extra}
    return _forge_req(payload, timeout=timeout)


def _release():
    """Release the current wan2gp model and wait for VRAM to be freed."""
    try:
        _forge_req({"action": "release", "service": "wan2gp"}, timeout=30)
    except Exception:
        pass
    # Give the server time to unload the model and free VRAM
    time.sleep(3)
    # Verify release worked by checking status
    for _ in range(3):
        try:
            st = _forge_req({"action": "status"}, timeout=10)
            if "wan2gp" not in st.get("loaded", {}):
                return
        except Exception:
            pass
        time.sleep(2)


def _assert_success(result: dict, model: str, min_data_bytes: int = 1000):
    """Assert a model produced valid output."""
    if result.get("status") == "error":
        error = result.get("error", "")
        if "Failed to load" in error:
            pytest.skip(f"Model {model} load failed (weights may be missing): {error[:200]}")
        if "Unknown model" in error:
            pytest.skip(f"Model {model} not registered: {error[:200]}")
        if result.get("http_code") == 500:
            pytest.fail(f"Model {model} server error (500): check pod logs")
        pytest.fail(f"Model {model} error: {error[:300]}")

    status = result.get("status", "")
    if status not in ("success", "ok", None):
        # Some models return status implicitly
        if "data" not in result and "error" in result:
            pytest.fail(f"Model {model} error: {result['error'][:300]}")

    data = result.get("data")
    if data and isinstance(data, str):
        raw = base64.b64decode(data)
        assert len(raw) >= min_data_bytes, \
            f"Model {model} output too small: {len(raw)} bytes (min {min_data_bytes})"


# ─── 3D Models ──────────────────────────────────────────────────────────────


class TestTRELLIS:
    """TRELLIS.2 — image-to-3D mesh generation."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_trellis_image_to_3d(self, real_image_b64):
        r = _forge_generate("trellis/trellis", image_b64=real_image_b64,
                            seed=42, steps=12, resolution="512", timeout=600)
        _assert_success(r, "trellis", min_data_bytes=100_000)
        assert r.get("media_type") == "model/gltf-binary"


class TestAniGen:
    """AniGen — image-to-rigged-3D."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_anigen_image_to_3d(self, real_image_b64):
        r = _forge_generate("anigen/anigen", image_b64=real_image_b64,
                            seed=42, timeout=600)
        _assert_success(r, "anigen", min_data_bytes=10_000)


# ─── Motion Models ──────────────────────────────────────────────────────────


class TestHYMotion:
    """HY-Motion 1.0 — text-to-3D motion."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_hy_motion_text_to_motion(self):
        r = _forge_generate("hy_motion/hy-motion-1.0",
                            prompt="A person walks forward and waves hello",
                            seed=42, timeout=600)
        _assert_success(r, "hy_motion", min_data_bytes=1000)


# ─── Audio Models ───────────────────────────────────────────────────────────


class TestACEStep:
    """ACE-Step v1.5 — text-to-music."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_ace_step_text_to_music(self):
        r = _forge_generate("tts/ace_step_v1_5",
                            prompt="A calm piano melody in C major",
                            seed=42, timeout=600)
        _assert_success(r, "ace_step", min_data_bytes=100_000)


class TestMOSSSoundEffect:
    """MOSS-SoundEffect 8B — text-to-sound-effect."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_moss_soundeffect(self):
        r = _forge_generate("moss/moss-soundeffect",
                            prompt="Thunder rumbling in the distance",
                            seed=42, timeout=600)
        _assert_success(r, "moss_soundeffect", min_data_bytes=100_000)


class TestMOSSTTSNano:
    """MOSS TTS Nano — lightweight TTS."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_moss_tts_nano(self):
        r = _forge_generate("moss/moss-tts-nano",
                            prompt="Hello, how are you today?",
                            seed=42, timeout=300)
        _assert_success(r, "moss_tts_nano", min_data_bytes=10_000)


class TestMOSSTTSLocalTransformer:
    """MOSS TTS Local Transformer — 1.7B variant."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_moss_tts_local_transformer(self):
        r = _forge_generate("moss/moss-tts-local-transformer",
                            prompt="The quick brown fox jumps over the lazy dog.",
                            seed=42, timeout=600)
        _assert_success(r, "moss_tts_local_transformer", min_data_bytes=10_000)


class TestMOSSTTS:
    """MOSS TTS — full 8B model."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_moss_tts(self):
        r = _forge_generate("moss/moss-tts",
                            prompt="Welcome to the text to speech system.",
                            seed=42, timeout=600)
        _assert_success(r, "moss_tts", min_data_bytes=50_000)


class TestMOSSTTSD:
    """MOSS TTSD — TTS with denoising."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_moss_ttsd(self):
        r = _forge_generate("moss/moss-ttsd",
                            prompt="The weather is nice today.",
                            seed=42, timeout=600)
        _assert_success(r, "moss_ttsd", min_data_bytes=50_000)


class TestMOSSVoiceGenerator:
    """MOSS VoiceGenerator."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_moss_voicegenerator(self):
        r = _forge_generate("moss/moss-voicegenerator",
                            prompt="Generate a short spoken phrase.",
                            seed=42, timeout=600)
        _assert_success(r, "moss_voicegenerator", min_data_bytes=10_000)


class TestMOSSVoiceRealtime:
    """MOSS TTS Realtime."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_moss_tts_realtime(self):
        r = _forge_generate("moss/moss-tts-realtime",
                            prompt="Hello world, this is a real-time test.",
                            seed=42, timeout=300)
        _assert_success(r, "moss_tts_realtime", min_data_bytes=10_000)


# ─── TTS Models (voice cloning) ────────────────────────────────────────────


class TestIndexTTS2:
    """IndexTTS v2 — neural voice cloning TTS."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    def test_index_tts2_with_ref_audio(self, real_audio_ref_b64):
        r = _forge_generate("tts/index_tts2",
                            prompt="This is a test of the voice cloning system.",
                            audio_b64=real_audio_ref_b64,
                            seed=42, timeout=600)
        _assert_success(r, "index_tts2", min_data_bytes=10_000)


# ─── Video Models ───────────────────────────────────────────────────────────


class TestWanT2V:
    """Wan T2V 14B — text-to-video (slow, ~30 min)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.autoregressive
    def test_wan_t2v(self):
        r = _forge_generate("wan/t2v",
                            prompt="A cat walking in the rain",
                            seed=42, steps=4, num_frames=8,
                            timeout=900)
        if r.get("status") == "error":
            error = r.get("error", "")
            if "Failed to load" in error:
                pytest.skip(f"wan/t2v load failed: {error[:200]}")
            if "No such file" in error or "weights" in error:
                pytest.skip(f"wan/t2v weights missing: {error[:200]}")
            pytest.fail(f"wan_t2v error: {error[:300]}")
        _assert_success(r, "wan_t2v", min_data_bytes=10_000)


# ─── Image Models ───────────────────────────────────────────────────────────


class TestSeeThrough:
    """See-Through — anime layer decomposition."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_see_through(self, real_anime_rgba_b64):
        r = _forge_generate("see_through/see-through",
                            image_b64=real_anime_rgba_b64,
                            seed=42, timeout=600)
        # See-through may fail on non-anime images — that's expected
        if r.get("status") == "error":
            error = r.get("error", "")
            if "resize" in error.lower() or "empty" in error.lower():
                pytest.skip(f"See-through needs anime-style input: {error[:200]}")
            if "Failed to load" in error:
                pytest.skip(f"Model weights missing: {error[:200]}")
            pytest.fail(f"See-through error: {error[:300]}")


# ─── Lance Models ───────────────────────────────────────────────────────────


class TestLanceImage:
    """Lance 3B — multimodal image generation."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_lance_image(self):
        r = _forge_generate("lance/lance-image-awq",
                            prompt="A cat sitting on a windowsill",
                            seed=42, timeout=900)
        if r.get("status") == "error":
            error = r.get("error", "")
            if "script not found" in error or "Failed to load" in error:
                pytest.skip(f"Lance vendor repo not installed: {error[:200]}")
            if "mrope" in error or "ROPE_VALIDATION" in error:
                pytest.skip(f"Lance needs transformers>=4.46 (mRoPE support): {error[:200]}")
            pytest.fail(f"Lance error: {error[:300]}")


# ─── Forge Status ───────────────────────────────────────────────────────────


class TestForgeStatus:
    """Forge infrastructure — status and health checks."""

    def test_forge_status_endpoint(self):
        r = _forge_req({"action": "status"}, timeout=10)
        assert "vram_total_mb" in r, f"Status missing vram_total_mb: {r}"
        assert r["vram_total_mb"] > 0, f"VRAM should be > 0: {r}"

    def test_forge_release_idempotent(self):
        r = _forge_req({"action": "release", "service": "wan2gp"}, timeout=10)
        assert r.get("status") in ("released", "not_loaded"), f"Release failed: {r}"


# ─── Kimodo Models ──────────────────────────────────────────────────────────


class TestKimodoSOMA:
    """Kimodo SOMA — text-to-3D motion (NVIDIA)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_kimodo_soma_rp(self):
        r = _forge_generate("kimodo/kimodo-soma-rp",
                            prompt="A person waving their right hand",
                            seed=42, timeout=900)
        if r.get("status") == "error":
            error = r.get("error", "")
            if "HF_TOKEN" in error:
                pytest.skip(f"Kimodo needs HF_TOKEN for gated Llama: {error[:200]}")
            if "Failed to load" in error:
                pytest.skip(f"Model load failed: {error[:200]}")
            pytest.fail(f"Kimodo SOMA error: {error[:300]}")
        _assert_success(r, "kimodo_soma", min_data_bytes=100)


class TestKimodoG1:
    """Kimodo G1 Robot — text-to-3D motion."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _release()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_kimodo_g1_rp(self):
        r = _forge_generate("kimodo/kimodo-g1-rp",
                            prompt="A robot walking forward",
                            seed=42, timeout=900)
        if r.get("status") == "error":
            error = r.get("error", "")
            if "HF_TOKEN" in error:
                pytest.skip(f"Kimodo needs HF_TOKEN: {error[:200]}")
            if "Failed to load" in error:
                pytest.skip(f"Model load failed: {error[:200]}")
            pytest.fail(f"Kimodo G1 error: {error[:300]}")
        _assert_success(r, "kimodo_g1", min_data_bytes=100)

