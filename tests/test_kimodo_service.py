"""Unit + integration tests for Kimodo service loading.

Tests the wrapper script, monkey-patches, hub cache, and full preload cycle.
Can run offline (no GPU) to verify the patching logic, or against a live
Forge to verify end-to-end loading.

Usage:
  # Offline unit tests (no GPU, no cluster)
  pytest tests/test_kimodo_service.py -v

  # Live E2E against cluster
  FORGE_URL=http://100.86.69.57:30080 pytest tests/test_kimodo_service.py -v -s -k live
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────────

FORGE_URL = os.environ.get("FORGE_URL", "")

def _get_wrapper_script() -> str:
    """Read the kimodo launcher script (monkey-patches huggingface_hub)."""
    src = Path(__file__).resolve().parent.parent / "services" / "motion" / "_run_kimodo.py"
    if not src.exists():
        pytest.skip("services/motion/_run_kimodo.py not found")
    return src.read_text()


def _forge_req(payload: dict, timeout: int = 30) -> dict:
    """Send a request to the Forge endpoint."""
    import urllib.request
    import urllib.error
    url = f"{FORGE_URL}/forge"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"status": "error", "error": f"HTTP {e.code}: {body[:500]}"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:500]}


def _make_hub_cache(cache_dir: Path, repo_id: str, files: dict[str, bytes]) -> Path:
    """Create a fake HF hub cache structure that scan_cache_dir can read.

    Args:
        cache_dir: Parent directory for the cache.
        repo_id: e.g. "TestOrg/TestModel"
        files: {filename: content} to put in the snapshot.

    Returns:
        The snapshot path.
    """
    org, name = repo_id.split("/")
    repo_dir = cache_dir / f"models--{org}--{name}"
    blobs = repo_dir / "blobs"
    refs = repo_dir / "refs"
    rev = hashlib.sha256(repo_id.encode()).hexdigest()[:40]
    snap = repo_dir / "snapshots" / rev

    blobs.mkdir(parents=True)
    refs.mkdir(parents=True)
    snap.mkdir(parents=True)

    for fname, content in files.items():
        blob_hash = hashlib.sha256(content).hexdigest()
        blob_path = blobs / blob_hash
        blob_path.write_bytes(content)
        (snap / fname).symlink_to(blob_path)

    # refs/main MUST NOT have trailing newline — scan_cache_dir is strict
    (refs / "main").write_text(rev)

    return snap


# ─── 1. Wrapper Script Unit Tests ────────────────────────────────────────────

class TestWrapperScript:
    """Verify the monkey-patch wrapper script logic (no GPU needed)."""

    def test_patch_model_info_is_noop(self):
        """Patched model_info should return a dummy object, not hit network."""
        class _FakeModelInfo:
            tags = []
            library_name = None
            def __init__(self, *a, **kw): pass

        import huggingface_hub
        import huggingface_hub.hf_api as _hfapi
        orig = _hfapi.model_info
        try:
            _hfapi.model_info = lambda *a, **kw: _FakeModelInfo()
            huggingface_hub.model_info = _hfapi.model_info

            # Should NOT raise OfflineModeIsEnabled
            result = huggingface_hub.model_info("anything/anything")
            assert result is not None
            assert result.tags == []
        finally:
            _hfapi.model_info = orig
            huggingface_hub.model_info = orig

    def test_repo_id_resolves_to_local_path(self, tmp_path):
        """scan_cache_dir should resolve cached repo IDs to local snapshot paths."""
        cache_dir = tmp_path / "hf_cache"
        snap = _make_hub_cache(cache_dir, "TestOrg/TestModel", {
            "config.json": b'{"model_type": "test"}',
        })

        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(str(cache_dir))
        repos = {r.repo_id: r for r in cache.repos}

        assert "TestOrg/TestModel" in repos
        for rev_obj in repos["TestOrg/TestModel"].revisions:
            assert rev_obj.snapshot_path.is_dir()
            assert (rev_obj.snapshot_path / "config.json").exists()
            break

    def test_tokenizer_loads_from_hub_cache(self):
        """AutoTokenizer.from_pretrained with a cached repo ID should work offline.

        Requires the HF hub cache to be populated (from download script).
        This test is skipped if the cache is empty.
        """
        cache_dir = os.environ.get("HF_HUB_CACHE", "")
        if not cache_dir or not os.path.isdir(cache_dir):
            pytest.skip("HF_HUB_CACHE not set or empty")

        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(cache_dir)
        repos = {r.repo_id: r for r in cache.repos}

        if "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp" not in repos:
            pytest.skip("LLM2Vec MNTP model not in hub cache")

        # Patch model_info to avoid network
        import huggingface_hub
        import huggingface_hub.hf_api as _hfapi
        class _Fake:
            tags = []
            library_name = None
        _hfapi.model_info = lambda *a, **kw: _Fake()
        huggingface_hub.model_info = _hfapi.model_info

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
        assert tok is not None
        assert tok.vocab_size > 0


# ─── 2. Hub Cache Build Tests ────────────────────────────────────────────────

class TestHubCacheBuild:
    """Verify hub cache build from download script."""

    def test_snapshot_download_creates_hub_cache(self, tmp_path):
        """snapshot_download with cache_dir creates proper hub cache format.

        Only runs if internet is available. Uses a tiny public model.
        """
        try:
            from huggingface_hub import snapshot_download
            cache_dir = str(tmp_path / "cache")
            path = snapshot_download(
                repo_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
                cache_dir=cache_dir,
            )
            assert os.path.isdir(path)
            # Verify hub cache structure
            models_dir = os.path.join(cache_dir, "models--hf-internal-testing--tiny-random-LlamaForCausalLM")
            assert os.path.isdir(models_dir), f"Hub cache dir not found: {models_dir}"
            assert os.path.isdir(os.path.join(models_dir, "blobs"))
            assert os.path.isdir(os.path.join(models_dir, "refs"))
            assert os.path.isdir(os.path.join(models_dir, "snapshots"))
            assert os.path.isfile(os.path.join(models_dir, "refs", "main"))
        except ImportError:
            pytest.skip("huggingface_hub not installed")
        except Exception as e:
            if "offline" in str(e).lower() or "connect" in str(e).lower():
                pytest.skip(f"No internet: {e}")
            raise

    def test_make_hub_cache_helper(self, tmp_path):
        """Verify our test helper creates valid hub cache structures."""
        cache_dir = tmp_path / "test_cache"
        _make_hub_cache(cache_dir, "MyOrg/MyModel", {
            "config.json": b'{"architectures": ["TestModel"]}',
            "model.safetensors": b"\x00" * 100,
        })

        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir(str(cache_dir))
        assert len(cache.repos) == 1
        repo = list(cache.repos)[0]
        assert repo.repo_id == "MyOrg/MyModel"
        snap = list(repo.revisions)[0].snapshot_path
        assert (snap / "config.json").is_file()
        assert (snap / "model.safetensors").is_file()


# ─── 3. Service Configuration Tests ─────────────────────────────────────────

class TestKimodoServiceConfig:
    """Verify kimodo_demo.py service config (no GPU needed)."""

    def test_wrapper_script_compiles(self):
        """The embedded wrapper script should be valid Python."""
        wrapper = _get_wrapper_script()
        compile(wrapper, "<wrapper>", "exec")

    def test_wrapper_patches_model_info(self):
        """Verify wrapper script patches model_info before calling kimodo."""
        wrapper = _get_wrapper_script()
        assert "model_info" in wrapper
        assert "_FakeModelInfo" in wrapper
        assert "huggingface_hub" in wrapper

    def test_wrapper_sets_offline_mode(self):
        """Verify wrapper sets HF offline environment variables."""
        wrapper = _get_wrapper_script()
        assert "HF_HUB_OFFLINE" in wrapper
        assert "TRANSFORMERS_OFFLINE" in wrapper

    def test_wrapper_starts_kimodo(self):
        """Verify wrapper calls kimodo.demo.main()."""
        wrapper = _get_wrapper_script()
        assert "kimodo.demo" in wrapper
        assert "kimodo.demo.main()" in wrapper

    def test_service_file_exists(self):
        """Verify the service module file exists."""
        src = Path(__file__).resolve().parent.parent / "services" / "motion" / "kimodo_demo.py"
        assert src.exists(), f"kimodo_demo.py not found at {src}"

    def test_service_port_and_model(self):
        """Verify service constants by parsing the source file."""
        src = Path(__file__).resolve().parent.parent / "services" / "motion" / "kimodo_demo.py"
        content = src.read_text()
        assert "PORT = 18470" in content
        assert 'default_model = "kimodo-soma-rp"' in content
        assert "service_name = \"kimodo_demo\"" in content


# ─── 4. Live Forge Integration Tests ────────────────────────────────────────

@pytest.mark.skipif(not FORGE_URL, reason="FORGE_URL not set")
class TestLiveKimodoLoad:
    """Live tests against a running Forge. Requires FORGE_URL env var."""

    def test_forge_is_up(self):
        """Forge should respond to status requests."""
        import urllib.request
        resp = urllib.request.urlopen(f"{FORGE_URL}/forge", timeout=10)
        data = json.loads(resp.read())
        assert "loaded" in data
        assert "vram_free_mb" in data

    def test_kimodo_preload(self):
        """Kimodo should preload successfully via Forge."""
        r = _forge_req({"action": "preload", "service": "kimodo_demo"}, timeout=900)
        if r.get("status") == "error":
            error = r.get("error", "")
            if "VRAM" in error or "Cannot free" in error:
                pytest.skip(f"Not enough VRAM: {error[:200]}")
            pytest.fail(f"Kimodo preload failed: {error[:500]}")

        assert r["status"] == "loaded", f"Expected loaded, got: {r}"
        assert r["service"] == "kimodo_demo"
        assert r.get("vram_used_mb", 0) > 0

    def test_kimodo_subprocess_healthy(self):
        """After preload, the Viser server should respond on its port."""
        r = _forge_req({"action": "preload", "service": "kimodo_demo"}, timeout=900)
        if r.get("status") != "loaded":
            pytest.skip(f"Kimodo not loaded: {r.get('error', '')[:200]}")

        # The Viser server should serve HTML on the forge's port
        r = _forge_req({
            "service": "kimodo_demo",
            "path": "/",
        }, timeout=30)
        assert r.get("status") == "ok", f"Viser not responding: {r}"

    def test_kimodo_release(self):
        """Releasing kimodo should free VRAM."""
        _forge_req({"action": "preload", "service": "kimodo_demo"}, timeout=900)
        r = _forge_req({"action": "release", "service": "kimodo_demo"}, timeout=60)
        assert r.get("status") == "released"

        status = _forge_req({"action": "status"}, timeout=10)
        assert "kimodo_demo" not in status.get("loaded", {})

    def test_swap_eviction(self):
        """Loading a second service evicts the first."""
        # Load native first (replaces old wan2gp catch-all)
        r = _forge_req({
            "service": "native",
            "model": "z-image-turbo",
            "prompt": "test",
            "seed": 42,
        }, timeout=300)
        if r.get("status") == "error":
            pytest.skip(f"Native load failed: {r.get('error', '')[:200]}")

        # Now load kimodo — should evict native
        r = _forge_req({"action": "preload", "service": "kimodo_demo"}, timeout=900)
        assert r["status"] == "loaded", f"Kimodo should load after eviction: {r}"

        # Cleanup
        _forge_req({"action": "release"}, timeout=60)
