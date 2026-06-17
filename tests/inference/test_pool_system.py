"""Tests for the 4-tier inference pool system.

Verifies:
  - YAML config loads and validates cleanly (no warnings).
  - Each declared model resolves to at least one pool.
  - Tier A models resolve to their dedicated specialized pool.
  - Tier B DiT models (qwen-image-edit) resolve to omni-vllm.
  - Multi-pool models (z-image) resolve with correct fallback order.
  - Per-model optimization configs are preserved (FP8, Cache-DiT, etc.).
  - The dispatch bridge produces a sensible ordered plan from a workflow step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure we can import from the repo root when run via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.inference import PoolManager
from services.inference.dispatch import resolve_step


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mgr() -> PoolManager:
    return PoolManager.from_yaml()


# ─── Config integrity ────────────────────────────────────────────────────────

def test_config_validates_clean(mgr: PoolManager):
    warns = mgr.validate()
    assert warns == [], f"Config has warnings:\n  - " + "\n  - ".join(warns)


def test_four_pools_present(mgr: PoolManager):
    names = {p.name for p in mgr.pools()}
    # Tier A specialized
    for required in ("moss", "diarization", "llama", "llama-bee", "comfyui"):
        assert required in names, f"Missing Tier A pool: {required}"
    # Tier B-D
    for required in ("omni-vllm", "sglang", "diffusers"):
        assert required in names, f"Missing pool: {required}"


def test_each_tier_has_pools(mgr: PoolManager):
    for tier in ("A", "B", "C", "D"):
        pools = mgr.system.pools_by_tier(tier)
        assert pools, f"No pools in tier {tier}"


def test_priorities_increase_with_tier(mgr: PoolManager):
    """Tier A pools have lower priority numbers than B, B than C, etc."""
    by_tier = {t: min(p.priority for p in mgr.system.pools_by_tier(t))
               for t in ("A", "B", "C", "D")}
    assert by_tier["A"] < by_tier["B"] < by_tier["C"] < by_tier["D"]


def test_no_port_collisions(mgr: PoolManager):
    ports = [p.port for p in mgr.pools()]
    assert len(ports) == len(set(ports)), f"Port collision: {ports}"


# ─── Model resolution ────────────────────────────────────────────────────────

def test_every_route_resolves(mgr: PoolManager):
    for model in mgr.models():
        targets = mgr.resolve(model)
        assert targets, f"Route for {model!r} resolves to nothing"


def test_every_route_pool_healthy_shape(mgr: PoolManager):
    for model in mgr.models():
        primary = mgr.resolve_primary(model)
        assert primary is not None
        assert primary.pool.port > 0
        assert primary.pool.framework


@pytest.mark.parametrize("model,primary_pool,primary_tier", [
    ("moss-tts", "moss", "A"),
    ("moss-soundeffect", "moss", "A"),
    ("diarization", "diarization", "A"),
    ("llama", "llama", "A"),
    ("llama-bee", "llama-bee", "A"),
    ("comfyui", "comfyui", "A"),
    ("qwen-image-edit", "omni-vllm", "B"),
    ("wan-vace", "omni-vllm", "B"),
    ("wan-t2v", "omni-vllm", "B"),
    ("wan-i2v", "omni-vllm", "B"),
    ("cosmos", "omni-vllm", "B"),
    ("anima-base", "omni-vllm", "B"),       # future slot
    ("ideogram4", "sglang", "C"),
    ("ltx-video", "sglang", "C"),
    ("kimodo", "diffusers", "D"),
    ("ace-step", "ace-step", "A"),
    ("kokoro", "diffusers", "D"),
])
def test_primary_pool(mgr: PoolManager, model, primary_pool, primary_tier):
    target = mgr.resolve_primary(model)
    assert target is not None, f"No route for {model}"
    assert target.pool.name == primary_pool
    assert target.pool.tier == primary_tier


def test_z_image_has_sglang_fallback(mgr: PoolManager):
    """z-image is the canonical multi-tier model: omni-vllm primary, sglang fallback."""
    targets = mgr.resolve("z-image")
    assert len(targets) == 2
    assert targets[0].pool.name == "omni-vllm"
    assert targets[0].is_primary
    assert targets[1].pool.name == "sglang"
    assert targets[1].fallback_index == 1


def test_qwen_image_edit_no_fallback(mgr: PoolManager):
    """qwen-image-edit's FP8 patch only exists in omni-vllm — no fallback."""
    targets = mgr.resolve("qwen-image-edit")
    assert len(targets) == 1
    assert targets[0].pool.name == "omni-vllm"


# ─── Optimization preserved ──────────────────────────────────────────────────

def test_qwen_image_edit_optimization(mgr: PoolManager):
    target = mgr.resolve_primary("qwen-image-edit")
    assert target.launcher is not None
    opt = target.launcher.optimization
    assert opt is not None
    assert opt.quant == "fp8-weight-only"
    assert opt.cache_dit is True
    assert opt.taylorseer is True
    assert opt.vae_tiling is True
    assert opt.vae_slicing is True


def test_qwen_image_edit_patch_path(mgr: PoolManager):
    target = mgr.resolve_primary("qwen-image-edit")
    assert target.launcher is not None
    assert target.launcher.patch == "scripts/pipeline_qwen_image_edit_plus_patch.py"
    assert target.launcher.script == "scripts/run_omni_qwen_img_edit_fp8.sh"


def test_qwen_image_edit_benchmark(mgr: PoolManager):
    target = mgr.resolve_primary("qwen-image-edit")
    assert target.launcher is not None
    bench = target.launcher.benchmark
    assert len(bench) == 2
    steps = {b["steps"] for b in bench}
    assert 4 in steps
    assert 20 in steps


def test_wan_vace_teacache(mgr: PoolManager):
    target = mgr.resolve_primary("wan-vace")
    assert target.launcher is not None
    opt = target.launcher.optimization
    assert opt is not None
    assert opt.teacache_thresh == pytest.approx(0.01)


def test_z_image_sglang_has_benchmark(mgr: PoolManager):
    """The sglang fallback for z-image has the famous 1.61s benchmark."""
    targets = mgr.resolve("z-image")
    sglang = next(t for t in targets if t.pool.name == "sglang")
    assert sglang.launcher is not None
    assert sglang.launcher.benchmark
    bench = sglang.launcher.benchmark[0]
    assert bench["time_s"] == pytest.approx(1.61)


# ─── Dispatch bridge ─────────────────────────────────────────────────────────

def test_dispatch_resolves_workflow_step():
    plan = resolve_step(service="forge", model="qwen-image-edit")
    assert len(plan) >= 1
    hop = plan[0]
    assert "v1/images" in hop.url or "generate" in hop.url
    assert hop.pool.name == "omni-vllm"
    assert hop.action in {"generate", "edit"}


def test_dispatch_z_image_has_two_hops():
    plan = resolve_step(service=None, model="z-image")
    assert len(plan) == 2
    assert plan[0].pool.name == "omni-vllm"
    assert plan[1].pool.name == "sglang"


def test_dispatch_payload_envelope():
    plan = resolve_step(service=None, model="z-image")
    body = plan[0].payload({"prompt": "test"})
    assert body["model"] == "z-image"
    assert body["tier"] in {"A", "B", "C", "D"}
    assert body["prompt"] == "test"
    assert body["service"] == "omni-vllm"


def test_dispatch_unknown_model_raises():
    with pytest.raises(ValueError, match="No pool serves"):
        resolve_step(service=None, model="definitely-not-a-real-model")


# ─── Tier A specialized services ─────────────────────────────────────────────

def test_tier_a_models_have_single_pool(mgr: PoolManager):
    """Tier A specialized dockers have no fallback (one pool each)."""
    for model in ("moss-tts", "moss-soundeffect", "diarization",
                  "llama", "llama-bee", "comfyui"):
        targets = mgr.resolve(model)
        assert len(targets) == 1, f"{model} should have exactly one target"
        assert targets[0].pool.tier == "A"


# ─── Action selection (honoring preferred action) ───────────────────────────

def test_dispatch_honors_edit_action():
    """resolve_step with action='edit' picks /v1/images/edits for qwen-image-edit."""
    plan = resolve_step(service=None, model="qwen-image-edit", action="edit")
    assert plan[0].url.endswith("/v1/images/edits")
    assert plan[0].action == "edit"


def test_dispatch_honors_generate_action():
    """resolve_step with action='generate' picks /v1/images/generations."""
    plan = resolve_step(service=None, model="qwen-image-edit", action="generate")
    assert plan[0].url.endswith("/v1/images/generations")
    assert plan[0].action == "generate"


def test_dispatch_falls_back_to_first_action_when_preferred_missing():
    """If the preferred action isn't declared, the first declared action is used."""
    plan = resolve_step(service=None, model="qwen-image-edit",
                        action="nonexistent-action")
    # Should fall back to "generate" (first key in YAML api: map)
    assert plan[0].action in {"generate", "edit"}
    assert plan[0].url.startswith("http://")


def test_dispatch_first_healthy_property():
    """DispatchPlan.first_healthy returns the first healthy hop (or None)."""
    from services.inference.dispatch import DispatchPlan, DispatchHop
    # Empty plan
    empty = DispatchPlan()
    assert empty.first_healthy is None


# ─── Workflow engine integration ─────────────────────────────────────────────

def test_pool_step_executor_imports():
    """The new pool step executor must be importable and registered."""
    from services.workflows.steps.pool import PoolStepExecutor
    from services.workflows.steps import StepExecutor
    assert issubclass(PoolStepExecutor, StepExecutor)


def test_img_edit_step_uses_pool_resolution():
    """ImageEditStep resolves URLs via the pool system, not the old hard-coded map."""
    from services.workflows.steps.img_edit import _resolve_api_url
    url = _resolve_api_url("qwen-image-edit", action="edit")
    # Pool resolution returns the omni-vllm pool's port (8093) with /v1/images/edits
    assert url.endswith("/v1/images/edits"), f"got {url}"
    assert ":8093" in url or "omni-qwen-img-edit" in url, f"got {url}"


def test_img_edit_step_legacy_fallback_on_unknown_model():
    """When the pool config doesn't have a model, legacy fallback kicks in."""
    from services.workflows.steps.img_edit import _resolve_api_url
    url = _resolve_api_url("qwen-image-edit-2511-fp8", action="edit")
    # Not in the YAML — should fall back to the legacy omni-qwen-img-edit-fp8 host
    assert "omni-qwen-img-edit-fp8" in url, f"got {url}"
    assert url.endswith("/v1/images/edits"), f"got {url}"


def test_vace_step_uses_pool_resolution_for_base_url():
    """VaceGenerateStep resolves the base URL via the pool system."""
    from services.workflows.steps.vace import _resolve_api_base
    base, path = _resolve_api_base("wan2.1-vace-14b-fp8-diffusers")
    # Pool resolution returns the omni-vllm pool's URL (port 8093)
    assert ":8093" in base, f"got base={base}"
    # The in-container path stays from the legacy map (declared by the script)
    assert path == "/models/vace-fp8", f"got path={path}"


def test_engine_registers_pool_step_type():
    """WorkflowEngine should register the 'pool' step type.

    Ray's @serve.deployment decorator wraps WorkflowEngine in a Deployment
    proxy, so we read the engine module source directly instead of inspecting
    the class.
    """
    import inspect
    from services.workflows import engine as engine_mod
    src = inspect.getsource(engine_mod)
    assert 'register("pool"' in src or "register('pool'" in src, \
        "pool step type not registered in engine source"


# ─── Gateway routes ──────────────────────────────────────────────────────────

class _FakeRequest:
    """Minimal Starlette Request stub for testing route handlers."""
    def __init__(self, path_params=None, body_json=None):
        self.path_params = path_params or {}
        self._json = body_json
    async def json(self):
        return self._json or {}


@pytest.mark.asyncio
async def test_route_list_pools():
    """GET /v1/inference/pools returns all 8 pools."""
    from gateway.routes import inference as inf
    resp = await inf.list_pools(_FakeRequest())
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "omni-vllm" in body
    assert "sglang" in body
    assert "diffusers" in body


@pytest.mark.asyncio
async def test_route_list_models():
    """GET /v1/inference/models returns every routable model."""
    from gateway.routes import inference as inf
    resp = await inf.list_models(_FakeRequest())
    assert resp.status_code == 200
    body = resp.body.decode()
    for model in ("qwen-image-edit", "z-image", "wan-vace", "moss-tts"):
        assert model in body, f"{model} missing from list_models response"


@pytest.mark.asyncio
async def test_route_resolve_known_model():
    """GET /v1/inference/models/{model}/resolve returns the chain."""
    from gateway.routes import inference as inf
    resp = await inf.resolve_model(_FakeRequest({"model": "qwen-image-edit"}))
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "resolution_chain" in body
    assert "omni-vllm" in body


@pytest.mark.asyncio
async def test_route_resolve_unknown_model_404():
    """GET /v1/inference/models/<unknown>/resolve returns 404."""
    from gateway.routes import inference as inf
    resp = await inf.resolve_model(_FakeRequest({"model": "definitely-not-real"}))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_route_optimization_known_model():
    """GET /v1/inference/models/qwen-image-edit/optimization returns FP8 config."""
    from gateway.routes import inference as inf
    resp = await inf.get_optimization(_FakeRequest({"model": "qwen-image-edit"}))
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "fp8-weight-only" in body


@pytest.mark.asyncio
async def test_route_optimization_unknown_model_404():
    """GET /v1/inference/models/<unknown>/optimization returns 404."""
    from gateway.routes import inference as inf
    resp = await inf.get_optimization(_FakeRequest({"model": "no-such"}))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_route_get_pool_known():
    """GET /v1/inference/pools/{name} returns the pool's status."""
    from gateway.routes import inference as inf
    resp = await inf.get_pool(_FakeRequest({"pool_name": "omni-vllm"}))
    # 200 if the pool exists; status reflects whether the container is running
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "omni-vllm" in body
    assert "vllm-omni" in body


@pytest.mark.asyncio
async def test_route_get_pool_unknown_404():
    """GET /v1/inference/pools/<unknown> returns 404."""
    from gateway.routes import inference as inf
    resp = await inf.get_pool(_FakeRequest({"pool_name": "no-such-pool"}))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_route_z_image_resolution_has_two_hops():
    """z-image resolution chain has both omni-vllm and sglang."""
    from gateway.routes import inference as inf
    resp = await inf.resolve_model(_FakeRequest({"model": "z-image"}))
    body = resp.body.decode()
    assert "omni-vllm" in body
    assert "sglang" in body
