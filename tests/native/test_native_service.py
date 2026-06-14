"""E2E tests for native diffusers service.

Tests model loading, generation, LoRA support, format selection,
and VRAM optimization. Run on a GPU node with diffusers installed.

Usage:
  pytest tests/native/test_native_service.py -v --gpu
  python tests/native/test_native_service.py  # standalone
"""
import os
import sys
import time
import io
import base64

# Prevent mmap OOM
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")

import pytest
import torch

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def service():
    """Create a NativeDiffusersService instance."""
    from services.native.service import NativeDiffusersService
    svc = NativeDiffusersService()
    yield svc
    svc.unload()


def has_gpu():
    return torch.cuda.is_available()


skip_no_gpu = pytest.mark.skipif(not has_gpu(), reason="No GPU available")


# ─── Model Registry Tests ──────────────────────────────────────────────────────

class TestModelRegistry:
    """Test that the model registry is correctly configured."""

    def test_z_image_registered(self):
        from services.native.models import get_model_config
        cfg = get_model_config("z-image")
        assert cfg is not None
        assert cfg.pipeline_class == "ZImagePipeline"

    def test_z_image_turbo_registered(self):
        from services.native.models import get_model_config
        cfg = get_model_config("z-image-turbo")
        assert cfg is not None
        assert cfg.default_steps == 8

    def test_anima_registered(self):
        from services.native.models import get_model_config
        cfg = get_model_config("anima")
        assert cfg is not None
        assert cfg.pipeline_class == "ModularPipeline"

    def test_flux_schnell_registered(self):
        from services.native.models import get_model_config
        cfg = get_model_config("flux-schnell")
        assert cfg is not None
        assert cfg.default_steps == 4

    def test_ltx_video_registered(self):
        from services.native.models import get_model_config
        cfg = get_model_config("ltx-video")
        assert cfg is not None
        assert cfg.task == "text2video"

    def test_all_models_have_components(self):
        from services.native.models import MODELS
        for name, cfg in MODELS.items():
            assert "transformer" in cfg.components, f"{name} missing transformer component"
            assert "vae" in cfg.components, f"{name} missing vae component"

    def test_all_models_have_licenses(self):
        from services.native.models import MODELS
        for name, cfg in MODELS.items():
            assert cfg.license, f"{name} missing license info"


# ─── VRAM Planning Tests ───────────────────────────────────────────────────────

class TestVRAMPlanning:
    """Test the adaptive VRAM planning logic."""

    def test_bf16_resident_when_fits(self):
        from services.native.vram import plan_vram, OffloadStrategy, Format
        # 8GB model, 24GB available → should fit resident
        plan = plan_vram(available_mb=24000, model_bf16_size_mb=8000, text_encoder_bf16_size_mb=9000)
        assert plan.strategy == OffloadStrategy.RESIDENT
        assert plan.transformer_format == Format.BF16
        assert plan.use_compile is True

    def test_group_offload_when_tight(self):
        from services.native.vram import plan_vram, OffloadStrategy, Format
        # 23GB model, 16GB available → needs streaming
        plan = plan_vram(available_mb=16000, model_bf16_size_mb=23000, text_encoder_bf16_size_mb=9000)
        assert plan.strategy == OffloadStrategy.GROUP_OFFLOAD
        assert plan.use_compile is False  # incompatible with group_offload

    def test_fp8_when_bf16_doesnt_fit(self):
        from services.native.vram import plan_vram, OffloadStrategy, Format
        # 23GB model, 18GB available → FP8 resident should fit
        plan = plan_vram(available_mb=18000, model_bf16_size_mb=23000, text_encoder_bf16_size_mb=9000)
        assert plan.transformer_format in (Format.FP8, Format.BF16)

    def test_vae_always_bf16(self):
        from services.native.vram import plan_vram, Format
        # Even in lowest tier, VAE stays BF16
        plan = plan_vram(available_mb=4000, model_bf16_size_mb=23000, text_encoder_bf16_size_mb=9000)
        assert plan.vae_format == Format.BF16


# ─── Service Integration Tests (require GPU + models) ──────────────────────────

@skip_no_gpu
class TestNativeService:
    """Integration tests that load and generate with actual models."""

    def test_z_image_turbo_generation(self, service):
        """Load Z-Image Turbo and generate an image."""
        pytest.skip("Requires Z-Image Turbo downloaded to /models/z-image-turbo")
        service.load("z-image-turbo")
        assert service.is_loaded()

        result = service.infer({
            "prompt": "a golden retriever puppy in a field",
            "steps": 8,
            "seed": 42,
        })
        assert result["status"] == "success"
        assert result["output"]["type"] == "image"
        assert len(result["output"]["content"]) > 1000  # base64 PNG

    def test_flux_schnell_generation(self, service):
        """Load FLUX-schnell and generate an image."""
        pytest.skip("Requires FLUX-schnell at /models/flux-schnell")
        service.load("flux-schnell")
        assert service.is_loaded()

        result = service.infer({
            "prompt": "a cinematic photo of a puppy",
            "steps": 4,
            "seed": 42,
        })
        assert result["status"] == "success"
        assert result["metrics"]["latency_ms"] > 0

    def test_lora_loading(self, service):
        """Test LoRA loading and scaling."""
        pytest.skip("Requires model + LoRA files")
        service.load("z-image-turbo")

        # Load a LoRA
        service.lora_manager.load("/models/loras/style.safetensors", adapter_name="style")
        assert "style" in service.lora_manager.list_adapters()

        # Set with scale
        service.lora_manager.set_active(["style"], [0.85])

        # Generate with LoRA
        result = service.infer({"prompt": "test", "steps": 8})
        assert result["status"] == "success"

    def test_unload_frees_vram(self, service):
        """Test that unloading releases VRAM."""
        pytest.skip("Requires GPU + model")
        service.load("z-image-turbo")
        vram_loaded = torch.cuda.memory_allocated(0)

        service.unload()
        vram_unloaded = torch.cuda.memory_allocated(0)

        assert vram_unloaded < vram_loaded


# ─── Standable Runner ──────────────────────────────────────────────────────────

def run_standalone():
    """Run tests without pytest — for quick verification on the pod."""
    import traceback

    tests = [
        ("Model Registry", test_registry),
        ("VRAM Planning", test_vram_planning),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            func()
            print(f"  ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


def test_registry():
    """Quick registry check."""
    from services.native.models import MODELS, get_model_config
    assert len(MODELS) >= 8, f"Expected >=8 models, got {len(MODELS)}"
    for name in MODELS:
        cfg = get_model_config(name)
        assert cfg is not None
        assert cfg.pipeline_class
        assert cfg.default_steps > 0
    print(f"  {len(MODELS)} models registered: {list(MODELS.keys())}")


def test_vram_planning():
    """Quick VRAM planning check."""
    from services.native.vram import plan_vram, OffloadStrategy
    # Small model on big GPU → resident
    plan = plan_vram(24000, 8000, 9000)
    assert plan.strategy == OffloadStrategy.RESIDENT
    print(f"  Small model: {plan.strategy.value} ({plan.notes})")

    # Large model on small GPU → group_offload
    plan = plan_vram(12000, 23000, 9000)
    assert plan.strategy in (OffloadStrategy.GROUP_OFFLOAD, OffloadStrategy.MODEL_CPU_OFFLOAD)
    print(f"  Large model: {plan.strategy.value} ({plan.notes})")


if __name__ == "__main__":
    run_standalone()
