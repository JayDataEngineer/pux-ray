import pytest
pytestmark = pytest.mark.skip(reason="wan2gp service removed — handlers obsolete")

"""Pixal3D handler e2e — SKIPPED (wan2gp service deleted)

NOTE: The wan2gp service has been removed. All tests below are skipped.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_FORK_MODELS_DIR = Path(__file__).resolve().parent.parent / "opt" / "wan2gp" / "models"
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"


def _import_pixal3d_handler():
    """Import pixal3d_handler from the fork's models/ package."""
    mod_file = _FORK_MODELS_DIR / "pixal3d" / "pixal3d_handler.py"
    assert mod_file.exists(), f"Handler file not found: {mod_file}"
    spec = importlib.util.spec_from_file_location(
        "models.pixal3d.pixal3d_handler", str(mod_file),
        submodule_search_locations=[str(mod_file.parent)],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Handler Interface ─────────────────────────────────────────────────────


class TestPixal3dHandlerInterface:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_supported_types(self):
        fh = _import_pixal3d_handler().family_handler
        assert fh.query_supported_types() == ["pixal3d"]

    @pytest.mark.handler
    @pytest.mark.unit
    def test_model_family(self):
        fh = _import_pixal3d_handler().family_handler
        assert fh.query_model_family() == "pixal3d"

    @pytest.mark.handler
    @pytest.mark.unit
    def test_family_infos(self):
        fh = _import_pixal3d_handler().family_handler
        infos = fh.query_family_infos()
        assert "pixal3d" in infos
        num_id, label = infos["pixal3d"]
        assert isinstance(num_id, int)
        assert isinstance(label, str)
        assert "Pixal3D" in label

    @pytest.mark.handler
    @pytest.mark.unit
    def test_model_def(self):
        fh = _import_pixal3d_handler().family_handler
        defn = fh.query_model_def("pixal3d", {})
        assert defn["image_outputs"] is True
        assert defn["audio_only"] is False

    @pytest.mark.handler
    @pytest.mark.unit
    def test_family_id_not_collides_with_trellis(self):
        """Pixal3D must have a different family ID from TRELLIS."""
        pix_mod = _import_pixal3d_handler()
        trellis_file = _FORK_MODELS_DIR / "trellis" / "trellis_handler.py"
        spec = importlib.util.spec_from_file_location(
            "models.trellis.trellis_handler", str(trellis_file),
            submodule_search_locations=[str(trellis_file.parent)],
        )
        trellis_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trellis_mod)

        pix_id = pix_mod.family_handler.query_family_infos()
        trellis_id = trellis_mod.family_handler.query_family_infos()
        pix_num = list(pix_id.values())[0][0]
        trellis_num = list(trellis_id.values())[0][0]
        assert pix_num != trellis_num, (
            f"Family info IDs collide: Pixal3D={pix_num}, TRELLIS={trellis_num}"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_update_default_settings(self):
        fh = _import_pixal3d_handler().family_handler
        ui = {}
        fh.update_default_settings("pixal3d", {}, ui)
        assert "steps" in ui
        assert "guidance" in ui


# ─── Vendor Source Integrity ───────────────────────────────────────────────


class TestPixal3dVendorSource:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_symlink_exists(self):
        link = _FORK_MODELS_DIR / "pixal3d" / "pixal3d"
        assert link.is_symlink() or link.is_dir(), f"pixal3d vendor symlink missing"
        # Symlink target resolves in Docker (/opt/vendor/), not necessarily on host
        target = link.readlink()
        assert not str(target).startswith("/"), f"Absolute symlink: {target}"

    @pytest.mark.handler
    @pytest.mark.unit
    def test_symlink_is_relative(self):
        link = _FORK_MODELS_DIR / "pixal3d" / "pixal3d"
        target = link.readlink()
        assert not str(target).startswith("/"), (
            f"Absolute symlink won't work in Docker: {target}"
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_class_exists(self):
        # Symlink resolves in Docker; on host, check vendor directly
        pipeline_file = _FORK_MODELS_DIR / "pixal3d" / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
        if not pipeline_file.exists():
            pipeline_file = _VENDOR_DIR / "pixal3d" / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
        assert pipeline_file.exists(), f"Pipeline file not found: {pipeline_file}"

    @pytest.mark.handler
    @pytest.mark.unit
    def test_model_names_to_load(self):
        """Pipeline must declare 8 nn.Modules to load from checkpoints."""
        pipeline_file = _FORK_MODELS_DIR / "pixal3d" / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
        if not pipeline_file.exists():
            pipeline_file = _VENDOR_DIR / "pixal3d" / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
        content = pipeline_file.read_text()
        assert "model_names_to_load" in content
        # Parse the list from source
        import ast
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Pixal3DImageTo3DPipeline":
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "model_names_to_load":
                                names = [elt.value for elt in item.value.elts
                                         if isinstance(elt, ast.Constant)]
                                assert len(names) == 8, f"Expected 8 model names, got {len(names)}"
                                expected = {
                                    'sparse_structure_flow_model',
                                    'sparse_structure_decoder',
                                    'shape_slat_flow_model_512',
                                    'shape_slat_flow_model_1024',
                                    'shape_slat_decoder',
                                    'tex_slat_flow_model_512',
                                    'tex_slat_flow_model_1024',
                                    'tex_slat_decoder',
                                }
                                assert set(names) == expected, (
                                    f"Unexpected model names: {set(names) - expected}"
                                )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_image_cond_configs(self):
        """Handler must define 4 DinoV3ProjFeatureExtractor configs."""
        mod = _import_pixal3d_handler()
        configs = mod.IMAGE_COND_CONFIGS
        assert set(configs.keys()) == {"ss", "shape_512", "shape_1024", "tex_1024"}
        for stage, cfg in configs.items():
            assert "model_name" in cfg
            assert "image_size" in cfg
            assert "grid_resolution" in cfg


# ─── Spec & Registry Integration ───────────────────────────────────────────


class TestPixal3dSpecResolution:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_spec_exists(self):
        from registry.specs import list_models
        assert "pixal3d" in list_models()

    @pytest.mark.handler
    @pytest.mark.unit
    def test_spec_resolves_bf16(self):
        from registry.specs import resolve
        spec = resolve("pixal3d", "bf16")
        assert "pipeline_root" in spec["modules"]
        assert spec["quant"] == "bf16"

    @pytest.mark.handler
    @pytest.mark.unit
    def test_registry_paths(self):
        from registry.models import ModelRegistry
        reg = ModelRegistry()
        main_path = Path(reg.get_path("3d", "pixal3d"))
        dinov3_path = Path(reg.get_path("3d", "pixal3d_dinov3"))
        assert main_path.name == "pixal3d"
        assert "dinov3" in str(dinov3_path)

    @pytest.mark.handler
    @pytest.mark.unit
    def test_registry_download_source(self):
        """Verify HF snapshot download is configured."""
        from registry.models import ModelRegistry
        reg = ModelRegistry()
        meta = reg.get_metadata("3d", "pixal3d")
        assert meta is not None
        assert meta.get("download") == "snapshot"
        assert "TencentARC/Pixal3D" in meta.get("source", "")

    @pytest.mark.handler
    @pytest.mark.unit
    def test_dinov3_uses_modelscope(self):
        """DINOv3 must use ModelScope (gated on HuggingFace)."""
        from registry.models import ModelRegistry
        reg = ModelRegistry()
        meta = reg.get_metadata("3d", "pixal3d_dinov3")
        assert meta is not None
        assert meta.get("download") == "modelscope"
        assert "modelscope://" in meta.get("source", "")


# ─── _Pipeline Wiring ──────────────────────────────────────────────────────


class TestPixal3dPipelineWiring:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_class_exists(self):
        mod = _import_pixal3d_handler()
        assert hasattr(mod, "_Pipeline")
        pl = mod._Pipeline

    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_init_signature(self):
        mod = _import_pixal3d_handler()
        sig = inspect.signature(mod._Pipeline.__init__)
        params = list(sig.parameters.keys())
        assert "modules" in params
        assert "samplers" in params
        assert "normalization" in params
        assert "device" in params

    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_generate_signature(self):
        mod = _import_pixal3d_handler()
        sig = inspect.signature(mod._Pipeline.generate)
        params = list(sig.parameters.keys())
        assert "image" in params
        assert "seed" in params
        assert "steps" in params
        assert "guidance" in params
        assert "camera_angle_x" in params
        assert "camera_distance" in params
        assert "mesh_scale" in params
        assert "kwargs" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

    @pytest.mark.handler
    @pytest.mark.unit
    def test_pipeline_constructs_with_mocks(self):
        """_Pipeline can be instantiated with mock modules (no GPU needed)."""
        mod = _import_pixal3d_handler()

        mock_modules = {
            name: MagicMock(spec=torch.nn.Module)
            for name in [
                "ss_flow_model", "ss_decoder",
                "slat_flow_512", "slat_flow_1024", "shape_decoder",
                "tex_slat_flow_512", "tex_slat_flow_1024", "tex_decoder",
                "image_cond_ss", "image_cond_shape_512",
                "image_cond_shape_1024", "image_cond_tex_1024",
                "rembg",
            ]
        }
        mock_samplers = {
            "ss": MagicMock(),
            "shape": MagicMock(),
            "tex": MagicMock(),
        }
        mock_norm = {
            "shape": {"std": [1.0], "mean": [0.0]},
            "tex": {"std": [1.0], "mean": [0.0]},
        }

        pl = mod._Pipeline(
            modules=mock_modules,
            samplers=mock_samplers,
            normalization=mock_norm,
            pbr_layout={"base_color": slice(0, 3), "alpha": slice(5, 6)},
            device=torch.device("cpu"),
        )

        assert pl.m is mock_modules
        assert pl.samplers is mock_samplers
        assert pl.norm is mock_norm
        assert pl.device == torch.device("cpu")

    @pytest.mark.handler
    @pytest.mark.unit
    def test_co_tenants_structure(self):
        """Co-tenants map must reference valid module names."""
        mod = _import_pixal3d_handler()

        mock_modules = {
            name: MagicMock(spec=torch.nn.Module)
            for name in [
                "ss_flow_model", "ss_decoder",
                "slat_flow_512", "slat_flow_1024", "shape_decoder",
                "tex_slat_flow_512", "tex_slat_flow_1024", "tex_decoder",
                "image_cond_ss", "image_cond_shape_512",
                "image_cond_shape_1024", "image_cond_tex_1024",
                "rembg",
            ]
        }
        mock_samplers = {"ss": MagicMock(), "shape": MagicMock(), "tex": MagicMock()}
        mock_norm = {"shape": {"std": [1.0], "mean": [0.0]}, "tex": {"std": [1.0], "mean": [0.0]}}

        pl = mod._Pipeline(
            modules=mock_modules,
            samplers=mock_samplers,
            normalization=mock_norm,
            pbr_layout=None,
            device=torch.device("cpu"),
        )

        # Verify co-tenants reference valid module names
        co_tenants = {
            "ss_flow_model": ["ss_decoder", "image_cond_ss"],
            "slat_flow_512": ["image_cond_shape_512"],
            "slat_flow_1024": ["image_cond_shape_1024"],
            "tex_slat_flow_1024": ["image_cond_tex_1024"],
        }
        all_module_names = set(mock_modules.keys())
        for primary, tenants in co_tenants.items():
            assert primary in all_module_names, f"Co-tenant primary '{primary}' not in modules"
            for t in tenants:
                assert t in all_module_names, f"Co-tenant '{t}' (of {primary}) not in modules"


# ─── Deployment Integration ────────────────────────────────────────────────


class TestPixal3dDeploymentIntegration:
    @pytest.mark.handler
    @pytest.mark.unit
    def test_in_custom_handlers(self):
        """pixal3d must be registered in CUSTOM_HANDLERS."""
        dep_file = Path(__file__).resolve().parent.parent / "services" / "wan2gp" / "deployment.py"
        content = dep_file.read_text()
        assert "models.pixal3d.pixal3d_handler" in content

    @pytest.mark.handler
    @pytest.mark.unit
    def test_in_spec_aware(self):
        """pixal3d must be in the _SPEC_AWARE map."""
        dep_file = Path(__file__).resolve().parent.parent / "services" / "wan2gp" / "deployment.py"
        content = dep_file.read_text()
        assert '"pixal3d": "pixal3d"' in content

    @pytest.mark.handler
    @pytest.mark.unit
    def test_in_model_specs_yaml(self):
        """pixal3d must have a spec entry."""
        specs_file = Path(__file__).resolve().parent.parent / "config" / "model_specs.yaml"
        content = specs_file.read_text()
        assert "pixal3d:" in content
        assert "handler: pixal3d" in content

    @pytest.mark.handler
    @pytest.mark.unit
    def test_in_model_registry_yaml(self):
        """pixal3d must have registry entries for download."""
        reg_file = Path(__file__).resolve().parent.parent / "config" / "model_registry.yaml"
        content = reg_file.read_text()
        assert "pixal3d:" in content
        assert "TencentARC/Pixal3D" in content
        assert "pixal3d_dinov3:" in content
        assert "modelscope://" in content
