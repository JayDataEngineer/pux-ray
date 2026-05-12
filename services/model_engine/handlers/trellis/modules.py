"""TRELLIS.2 raw nn.Module loading + mmgp setup.

Loads each model individually from pipeline.json:
- ss_flow_model: SparseStructureFlowModel (voxel structure generation)
- ss_decoder: SparseStructureDecoder (voxel → coordinates)
- shape_slat_flow_512 / _1024: SLatFlowModel (shape latent generation)
- shape_slat_decoder: SparseUnetVaeDecoder (shape latent → mesh)
- tex_slat_flow_1024: SLatFlowModel (texture latent generation)
- tex_slat_decoder: FlexiDualGridVaeDecoder (texture latent → PBR voxels)
- dinov3: DinoV3FeatureExtractor (image conditioning)
- rembg: BiRefNet (background removal)

Samplers (FlowEulerGuidanceIntervalSampler) are NOT nn.Modules — instantiated in orchestrator.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrellisModules:
    """All raw nn.Modules for TRELLIS.2 inference."""

    # Flow models
    ss_flow_model: Any
    shape_slat_flow_512: Any
    shape_slat_flow_1024: Any
    tex_slat_flow_1024: Any

    # Decoders
    ss_decoder: Any
    shape_slat_decoder: Any
    tex_slat_decoder: Any

    # Conditioning
    dinov3: Any
    rembg: Any

    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")

    # Sampler configs (not nn.Modules)
    ss_sampler_config: dict = field(default_factory=dict)
    shape_sampler_config: dict = field(default_factory=dict)
    tex_sampler_config: dict = field(default_factory=dict)

    # Normalization stats
    shape_slat_normalization: Optional[dict] = None
    tex_slat_normalization: Optional[dict] = None

    # mmgp
    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        precision: str = "bf16",
        device: str = "cuda",
    ) -> TrellisModules:
        import sys
        from registry.config import Config
        cfg = Config()
        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        import os
        os.environ["TRELLIS_PIPELINE_ROOT"] = str(model_path)

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(precision, torch.bfloat16)
        dev = torch.device(device)

        # Find pipeline.json — may be in a subdirectory (e.g. TRELLIS.2-4B/ckpts/)
        pipeline_json = model_path / "pipeline.json"
        if not pipeline_json.exists():
            candidates = sorted(model_path.rglob("pipeline.json"))
            if not candidates:
                raise FileNotFoundError(f"pipeline.json not found under {model_path}")
            pipeline_json = candidates[0]

        # Base dir for resolving relative weight paths in pipeline.json
        pipeline_root = pipeline_json.parent

        with open(pipeline_json) as f:
            pipeline_cfg = json.load(f)

        args = pipeline_cfg.get("args", {})
        models_cfg = args.get("models", {})
        sampler_configs = {
            "ss": args.get("sparse_structure_sampler", {}),
            "shape": args.get("shape_slat_sampler", {}),
            "tex": args.get("tex_slat_sampler", {}),
        }

        # Import model classes
        from trellis2.models import SparseStructureFlowModel, SparseStructureDecoder
        from trellis2.models import SLatFlowModel, ElasticSLatFlowModel
        from trellis2.models import SparseUnetVaeDecoder, FlexiDualGridVaeDecoder
        from trellis2.modules import DinoV3FeatureExtractor
        from trellis2.pipelines.rembg import BiRefNet

        def _load_model(model_cls, model_path_or_name, **kw):
            if isinstance(model_path_or_name, str) and Path(model_path_or_name).is_file():
                from safetensors.torch import load_file
                state_dict = load_file(model_path_or_name)
                model = model_cls(**kw) if kw else model_cls()
                model.load_state_dict(state_dict, strict=False)
                return model.to(dtype).to(dev)
            # Some models use from_pretrained or class constructors
            return model_cls(model_path_or_name).to(dtype).to(dev)

        logger.info("Loading TRELLIS models from %s (pipeline.json in %s)", model_path, pipeline_root)

        # Load flow models — resolve weight paths relative to pipeline_root
        ss_flow = _load_from_config(models_cfg, "sparse_structure_flow_model", pipeline_root)
        shape_flow_512 = _load_from_config(models_cfg, "shape_slat_flow_model_512", pipeline_root)
        shape_flow_1024 = _load_from_config(models_cfg, "shape_slat_flow_model_1024", pipeline_root)
        tex_flow_1024 = _load_from_config(models_cfg, "tex_slat_flow_model_1024", pipeline_root)

        # Load decoders
        ss_dec = _load_from_config(models_cfg, "sparse_structure_decoder", pipeline_root)
        shape_dec = _load_from_config(models_cfg, "shape_slat_decoder", pipeline_root)
        tex_dec = _load_from_config(models_cfg, "tex_slat_decoder", pipeline_root)

        # Load conditioning models
        dinov3_cfg = args.get("image_cond_model", {})
        dinov3 = DinoV3FeatureExtractor(
            model_name=dinov3_cfg.get("args", {}).get("model_name", "dinov3-vit-g"),
            image_size=dinov3_cfg.get("args", {}).get("image_size", 1024),
        ).to(dev)

        rembg_cfg = args.get("rembg_model", {})
        rembg = BiRefNet(
            model_name=rembg_cfg.get("args", {}).get("model_name", "RMBG-2.0"),
        ).to(dev)

        # Apply precision to all modules
        for m in [ss_flow, shape_flow_512, shape_flow_1024, tex_flow_1024,
                  ss_dec, shape_dec, tex_dec]:
            if hasattr(m, 'to'):
                m.to(dtype).to(dev)

        # Build pipe dict for mmgp
        pipe = {
            "ss_flow_model": ss_flow,
            "shape_slat_flow_512": shape_flow_512,
            "shape_slat_flow_1024": shape_flow_1024,
            "tex_slat_flow_1024": tex_flow_1024,
            "ss_decoder": ss_dec,
            "shape_slat_decoder": shape_dec,
            "tex_slat_decoder": tex_dec,
            "dinov3": dinov3,
            "rembg": rembg,
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("TRELLIS loaded: %d modules, VRAM=%.0fMB", len(pipe), vram)

        return cls(
            ss_flow_model=ss_flow,
            shape_slat_flow_512=shape_flow_512,
            shape_slat_flow_1024=shape_flow_1024,
            tex_slat_flow_1024=tex_flow_1024,
            ss_decoder=ss_dec,
            shape_slat_decoder=shape_dec,
            tex_slat_decoder=tex_dec,
            dinov3=dinov3,
            rembg=rembg,
            dtype=dtype,
            device=dev,
            ss_sampler_config=sampler_configs["ss"],
            shape_sampler_config=sampler_configs["shape"],
            tex_sampler_config=sampler_configs["tex"],
            shape_slat_normalization=args.get("shape_slat_normalization"),
            tex_slat_normalization=args.get("tex_slat_normalization"),
            pipe=pipe,
            co_tenants={"ss_flow_model": ["ss_decoder"],
                        "shape_slat_flow_1024": ["shape_slat_decoder"],
                        "tex_slat_flow_1024": ["tex_slat_decoder"]},
        )


def _load_from_config(models_cfg: dict, key: str, model_path: Path) -> torch.nn.Module:
    """Load a model from pipeline.json config entry."""
    from trellis2.models import SparseStructureFlowModel, SparseStructureDecoder
    from trellis2.models import SLatFlowModel, ElasticSLatFlowModel
    from trellis2.models import SparseUnetVaeDecoder, FlexiDualGridVaeDecoder
    from safetensors.torch import load_file
    import json

    weight_path = models_cfg.get(key)
    if weight_path is None:
        raise FileNotFoundError(f"No weight path for {key} in pipeline.json")

    # Resolve relative paths
    wp = Path(weight_path)
    if not wp.is_absolute():
        wp = model_path / wp

    # Pipeline.json omits extensions — try .safetensors
    if not wp.exists():
        for ext in (".safetensors", ".bin", ".pt"):
            if wp.with_suffix(ext).exists():
                wp = wp.with_suffix(ext)
                break

    if not wp.exists():
        raise FileNotFoundError(f"Weight file not found: {wp}")

    # Find config.json alongside weights
    config_path = wp.parent / "config.json"
    if not config_path.exists():
        config_path = wp.parent / (key + "_config.json")

    # Load state dict
    state_dict = load_file(str(wp))

    # Determine model class from config or key name
    model_cls = _resolve_model_class(key, config_path)

    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        # Try creating from config
        config_cls = getattr(model_cls, "config_class", None)
        if config_cls:
            try:
                config = config_cls(**config_dict)
                model = model_cls(config)
            except Exception:
                model = model_cls(**config_dict)
        else:
            model = model_cls(**config_dict)
    else:
        model = model_cls()

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _resolve_model_class(key: str, config_path: Path) -> type:
    """Map model key name to class."""
    from trellis2.models import (
        SparseStructureFlowModel, SparseStructureDecoder,
        SLatFlowModel, ElasticSLatFlowModel,
        SparseUnetVaeDecoder, FlexiDualGridVaeDecoder,
    )

    if "ss_flow" in key or "sparse_structure_flow" in key:
        return SparseStructureFlowModel
    if "ss_decoder" in key or "sparse_structure_decoder" in key:
        return SparseStructureDecoder
    if "tex_slat_flow" in key:
        return SLatFlowModel
    if "shape_slat_flow" in key:
        if "elastic" in key:
            return ElasticSLatFlowModel
        return SLatFlowModel
    if "tex_slat_decoder" in key:
        return FlexiDualGridVaeDecoder
    if "shape_slat_decoder" in key:
        return SparseUnetVaeDecoder
    # Fallback: try to read class from config
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        cls_name = cfg.get("_class_name", "")
        import trellis2.models as _m
        if hasattr(_m, cls_name):
            return getattr(_m, cls_name)
    return SLatFlowModel
