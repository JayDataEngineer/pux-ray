"""AniGen raw nn.Module loading + mmgp setup.

Decomposes AnigenImageTo3DPipeline into 6 nn.Modules:
- dinov2: DINOv2 vision encoder (image conditioning)
- dsine: DSINE normal estimation (surface normals from RGB)
- ss_flow_model: AniGenSparseStructureFlowModel (sparse voxel + skeleton diffusion)
- ss_decoder: AniGenSparseStructureDecoder (voxel → dense coordinates)
- slat_flow_model: AniGenElasticSLatFlowModel (structured latent diffusion)
- slat_decoder: AniGenElasticSLatMeshDecoder (latent → rigged mesh + skin weights)

All models forced to FP32. Flash attention + pytorch3d patches applied during load.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class AniGenModules:
    """All raw nn.Modules for AniGen inference."""

    dinov2: Any
    dsine: Any
    ss_flow_model: Any
    ss_decoder: Any
    slat_flow_model: Any
    slat_decoder: Any

    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda")

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
    ) -> AniGenModules:
        from registry.config import Config
        cfg = Config()

        # Apply patches before any imports
        _apply_patches()

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        if not model_path.is_dir():
            raise FileNotFoundError(f"AniGen model not found at {model_path}")

        # torch.hub.load in the pipeline uses ./ckpts/ relative to CWD
        old_cwd = os.getcwd()
        os.chdir(str(model_path))

        logger.info("Loading AniGen modules from %s", model_path)

        from anigen.pipelines import AnigenImageTo3DPipeline

        ckpts_dir = str(model_path / "ckpts" / "anigen")
        ss_flow_path = str(Path(ckpts_dir) / "ss_flow_duet")
        slat_flow_path = str(Path(ckpts_dir) / "slat_flow_auto")

        # Load the pipeline, then extract individual modules
        pipeline = AnigenImageTo3DPipeline.from_pretrained(
            ss_flow_path=ss_flow_path,
            slat_flow_path=slat_flow_path,
        )

        os.chdir(old_cwd)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        # Extract the 6 models from pipeline.models dict
        models = getattr(pipeline, "models", {})

        dinov2 = models.get("image_cond_model")
        dsine = models.get("dsine")
        ss_flow = models.get("ss_flow_model")
        ss_dec = models.get("ss_decoder")
        slat_flow = models.get("slat_flow_model")
        slat_dec = models.get("slat_decoder")

        # Force all to FP32
        for m in [dinov2, dsine, ss_flow, ss_dec, slat_flow, slat_dec]:
            if isinstance(m, torch.nn.Module):
                _force_fp32(m)

        # Apply post-load patches
        _patch_flash_attn()
        _patch_pytorch3d()

        torch.cuda.empty_cache()
        gc.collect()

        pipe = {}
        for name, mod in [
            ("dinov2", dinov2),
            ("dsine", dsine),
            ("ss_flow_model", ss_flow),
            ("ss_decoder", ss_dec),
            ("slat_flow_model", slat_flow),
            ("slat_decoder", slat_dec),
        ]:
            if isinstance(mod, torch.nn.Module):
                pipe[name] = mod

        vram = torch.cuda.memory_allocated(0) / (1024**2) if device == "cuda" else 0
        logger.info("AniGen loaded: %d modules, VRAM=%.0fMB", len(pipe), vram)

        return cls(
            dinov2=dinov2,
            dsine=dsine,
            ss_flow_model=ss_flow,
            ss_decoder=ss_dec,
            slat_flow_model=slat_flow,
            slat_decoder=slat_dec,
            device=torch.device(device),
            pipe=pipe,
            co_tenants={"ss_flow_model": ["ss_decoder"], "slat_flow_model": ["slat_decoder"]},
        )


def _apply_patches():
    """Pre-load compatibility patches."""
    from services.compat import apply
    apply()

    os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("U2NET_DEVICE", "cpu")

    import importlib.util as _iu
    _orig = _iu.find_spec
    def _safe(name, *a, **kw):
        try:
            return _orig(name, *a, **kw)
        except (ModuleNotFoundError, ValueError):
            return None
    _iu.find_spec = _safe

    import functools
    try:
        import rembg
        _orig_new = rembg.new_session
        @functools.wraps(_orig_new)
        def _cpu(*a, **kw):
            kw.setdefault("providers", ["CPUExecutionProvider"])
            return _orig_new(*a, **kw)
        rembg.new_session = _cpu
    except ImportError:
        pass


def _force_fp32(model: torch.nn.Module):
    """Convert all params to FP32 and disable fp16 flags."""
    for p in model.parameters():
        if p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.to(torch.float32)
    for b in model.buffers():
        if b.dtype in (torch.bfloat16, torch.float16):
            b.data = b.data.to(torch.float32)
    for sub in model.modules():
        if getattr(sub, "use_fp16", False):
            sub.use_fp16 = False
        if getattr(sub, "dtype", None) in (torch.float16, torch.bfloat16):
            sub.dtype = torch.float32


def _patch_flash_attn():
    """Replace flash_attn varlen with SDPA fallbacks."""
    try:
        import flash_attn as _fa
        import torch.nn.functional as _F

        def _sdpa_varlen(q, k, v, cu_q, cu_kv, max_q, max_kv,
                         dropout_p=0.0, softmax_scale=None, causal=False):
            scale = softmax_scale or (q.shape[-1] ** -0.5)
            batch = cu_q.shape[0] - 1
            parts = []
            for i in range(batch):
                qs, qe = cu_q[i].item(), cu_q[i + 1].item()
                ks, ke = cu_kv[i].item(), cu_kv[i + 1].item()
                qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
                ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
                vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
                o = _F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, scale=scale, is_causal=causal)
                parts.append(o.squeeze(0).transpose(0, 1))
            return torch.cat(parts, dim=0).unsqueeze(1)

        _fa.flash_attn_varlen_func = lambda q, k, v, cu_q, cu_kv, max_q, max_kv, **kw: \
            _sdpa_varlen(q, k, v, cu_q, cu_kv, max_q, max_kv)
        _fa.flash_attn_varlen_qkvpacked_func = lambda qkv, cu, ms, **kw: \
            _sdpa_varlen(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu, cu, ms, ms)
        _fa.flash_attn_varlen_kvpacked_func = lambda q, kv, cu_q, cu_kv, mq, mk, **kw: \
            _sdpa_varlen(q, kv[:, 0], kv[:, 1], cu_q, cu_kv, mq, mk)
    except ImportError:
        pass


def _patch_pytorch3d():
    """Patch pytorch3d ball_query and knn_points for CPU fallback."""
    import functools
    try:
        from pytorch3d.ops import ball_query as _oq, knn_points as _ok

        @functools.wraps(_oq)
        def _cpu_bq(p1, p2, K=1, radius=1.0, return_nn=False):
            dev = p1.device
            r = _oq(p1.cpu(), p2.cpu(), K=K, radius=radius, return_nn=return_nn)
            if isinstance(r, tuple):
                return tuple(x.to(dev) if x is not None else None for x in r)
            return r.to(dev)

        @functools.wraps(_ok)
        def _cpu_knn(p1, p2, lengths1=None, lengths2=None, **kw):
            dev = p1.device
            r = _ok(p1.cpu(), p2.cpu(),
                    lengths1=lengths1.cpu() if lengths1 is not None else None,
                    lengths2=lengths2.cpu() if lengths2 is not None else None, **kw)
            return r._replace(dists=r.dists.to(dev), idx=r.idx.to(dev),
                              knn=r.knn.to(dev) if r.knn is not None else None)

        import pytorch3d.ops.ball_query as _bqm
        import pytorch3d.ops.knn as _knnm
        _bqm.ball_query = _cpu_bq
        _knnm.knn_points = _cpu_knn

        for mod_path in [
            "anigen.representations.skeleton.grouping",
            "anigen.models.structured_latent_vae.anigen_decoder",
            "anigen.models.structured_latent_vae.anigen_encoder",
        ]:
            try:
                m = __import__(mod_path, fromlist=["ball_query", "knn_points"])
                if hasattr(m, "ball_query"):
                    m.ball_query = _cpu_bq
                if hasattr(m, "knn_points"):
                    m.knn_points = _cpu_knn
            except ImportError:
                pass
    except ImportError:
        pass
