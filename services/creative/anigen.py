"""AniGen — Animated 3D asset generation from images (Ray-native).

Generates rigged, skinned 3D meshes (GLB) from single character images.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from ray import serve
from starlette.responses import JSONResponse

from PIL import Image

from services.base import BaseGPUDeployment, InferenceConfig, _b64_decode

logger = logging.getLogger(__name__)


@serve.deployment(
    name="anigen",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "FORCE_CUDA": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
    },
)
class AniGenDeployment(BaseGPUDeployment):
    """AniGen image-to-3D via native PyTorch inference."""

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def _load(self, model_name: str = "anigen") -> None:
        from services.compat import apply as _apply_compat
        _apply_compat()

        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()

        model_path = registry.get_path("3d", model_name)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"AniGen model not found at {model_path}. "
                f"Check model_registry.yaml '3d.anigen' entry."
            )

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        if self.config.low_resource:
            logger.info("AniGen LOW_RESOURCE mode — fp16, reduced steps")
            self.config.precision = "fp16"

        os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("U2NET_DEVICE", "cpu")

        # torch.hub.load in the pipeline uses ./ckpts/ relative to CWD.
        # model_path already includes ckpts/ (from registry path). The actual
        # ckpts dir with submodels is at model_path/ckpts/, so set CWD to
        # model_path itself so that ./ckpts/ resolves to model_path/ckpts/.
        old_cwd = os.getcwd()
        os.chdir(str(model_path))

        logger.info("Loading AniGen pipeline from %s (CWD=%s)", model_path, model_path)

        # anigen/modules/sparse's backend detection calls
        # importlib.util.find_spec('xformers.ops') which raises ModuleNotFoundError
        # in Python 3.12+ when xformers itself is absent. Patch to return None.
        import importlib.util as _iu
        _orig_find_spec = _iu.find_spec
        def _safe_find_spec(name, *args, **kwargs):
            try:
                return _orig_find_spec(name, *args, **kwargs)
            except (ModuleNotFoundError, ValueError):
                return None
        _iu.find_spec = _safe_find_spec

        from anigen.pipelines import AnigenImageTo3DPipeline

        # rembg uses ONNX Runtime which defaults to CUDA, consuming GPU memory
        # needed by AniGen. Force it to CPU.
        import functools, rembg
        _orig_new_session = rembg.new_session
        @functools.wraps(_orig_new_session)
        def _cpu_new_session(*args, **kwargs):
            kwargs.setdefault("providers", ["CPUExecutionProvider"])
            return _orig_new_session(*args, **kwargs)
        rembg.new_session = _cpu_new_session

        ckpts_dir = str(model_path / "ckpts" / "anigen")
        ss_flow_path = str(Path(ckpts_dir) / "ss_flow_duet")
        slat_flow_path = str(Path(ckpts_dir) / "slat_flow_auto")

        self.pipeline = AnigenImageTo3DPipeline.from_pretrained(
            ss_flow_path=ss_flow_path,
            slat_flow_path=slat_flow_path,
        )

        os.chdir(old_cwd)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(device)

        # spconv's dtype mapping (_TORCH_DTYPE_TO_TV in cppcore.py) does NOT
        # include torch.bfloat16, and the flow models may have called
        # convert_to_fp16() internally (converting blocks to fp16). Normalize
        # all parameters across every sub-model to fp32.
        for m in self.pipeline.models.values():
            if not isinstance(m, torch.nn.Module):
                continue
            for p in m.parameters():
                if p.dtype in (torch.bfloat16, torch.float16):
                    p.data = p.data.to(torch.float32)
            for b in m.buffers():
                if b.dtype in (torch.bfloat16, torch.float16):
                    b.data = b.data.to(torch.float32)

        # The flow models and decoders may have use_fp16=True set internally,
        # which causes forward() to cast all inputs to fp16, conflicting with
        # the fp32 params we set above. Sweep all submodules for dtype flags.
        for m in self.pipeline.models.values():
            if not isinstance(m, torch.nn.Module):
                continue
            for sub in m.modules():
                if getattr(sub, "use_fp16", False):
                    sub.use_fp16 = False
                if getattr(sub, "dtype", None) in (torch.float16, torch.bfloat16):
                    sub.dtype = torch.float32

        # vllm-flash-attn compiled extension has arg count mismatches AND
        # only supports fp16/bf16 (not fp32). Replace all varlen functions
        # with pure PyTorch SDPA fallbacks that work with any dtype.
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
                o = _F.scaled_dot_product_attention(
                    qi, ki, vi, dropout_p=dropout_p,
                    scale=scale, is_causal=causal)
                parts.append(o.squeeze(0).transpose(0, 1))
            return torch.cat(parts, dim=0).unsqueeze(1)

        def _sdpa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen,
                                   dropout_p=0.0, softmax_scale=None,
                                   causal=False, **_kw):
            return _sdpa_varlen(
                qkv[:, 0], qkv[:, 1], qkv[:, 2],
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                dropout_p, softmax_scale, causal)

        def _sdpa_varlen_kvpacked(q, kv, cu_q, cu_kv, max_q, max_kv,
                                  dropout_p=0.0, softmax_scale=None,
                                  causal=False, **_kw):
            return _sdpa_varlen(
                q, kv[:, 0], kv[:, 1],
                cu_q, cu_kv, max_q, max_kv,
                dropout_p, softmax_scale, causal)

        def _sdpa_varlen_func(q, k, v, cu_q, cu_kv, max_q, max_kv,
                              dropout_p=0.0, softmax_scale=None,
                              causal=False, **_kw):
            return _sdpa_varlen(
                q, k, v, cu_q, cu_kv, max_q, max_kv,
                dropout_p, softmax_scale, causal)

        _fa.flash_attn_varlen_qkvpacked_func = _sdpa_varlen_qkvpacked
        _fa.flash_attn_varlen_kvpacked_func = _sdpa_varlen_kvpacked
        _fa.flash_attn_varlen_func = _sdpa_varlen_func

        # pytorch3d is installed without CUDA support. Patch ball_query and
        # knn_points to fall back to CPU transparently.
        try:
            from pytorch3d.ops import ball_query as _orig_ball_query, knn_points as _orig_knn_points
            import functools

            @functools.wraps(_orig_ball_query)
            def _cpu_ball_query(p1, p2, K=1, radius=1.0, return_nn=False):
                dev = p1.device
                cpu_p1, cpu_p2 = p1.cpu(), p2.cpu()
                result = _orig_ball_query(cpu_p1, cpu_p2, K=K, radius=radius, return_nn=return_nn)
                if isinstance(result, tuple):
                    return tuple(r.to(dev) if r is not None else None for r in result)
                return result.to(dev)

            @functools.wraps(_orig_knn_points)
            def _cpu_knn_points(p1, p2, lengths1=None, lengths2=None,
                                norm=2, K=1, version=-1,
                                return_nn=False, return_sorted=True):
                dev = p1.device
                result = _orig_knn_points(
                    p1.cpu(), p2.cpu(),
                    lengths1=lengths1.cpu() if lengths1 is not None else None,
                    lengths2=lengths2.cpu() if lengths2 is not None else None,
                    norm=norm, K=K, version=version,
                    return_nn=return_nn, return_sorted=return_sorted,
                )
                return result._replace(
                    dists=result.dists.to(dev),
                    idx=result.idx.to(dev),
                    knn=result.knn.to(dev) if result.knn is not None else None,
                )

            import pytorch3d.ops.ball_query as _bq_mod
            import pytorch3d.ops.knn as _knn_mod
            _bq_mod.ball_query = _cpu_ball_query
            _knn_mod.knn_points = _cpu_knn_points

            # Several anigen modules import these functions at module level
            # via `from pytorch3d.ops import ball_query, knn_points`, so they
            # hold direct references. Patch all known importers.
            from anigen.representations.skeleton import grouping as _grp_mod
            _grp_mod.ball_query = _cpu_ball_query
            _grp_mod.knn_points = _cpu_knn_points

            from anigen.models.structured_latent_vae import anigen_decoder as _dec_mod
            _dec_mod.knn_points = _cpu_knn_points

            from anigen.models.structured_latent_vae import anigen_encoder as _enc_mod
            _enc_mod.knn_points = _cpu_knn_points

            try:
                from anigen.representations.mesh import cube2mesh_skeleton as _c2m_mod
                _c2m_mod.knn_points = _cpu_knn_points
            except ImportError:
                pass
        except ImportError:
            pass

        self.model = True
        self.model_name = model_name
        torch.cuda.empty_cache()
        gc.collect()

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("AniGen loaded (precision=%s, low_resource=%s, VRAM=%.0fMB)",
                    self.config.precision, self.config.low_resource, vram)

    def _unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {image_b64, seed}, config}."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "precision": self.config.precision,
                "low_resource": self.config.low_resource,
            }

        start = time.perf_counter()

        try:
            content_type = request.headers.get("content-type", "")
            img = None
            seed = 42

            if "multipart/form-data" in content_type:
                form = await request.form()
                if "config" in form:
                    requested = InferenceConfig(**json.loads(str(form["config"])))
                    if requested != self.config:
                        self.config = requested

                image_file = form.get("image") or form.get("file")
                if not image_file:
                    return JSONResponse(self.handle_error("image file required"), status_code=400)

                img_bytes = await image_file.read()
                img = Image.open(io.BytesIO(img_bytes))
                seed = int(form.get("seed", 42))
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                img_bytes = extracted.get("image")
                if not img_bytes:
                    return JSONResponse(self.handle_error("image_b64 required"), status_code=400)

                img = Image.open(io.BytesIO(img_bytes))
                seed = extracted.get("seed", 42)

            if not self.is_loaded():
                import asyncio
                await asyncio.to_thread(self.load_model, "anigen")

            path = request.url.path
            if path.endswith("/mesh"):
                result = await self._infer_mesh(img, seed)
            else:
                result = await self._infer(img, seed)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                )
            )
        except Exception as e:
            logger.error("AniGen error: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    async def _infer_mesh(self, img, seed: int) -> dict:
        def _run():
            result = self._run_pipeline(img, seed)
            mesh = result.get("mesh")
            if mesh is None:
                return {"data": json.dumps({"error": "No mesh produced"}).encode(), "media_type": "application/json"}

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                mesh.export(tmp.name, file_type="glb")
                data = Path(tmp.name).read_bytes()
                Path(tmp.name).unlink(missing_ok=True)

            logger.info("AniGen mesh done: %dKB GLB", len(data) // 1024)
            return {"data": data, "media_type": "model/gltf-binary"}

        import asyncio
        return await asyncio.to_thread(_run)

    async def _infer(self, img, seed: int) -> dict:
        def _run():
            result = self._run_pipeline(img, seed)
            keys = list(result.keys())
            mesh = result.get("mesh")
            mesh_info = None
            if mesh is not None:
                mesh_info = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}

            return {
                "data": json.dumps({"status": "ok", "seed": seed, "mesh": mesh_info, "keys": keys}).encode(),
                "media_type": "application/json",
            }

        import asyncio
        return await asyncio.to_thread(_run)

    def _run_pipeline(self, img, seed: int) -> dict:
        ss_steps = 25
        slat_steps = 25
        cfg_scale_ss = 7.5
        cfg_scale_slat = 3.0

        if self.config.low_resource:
            ss_steps = 4
            slat_steps = 4

        try:
            with torch.no_grad():
                return self.pipeline.run(
                    img,
                    seed=seed,
                    cfg_scale_ss=cfg_scale_ss,
                    cfg_scale_slat=cfg_scale_slat,
                    ss_steps=ss_steps,
                    slat_steps=slat_steps,
                    texture_size=512 if self.config.low_resource else 1024,
                )
        except Exception as e:
            logger.error("AniGen inference failed: %s", e, exc_info=True)
            raise