"""HY-Motion 1.0 — Text-to-3D human motion generation (Ray-native).

Generates skeleton-based 3D character animations from text prompts.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig

logger = logging.getLogger(__name__)

# Absolute paths to text encoder models on the PVC.
# The vendor code (hymotion) expects ckpts/Qwen3-8B and ckpts/clip-vit-large-patch14
# relative to CWD. Our service sets up a workspace that satisfies this.
CLIP_PVC_PATH = "/models/image-gen/comfyui/HY-Motion/ckpts/clip-vit-large-patch14"
QWEN_PVC_BASE = "/models/motion/{model}/ckpts/Qwen3-8B"
STATS_VENDOR_PATH = "/opt/hymotion/stats"


@serve.deployment(
    name="hy_motion",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 0},
)
class HYMotionDeployment(BaseGPUDeployment):
    """HY-Motion text-to-3D human motion via native PyTorch inference."""
    vram_mb = 6_144

    def __init__(self):
        super().__init__()
        self.runtime = None

    def _load(self, model_name: str = "hy-motion-1.0") -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()

        vendor_path = Path(cfg.project_root) / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        if self.config.low_resource:
            logger.info("HY-Motion LOW_RESOURCE mode — lite model, fp16, no prompt rewriter")
            model_name = "hy-motion-1.0-lite"
            self.config.precision = "fp16"
            disable_prompt_eng = True
        else:
            disable_prompt_eng = False

        model_path = registry.get_path("motion", model_name)
        config_yml = model_path / "config.yml"
        ckpt_file = model_path / "latest.ckpt"

        if not config_yml.exists():
            raise FileNotFoundError(
                f"HY-Motion config.yml not found at {config_yml}. "
                f"Download from: hf://tencent/HY-Motion-1.0[-Lite]"
            )
        if not ckpt_file.exists():
            raise FileNotFoundError(f"HY-Motion checkpoint not found at {ckpt_file}")

        prompter_path = os.environ.get(
            "TEXT2MOTION_PROMPTER_PATH",
            str(Path(cfg.models_root) / "motion" / "text2motion-prompter"),
        )
        if not Path(prompter_path).is_dir():
            logger.warning("text2motion-prompter not found at %s — disabling prompt engineering", prompter_path)
            disable_prompt_eng = True

        target_dtype = self.config.dtype()
        logger.info(
            "Loading HY-Motion %s (dtype=%s, low_resource=%s): config=%s ckpt=%s",
            model_name, self.config.precision, self.config.low_resource,
            config_yml, ckpt_file,
        )

        # Set up workspace: writable tmp dir with ckpts/ pointing to PVC models
        # and stats/ pointing to vendor stats. Vendor code resolves paths relative to CWD.
        import tempfile
        workspace = Path(tempfile.mkdtemp(prefix="hymotion_"))
        ckpts_dir = workspace / "ckpts"
        ckpts_dir.mkdir()
        qwen_src = Path(QWEN_PVC_BASE.format(model=model_name))
        if not qwen_src.is_dir():
            qwen_src = Path(QWEN_PVC_BASE.format(model="hy-motion-1.0"))
        if qwen_src.is_dir():
            (ckpts_dir / "Qwen3-8B").symlink_to(qwen_src)
        if Path(CLIP_PVC_PATH).is_dir():
            (ckpts_dir / "clip-vit-large-patch14").symlink_to(CLIP_PVC_PATH)
        if Path(STATS_VENDOR_PATH).is_dir():
            (workspace / "stats").symlink_to(STATS_VENDOR_PATH)
        logger.info("Workspace at %s: %s", workspace, list(workspace.iterdir()))

        # Patch config to resolve mean_std_dir to absolute path
        import yaml
        with open(config_yml) as f:
            config_dict = yaml.safe_load(f)
        test_cfg = config_dict.get("train_pipeline_args", {}).get("test_cfg", {})
        msd = test_cfg.get("mean_std_dir", "")
        if msd and not os.path.isabs(msd):
            config_dict["train_pipeline_args"]["test_cfg"]["mean_std_dir"] = str(workspace / msd)
        patched_yml = workspace / "config.yml"
        with open(patched_yml, "w") as f:
            yaml.dump(config_dict, f)

        from hymotion.utils.t2m_runtime import T2MRuntime

        prev_cwd = os.getcwd()
        os.chdir(str(workspace))
        try:
            self.runtime = T2MRuntime(
                config_path=str(patched_yml),
                ckpt_name=str(ckpt_file),
                skip_text=False,
                device_ids=None,
                force_cpu=False,
                disable_prompt_engineering=disable_prompt_eng,
                prompt_engineering_model_path=prompter_path if not disable_prompt_eng else None,
            )
        finally:
            os.chdir(prev_cwd)

        if target_dtype != torch.bfloat16:
            for pipeline in self.runtime.pipelines:
                pipeline.to(target_dtype)
            logger.info("HY-Motion cast to %s", self.config.precision)

        self.model = True
        self.model_name = model_name
        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("HY-Motion loaded (VRAM: %.0fMB, dtype: %s)", vram, self.config.precision)

    def _unload(self) -> None:
        if self.runtime is not None:
            del self.runtime
            self.runtime = None
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, seed, duration, cfg_scale}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if "config" in body:
                requested = InferenceConfig(**body["config"])
                if requested != self.config:
                    self.config = requested
                    if self.runtime is not None:
                        target_dtype = self.config.dtype()
                        for pipeline in self.runtime.pipelines:
                            pipeline.to(target_dtype)
                        logger.info("HY-Motion precision switched to %s", self.config.precision)

            import asyncio

            model_name = extracted.get("model", self.model_name or "hy-motion-1.0")
            if not self.is_loaded() or self.model_name != model_name:
                await asyncio.to_thread(self.load_model, model_name)

            result = await self._infer(extracted)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    json.dumps(result).encode("utf-8"),
                    "application/json",
                    latency_ms,
                )
            )
        except Exception as e:
            logger.error("hy_motion error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    async def _infer(self, extracted: dict) -> dict:
        def _run():
            text = extracted.get("text", "")
            if not text:
                return {"error": "Missing 'text' field"}

            seeds_csv = str(extracted.get("seed", extracted.get("seeds_csv", "42")))
            duration = float(extracted.get("duration", 3.0))
            cfg_scale = float(extracted.get("cfg_scale", extracted.get("guidance", 3.0)))
            output_format = extracted.get("output_format", "dict")

            logger.info(
                "HY-Motion generate: text=%r dur=%.1fs seeds=%s cfg=%.1f",
                text[:80], duration, seeds_csv, cfg_scale,
            )

            try:
                html_content, fbx_files, model_output = self.runtime.generate_motion(
                    text=text,
                    seeds_csv=seeds_csv,
                    duration=duration,
                    cfg_scale=cfg_scale,
                    output_format=output_format,
                )
            except Exception as e:
                logger.error("HY-Motion inference failed: %s", e, exc_info=True)
                return {"error": str(e)}

            motion_keys = []
            if isinstance(model_output, dict):
                motion_keys = [
                    k for k in model_output
                    if isinstance(model_output[k], torch.Tensor)
                ]

            return {
                "status": "success",
                "text": text,
                "duration": duration,
                "cfg_scale": cfg_scale,
                "seeds": [int(s.strip()) for s in seeds_csv.split(",") if s.strip()],
                "output_format": output_format,
                "precision": self.config.precision,
                "html_visualization": html_content if output_format != "dict" else None,
                "fbx_files": fbx_files if output_format == "fbx" else [],
                "motion_data": {
                    k: {"shape": list(model_output[k].shape), "dtype": str(model_output[k].dtype)}
                    for k in motion_keys
                } if isinstance(model_output, dict) else "non-dict output",
            }

        import asyncio
        return await asyncio.to_thread(_run)
