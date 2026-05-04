"""HY-Motion 1.0 — Text-to-3D human motion generation.

Generates skeleton-based 3D character animations from text prompts.
Runs inside Ray-managed container (tech-noir/hymotion:latest).

Requires ~26GB VRAM for HY-Motion-1.0, ~24GB for Lite variant.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import shutil
import tempfile
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "HYMOTION_MODEL_PATH",
    "/models/image-gen/comfyui/HY-Motion/ckpts/tencent/HY-Motion-1.0",
)


@serve.deployment(
    name="hy_motion",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class HYMotionDeployment(BaseGPUDeployment):
    """HY-Motion text-to-3D motion via Ray native container."""

    def _load(self, model_name: str = "hy-motion-1.0") -> None:
        self.model_name = model_name
        self.model = True
        logger.info("HY-Motion ready (model_path=%s)", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None

    async def __call__(self, request):
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        fmt = body.get("format", "glb")
        if fmt not in ("glb", "fbx", "npz"):
            return JSONResponse({"error": f"unsupported format: {fmt}"}, status_code=400)

        data = await asyncio.to_thread(
            self._generate,
            prompt=prompt,
            duration=body.get("duration", 5.0),
            seed=body.get("seed", 42),
            fmt=fmt,
        )
        media_types = {"glb": "model/gltf-binary", "fbx": "application/octet-stream", "npz": "application/octet-stream"}
        return Response(content=data, media_type=media_types.get(fmt, "application/octet-stream"))

    def _generate(self, prompt: str, duration: float, seed: int, fmt: str) -> bytes:
        import subprocess
        import torch

        tmpdir = tempfile.mkdtemp(prefix="hymotion_")
        try:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            duration_frames = int(duration * 30)
            (input_dir / "prompt.txt").write_text(f"{prompt}#{duration_frames}#001\n")

            result = subprocess.run(
                [
                    "python", "/opt/hymotion/local_infer.py",
                    "--model_path", MODEL_PATH,
                    "--input_text_dir", str(input_dir),
                    "--output_dir", str(output_dir),
                    "--num_seeds", "1",
                    "--seed", str(seed),
                    "--disable_duration_est",
                    "--disable_rewrite",
                ],
                capture_output=True, text=True, timeout=300,
                cwd="/opt/hymotion",
            )
            if result.returncode != 0:
                raise RuntimeError(f"HY-Motion failed: {result.stderr[-500:]}")

            ext_map = {"glb": "*.glb", "fbx": "*.fbx", "npz": "*.npz"}
            output_files = sorted(output_dir.rglob(ext_map[fmt]))
            if not output_files:
                output_files = sorted(output_dir.rglob("*.glb"))
            if not output_files:
                output_files = sorted(output_dir.rglob("*.npz"))
            if not output_files:
                raise RuntimeError("No output files produced")

            return output_files[0].read_bytes()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            torch.cuda.empty_cache()
            gc.collect()
