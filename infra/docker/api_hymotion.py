"""HY-Motion 1.0 API server — text-to-3D human motion generation inside Docker.

Uses local_infer.py CLI (no Python pipeline import — the repo uses a CLI interface).
Models at /models/image-gen/comfyui/HY-Motion/ckpts/tencent/ (host-mounted).
"""
from __future__ import annotations

import os
import gc
import tempfile
import shutil
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import uvicorn

app = FastAPI(title="HY-Motion API")
MODEL_PATH = os.environ.get(
    "HYMOTION_MODEL_PATH",
    "/models/image-gen/comfyui/HY-Motion/ckpts/tencent/HY-Motion-1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/generate")
async def generate(data: dict):
    """Generate 3D human motion from text prompt. Returns GLB/FBX/NPZ."""
    prompt = data.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    duration = data.get("duration", 5.0)
    seed = data.get("seed", 42)
    fmt = data.get("format", "glb")
    if fmt not in ("glb", "fbx", "npz"):
        return JSONResponse({"error": f"unsupported format: {fmt}"}, status_code=400)

    tmpdir = tempfile.mkdtemp(prefix="hymotion_")
    try:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        duration_frames = int(duration * 30)
        prompt_file = input_dir / "prompt.txt"
        prompt_file.write_text(f"{prompt}#{duration_frames}#001\n")

        import subprocess
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

        data_bytes = output_files[0].read_bytes()
        media_types = {"glb": "model/gltf-binary", "fbx": "application/octet-stream", "npz": "application/octet-stream"}
        torch.cuda.empty_cache()
        gc.collect()
        return Response(content=data_bytes, media_type=media_types.get(fmt, "application/octet-stream"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
