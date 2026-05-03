"""TRELLIS.2 API server — runs inside Docker container.

Receives image via HTTP, generates 3D GLB mesh, returns binary.
Used by Ray Serve deployment via runtime_env["container"].
"""
from __future__ import annotations

import os
import io
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
import uvicorn

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

app = FastAPI(title="TRELLIS.2 API")
pipeline: Trellis2ImageTo3DPipeline | None = None
MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")


@app.on_event("startup")
def load_model():
    global pipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipeline.cuda()
    print(f"TRELLIS.2 loaded: {MODEL_ID}")


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    resolution: int = Query(512),
    decimation: int = Query(1_000_000),
    simplify: int = Query(16_777_216),
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    mesh = pipeline.run(img)[0]
    mesh.simplify(simplify)

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size={256: 2048, 512: 4096, 1024: 4096, 2048: 8192}.get(resolution, 4096),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(tmp.name, extension_webp=True)
        data = Path(tmp.name).read_bytes()

    Path(tmp.name).unlink(missing_ok=True)
    return Response(content=data, media_type="model/gltf-binary")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": pipeline is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
