"""TRELLIS StableProjectorz API server — VRAM-optimized 3D generation.

Uses trellis-stable-projectorz (float16, int32) — fits in 8GB VRAM cards.
API contract matches the existing tech-noir TRELLIS API for drop-in replacement.

Endpoints:
  POST /generate   — Image to 3D GLB mesh
  GET  /health     — Health check
"""
from __future__ import annotations

import os
import io
import gc
import tempfile
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
import uvicorn

app = FastAPI(title="TRELLIS SPZ API")
pipeline = None
MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "jetx/TRELLIS-image-large")


@app.on_event("startup")
def load_model():
    global pipeline
    import sys
    sys.path.insert(0, "/opt/trellis")
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipeline.to(torch.float16)
    if "image_cond_model" in pipeline.models:
        pipeline.models["image_cond_model"].half()

    torch.cuda.empty_cache()
    gc.collect()
    print(f"TRELLIS SPZ loaded: {MODEL_ID} (float16, VRAM-optimized)")


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    resolution: int = Query(512),
    decimation: int = Query(950000),
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    outputs = pipeline.run(
        img,
        seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
        formats=["mesh", "gaussian"],
    )

    simplify_ratio = min(decimation / 1_000_000, 1.0)
    texture_size = {256: 2048, 512: 4096, 1024: 4096, 2048: 8192}.get(resolution, 4096)

    from trellis.utils import postprocessing_utils

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify_ratio,
        texture_size=texture_size,
    )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb.export(tmp.name)
        data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)

    del outputs, glb
    torch.cuda.empty_cache()
    gc.collect()

    return Response(content=data, media_type="model/gltf-binary")


@app.get("/health")
def health():
    mtype = "spz-float16"
    return {"status": "ok", "model": MODEL_ID, "loaded": pipeline is not None, "variant": mtype}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
