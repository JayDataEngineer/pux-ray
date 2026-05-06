"""AniGen API server — runs inside Docker container.

Receives image via HTTP, generates rigged 3D GLB mesh with skeleton.
Used by Ray Serve deployment via runtime_env["container"].
"""
from __future__ import annotations

import os
import io
import tempfile
from pathlib import Path

os.environ.setdefault("FORCE_CUDA", "1")

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
import uvicorn

app = FastAPI(title="AniGen API")

_pipeline = None
SS_MODEL = os.environ.get("ANIGEN_SS_MODEL", "ckpts/anigen/ss_flow_duet")
SLAT_MODEL = os.environ.get("ANIGEN_SLAT_MODEL", "ckpts/anigen/slat_flow_auto")


@app.on_event("startup")
def load_model():
    global _pipeline
    import sys
    sys.path.insert(0, "/opt/anigen")
    from anigen.pipelines import AnigenImageTo3DPipeline
    _pipeline = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path=SS_MODEL,
        slat_flow_path=SLAT_MODEL,
    )
    _pipeline.cuda()
    print(f"AniGen loaded: ss={SS_MODEL}, slat={SLAT_MODEL}")


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Query(42),
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    result = _pipeline.run(img)

    response = {}
    for name, data in result.items():
        if hasattr(data, "export"):
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                data.export(tmp.name)
                response[name] = Path(tmp.name).read_bytes()
                Path(tmp.name).unlink(missing_ok=True)

    if not response:
        raise RuntimeError("AniGen produced no output")

    keys = list(response.keys())
    return JSONResponse({"status": "ok", "keys": keys, "count": len(response)})


@app.post("/generate/mesh")
async def generate_mesh(
    image: UploadFile = File(...),
    seed: int = Query(42),
):
    """Return just the mesh GLB bytes."""
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    torch.manual_seed(seed)
    result = _pipeline.run(img)

    data = result.get("mesh", result.get(list(result.keys())[0]))
    if hasattr(data, "export"):
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            data.export(tmp.name)
            glb_bytes = Path(tmp.name).read_bytes()
            Path(tmp.name).unlink(missing_ok=True)
        return Response(content=glb_bytes, media_type="model/gltf-binary")

    raise RuntimeError("No mesh in output")


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _pipeline is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
