"""TRELLIS.2-4B image-to-3D API server (native, not Wan2GP).

Loads the Trellis2ImageTo3DPipeline from /opt/trellis/trellis2 (vendor code mounted
at /opt/trellis/) and produces a textured GLB mesh using o_voxel.postprocess.to_glb.

Output GLB uses 1M-face decimation + 4096px texture by default; pass
?resolution=1024 (cascaded 512→1024) for higher quality (slower).

Endpoints:
  GET  /health
  POST /generate   — multipart image upload → GLB binary
"""
from __future__ import annotations

import gc
import io
import os
import sys
import time
import tempfile
from pathlib import Path

# Apply env config BEFORE importing trellis2 (flash-attn backend, spconv algo).
os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# /opt/trellis must be on PYTHONPATH so `from trellis2.pipelines import ...` works.
TRELLIS_SRC = os.environ.get("TRELLIS_SRC", "/opt/trellis")
if TRELLIS_SRC not in sys.path:
    sys.path.insert(0, TRELLIS_SRC)

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import Response
import uvicorn

app = FastAPI(title="TRELLIS.2-4B API")
pipeline = None  # type: ignore

MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "/mnt/data/models/3d/trellis/TRELLIS.2-4B/ckpts")
DEFAULT_PIPELINE_TYPE = os.environ.get("TRELLIS2_PIPELINE_TYPE", "1024_cascade")


@app.on_event("startup")
def _load():
    global pipeline
    if pipeline is not None:
        return
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print(f"[startup] Loading TRELLIS.2-4B from {MODEL_PATH} (pipeline={DEFAULT_PIPELINE_TYPE})")
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.cuda()  # moves to GPU (low_vram=True by default → lazy sub-pipeline loads)
    print(f"[startup] TRELLIS.2-4B loaded in {time.perf_counter()-t0:.1f}s")
    print(f"[startup] low_vram={pipeline.low_vram}, default_pipeline_type={pipeline.default_pipeline_type}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TRELLIS.2-4B",
        "loaded": pipeline is not None,
        "model_path": MODEL_PATH,
        "pipeline_type": DEFAULT_PIPELINE_TYPE,
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Query(42),
    decimation: int = Query(1_000_000),
    texture_size: int = Query(4096),
    resolution: int = Query(1024, description="Target voxel resolution: 512 | 1024 | 1536"),
    max_num_tokens: int = Query(49_152),
):
    """Generate a textured GLB from an input image.

    Resolution maps to pipeline_type:
      512  → '512'         (fastest, ~3s on H100, ~15s on 4090)
      1024 → '1024_cascade' (default, ~17s on H100, ~60-90s on 4090)
      1536 → '1536_cascade' (slowest, highest quality)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty image upload")
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pipeline_type_map = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}
    if resolution not in pipeline_type_map:
        raise HTTPException(status_code=400, detail=f"resolution must be in {list(pipeline_type_map)}")
    pipeline_type = pipeline_type_map[resolution]

    t0 = time.perf_counter()
    try:
        outputs = pipeline.run(
            pil,
            seed=seed,
            pipeline_type=pipeline_type,
            max_num_tokens=max_num_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline.run failed: {e}")

    if not outputs:
        raise HTTPException(status_code=500, detail="pipeline produced no mesh")
    mesh = outputs[0]

    # Mesh decimation (cumesh). Bounded by nvdiffrast 16M-face limit.
    try:
        mesh.simplify(min(decimation, 16_777_216))
    except Exception as e:
        print(f"[warn] mesh.simplify failed (continuing): {e}")

    # GLB baking via o_voxel.postprocess.to_glb.
    from o_voxel.postprocess import to_glb
    try:
        glb = to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"to_glb failed: {e}")

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb.export(tmp.name, extension_webp=True)
        data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)

    elapsed = round(time.perf_counter() - t0, 2)

    del outputs, mesh, glb
    torch.cuda.empty_cache()
    gc.collect()

    return Response(
        content=data,
        media_type="model/gltf-binary",
        headers={
            "X-Inference-Time-S": str(elapsed),
            "X-Resolution": str(resolution),
            "X-Pipeline-Type": pipeline_type,
            "X-Decimation": str(decimation),
            "X-Texture-Size": str(texture_size),
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("TRELLIS2_PORT", "8099"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
