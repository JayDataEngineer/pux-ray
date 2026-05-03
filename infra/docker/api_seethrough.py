"""See-Through API server — anime character layer decomposition."""
from __future__ import annotations

import os
import io
import tempfile
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="See-Through API")


@app.on_event("startup")
def load_model():
    import sys
    sys.path.insert(0, "/opt/seethrough")
    sys.path.insert(0, "/opt/seethrough/inference")
    sys.path.insert(0, "/opt/seethrough/inference/scripts")
    print("See-Through ready (model loads on first request)")


@app.post("/decompose")
async def decompose(
    image: UploadFile = File(...),
    resolution: int = Query(1280),
    inference_steps: int = Query(30),
):
    img_bytes = await image.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.png"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        input_path.write_bytes(img_bytes)

        import subprocess
        result = subprocess.run([
            "python", "/opt/seethrough/inference/scripts/inference_psd.py",
            "--srcp", str(input_path),
            "--save_dir", str(output_dir),
            "--resolution", str(resolution),
            "--inference_steps", str(inference_steps),
            "--save_to_psd",
        ], capture_output=True, text=True, timeout=600, cwd="/opt/seethrough")

        if result.returncode != 0:
            raise RuntimeError(f"See-Through failed: {result.stderr}")

        layers = []
        psd_data = None
        for png in sorted(output_dir.rglob("*.png")):
            if "layer" in png.name.lower() or "part" in png.name.lower():
                layers.append({"name": png.stem})
        for psd in output_dir.rglob("*.psd"):
            psd_data = psd.read_bytes()
            break

        return JSONResponse({
            "layers": layers,
            "has_psd": psd_data is not None,
        })


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
