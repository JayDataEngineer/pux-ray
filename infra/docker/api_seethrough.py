"""See-Through FastAPI server — anime character layer decomposition.

Uses the See-Through pipeline bundled in the gpu-all image at /opt/seethrough/.
Two-stage inference:
  1. LayerDiff 3D (SDXL-based UNet + TransparentVAE) — splits character into
     up to 23 semantic layers (hair, face, eyes, clothing, ...).
  2. Marigold depth (fine-tuned for anime) — estimates per-layer depth used
     to derive occlusion order in the exported PSD.

The base SDXL model (`frankjoshua/juggernautXL_version6Rundiffusion`) is the
upstream checkpoint that LayerDiff 3D finetunes; it is downloaded alongside
the LayerDiff repo and resolved automatically by diffusers when the local
cache is warm.

Endpoints:
  GET  /health
  POST /generate — image upload → layered PSD file (binary)
"""
from __future__ import annotations

import gc
import io
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
import uvicorn

# /opt/seethrough is the project root. /opt/seethrough/common is where
# `utils.inference_utils` lives. Add both to PYTHONPATH.
SEETHROUGH_SRC = os.environ.get("SEETHROUGH_SRC", "/opt/seethrough")
COMMON_SRC = os.environ.get("SEETHROUGH_COMMON_SRC", "/opt/seethrough/common")
for p in (SEETHROUGH_SRC, COMMON_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# chdir to project root so internal relative imports/assets resolve
os.chdir(SEETHROUGH_SRC)

# Model paths (host-mounted at /mnt/data/models/image/see-through)
MODEL_ROOT = os.environ.get(
    "SEETHROUGH_MODEL_ROOT", "/mnt/data/models/image/see-through"
)
LAYERDIFF_PATH = os.environ.get(
    "SEETHROUGH_LAYERDIFF_PATH", os.path.join(MODEL_ROOT, "layerdiff3d")
)
MARIGOLD_PATH = os.environ.get(
    "SEETHROUGH_MARIGOLD_PATH", os.path.join(MODEL_ROOT, "marigold")
)

DEFAULT_RESOLUTION = int(os.environ.get("SEETHROUGH_RESOLUTION", "1280"))
DEFAULT_RESOLUTION_DEPTH = int(os.environ.get("SEETHROUGH_RESOLUTION_DEPTH", "768"))
DEFAULT_INFERENCE_STEPS = int(os.environ.get("SEETHROUGH_INFERENCE_STEPS", "30"))
GROUP_OFFLOAD = os.environ.get("SEETHROUGH_GROUP_OFFLOAD", "0") == "1"

app = FastAPI(title="See-Through")

# Loaded lazily on first /generate (or on /load)
_layerdiff_pipe = None
_marigold_pipe = None
_loaded = False


def _check_models() -> list[str]:
    """Return list of missing model paths (empty if all OK)."""
    missing = []
    for label, path in (("layerdiff", LAYERDIFF_PATH), ("marigold", MARIGOLD_PATH)):
        if not os.path.isdir(path) or not any(
            f.endswith((".safetensors", ".bin", ".json"))
            for _, _, fs in os.walk(path)
            for f in fs
        ):
            missing.append(f"{label}={path}")
    return missing


def load_models():
    """Warm-load both pipelines. Subsequent /generate calls reuse them."""
    global _layerdiff_pipe, _marigold_pipe, _loaded
    if _loaded:
        return

    missing = _check_models()
    if missing:
        raise RuntimeError(
            f"See-Through models missing: {', '.join(missing)}. "
            "Set SEETHROUGH_MODEL_ROOT or download the repos."
        )

    # Defer imports until after sys.path manipulation so the bundled
    # /opt/seethrough/common/utils is what gets imported (not any host
    # `utils` package).
    from utils.inference_utils import apply_layerdiff, apply_marigold

    # apply_layerdiff / apply_marigold create their pipelines lazily on first
    # call. We trigger a dummy load by calling them with a 1x1 transparent
    # image so the pipelines are warm. Cheaper alternative: import the
    # underlying classes directly. Here we use a temp image to exercise the
    # real path including TransparentVAE.
    print(
        f"See-Through: warming layerdiff ({LAYERDIFF_PATH}) "
        f"and marigold ({MARIGOLD_PATH})..."
    )
    t0 = time.perf_counter()

    # Warm via real-but-tiny inference using a transparent 128² RGBA image.
    # This loads UNetFrameConditionModel, TransparentVAE, KDiffusionSDXL, and
    # MarigoldDepthPipeline. Reuses apply_layerdiff/apply_marigold so we get
    # the exact same code path as production inference.
    with tempfile.TemporaryDirectory() as tmpd:
        dummy_png = os.path.join(tmpd, "warmup.png")
        from PIL import Image
        Image.fromarray(
            np.zeros((128, 128, 4), dtype=np.uint8), mode="RGBA"
        ).save(dummy_png)

        try:
            apply_layerdiff(
                dummy_png, LAYERDIFF_PATH, save_dir=tmpd + "/out",
                resolution=512, num_inference_steps=1,
                disable_progressbar=True, group_offload=GROUP_OFFLOAD,
            )
        except Exception as e:
            # Warmup may fail at PSD composition (no layers found), but the
            # pipeline itself should be cached. Continue.
            print(f"  layerdiff warmup returned early ({e})")

        try:
            apply_marigold(
                dummy_png, MARIGOLD_PATH, save_dir=tmpd + "/out",
                resolution=512, num_inference_steps=1,
                disable_progressbar=True, group_offload=GROUP_OFFLOAD,
            )
        except Exception as e:
            print(f"  marigold warmup returned early ({e})")

    t1 = time.perf_counter()
    print(f"See-Through: warm load done in {t1 - t0:.1f}s")
    _loaded = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": _loaded,
        "layerdiff_path": LAYERDIFF_PATH,
        "marigold_path": MARIGOLD_PATH,
        "missing": _check_models(),
    }


@app.post("/load")
def load():
    try:
        load_models()
        return {"status": "ok", "loaded": _loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {e}")


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    resolution: int = Form(default=DEFAULT_RESOLUTION),
    resolution_depth: int = Form(default=DEFAULT_RESOLUTION_DEPTH),
    inference_steps: int = Form(default=DEFAULT_INFERENCE_STEPS),
    seed: int = Form(default=42),
    tblr_split: bool = Form(default=False),
):
    """Decompose an anime illustration into a layered PSD.

    Returns the PSD file as application/octet-stream with X-Inference-Time-S.
    """
    if not _loaded:
        try:
            load_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"model load failed: {e}")

    from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
    from utils.torch_utils import seed_everything

    # Persist uploaded image to a temp file (the pipeline expects a path).
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty image upload")

    workdir = tempfile.mkdtemp(prefix="seethrough_")
    srcname = Path(image.filename or "input.png").stem
    src_path = os.path.join(workdir, f"{srcname}.png")
    save_dir = os.path.join(workdir, "out")

    with open(src_path, "wb") as f:
        f.write(contents)

    t0 = time.perf_counter()
    try:
        seed_everything(seed)
        print(f"See-Through: layerdiff on {image.filename} "
              f"(res={resolution}, steps={inference_steps})")
        apply_layerdiff(
            src_path, LAYERDIFF_PATH, save_dir=save_dir, seed=seed,
            resolution=resolution,
            num_inference_steps=inference_steps,
            disable_progressbar=True, group_offload=GROUP_OFFLOAD,
        )

        print(f"See-Through: marigold on {image.filename} "
              f"(res={resolution_depth})")
        apply_marigold(
            src_path, MARIGOLD_PATH, save_dir=save_dir, seed=seed,
            resolution=resolution_depth,
            num_inference_steps=-1,
            disable_progressbar=True, group_offload=GROUP_OFFLOAD,
        )

        # Compose PSD — further_extr writes to <save_dir>/<srcname>.psd (one
        # level above the per-image subdir where layerdiff/marigold dropped
        # the per-tag PNGs).
        out_dir = os.path.join(save_dir, srcname)
        further_extr(out_dir, rotate=False, save_to_psd=True, tblr_split=tblr_split)

        # PSD path: see further_extr source — save_dir/out_dir_basename.psd
        expected_psd = os.path.join(save_dir, f"{srcname}.psd")
        if not os.path.isfile(expected_psd):
            # Fallback: glob for any PSD in save_dir
            psd_files = sorted(Path(save_dir).glob("*.psd"))
            if not psd_files:
                raise RuntimeError(
                    f"no PSD produced (expected {expected_psd}); "
                    f"save_dir contents: {os.listdir(save_dir) if os.path.isdir(save_dir) else 'MISSING'}"
                )
            expected_psd = str(psd_files[0])
        psd_path = expected_psd
        psd_bytes = Path(psd_path).read_bytes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = round(time.perf_counter() - t0, 2)

    # cleanup workdir (we already have the PSD bytes)
    try:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    return Response(
        content=psd_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Inference-Time-S": str(elapsed),
            "X-Output-File": os.path.basename(psd_path),
            "X-Output-Bytes": str(len(psd_bytes)),
            "Content-Disposition": f'attachment; filename="{srcname}.psd"',
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("SEETHROUGH_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
