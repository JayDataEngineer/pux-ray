"""Kimodo-SOMA-RP FastAPI server — text-to-3D human motion generation.

Uses NVIDIA's Kimodo model (Apache-2.0) via the `kimodo` Python package
bundled in the gpu-all image at /opt/kimodo/. The LLM2Vec text encoder
(Llama-3-8B-Instruct) runs on CPU to free ~14 GB VRAM and avoid dtype
mismatches; the diffusion denoiser runs on GPU.

Endpoints:
  GET  /health
  POST /generate  — text → motion NPZ
"""
from __future__ import annotations

import gc
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

# /opt/kimodo must be on PYTHONPATH so `import kimodo` resolves.
KIMODO_SRC = os.environ.get("KIMODO_SRC", "/opt/kimodo")
if KIMODO_SRC not in sys.path:
    sys.path.insert(0, KIMODO_SRC)

# Kimodo checkpoints live under this path (snapshot layout).
# Resolve "Kimodo-SOMA-RP-v1.1" → /mnt/data/models/avatar/kimodo/Kimodo-SOMA-RP-v1.1
MODEL_ROOT = os.environ.get(
    "KIMODO_MODEL_ROOT", "/mnt/data/models/avatar/kimodo"
)
VARIANT = os.environ.get("KIMODO_VARIANT", "Kimodo-SOMA-RP-v1.1")
DEFAULT_MODEL_PATH = os.path.join(MODEL_ROOT, VARIANT)

# Text encoder LLM2Vec-Meta-Llama-3-8B-Instruct-mntp lives here on host
LLM2VEC_ROOT = os.environ.get(
    "KIMODO_LLM2VEC_ROOT",
    "/mnt/data/models/cache/huggingface/McGill-NLP",
)
TEXT_ENCODER_DEVICE = os.environ.get("TEXT_ENCODER_DEVICE", "cpu")

app = FastAPI(title="Kimodo-SOMA-RP")
_model = None
_loaded_variant: str | None = None


def _resolve_variant_dir(requested: str) -> str:
    """Return the on-disk dir for a Kimodo variant (e.g. Kimodo-SOMA-RP-v1.1)."""
    candidate = os.path.join(MODEL_ROOT, requested)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Kimodo variant {requested!r} not found under {MODEL_ROOT}. "
        f"Available: {os.listdir(MODEL_ROOT) if os.path.isdir(MODEL_ROOT) else 'NONE'}"
    )


def load_model(variant: str = VARIANT):
    """Load Kimodo model. Runs text encoder on CPU to avoid dtype issues on RTX 4090."""
    global _model, _loaded_variant
    if _model is not None and _loaded_variant == variant:
        return _model

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN required for Kimodo (gated Llama-3-8B-Instruct text encoder)."
        )

    os.environ["TEXT_ENCODER_DEVICE"] = TEXT_ENCODER_DEVICE
    # Force local_files_only so Kimodo uses the host-mounted LLM2Vec cache
    os.environ.setdefault("LOCAL_CACHE", "True")

    # Point HuggingFace caches at our host-mounted snapshot dirs.
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/models/cache/huggingface")
    os.environ.setdefault("HF_HOME", "/mnt/data/models/cache/huggingface")
    # TEXT_ENCODERS_DIR is read by kimodo's LLM2VecEncoder wrapper to resolve
    # "McGill-NLP/LLM2Vec-..." into a local dir, avoiding HF Hub lookups.
    os.environ.setdefault("TEXT_ENCODERS_DIR", "/mnt/data/models/cache/huggingface")

    # Critical: set CHECKPOINT_DIR so kimodo.load_model bypasses snapshot_download
    # (which fails offline for gated repos). The folder name is the display_name
    # (e.g. /mnt/data/.../Kimodo-SOMA-RP-v1.1/) — load_model joins CHECKPOINT_DIR
    # with info.display_name.
    variant_dir = _resolve_variant_dir(variant)
    os.environ["CHECKPOINT_DIR"] = MODEL_ROOT

    from kimodo import load_model as kimodo_load

    print(f"Kimodo: loading {variant} from {variant_dir} (text_encoder={TEXT_ENCODER_DEVICE})")
    t0 = time.perf_counter()

    # Pass the display_name (e.g. "Kimodo-SOMA-RP-v1.1") — resolve_model_name()
    # accepts it case-insensitively and returns the short_key, then load_model
    # uses CHECKPOINT_DIR + info.display_name to find the local checkpoint.
    model = kimodo_load(variant, device="cuda")
    # NVIDIA Kimodo checkpoints store denoiser weights in bfloat16; inference inputs
    # are float32 — convert to float32 to avoid matmul dtype errors.
    model.float()
    model.eval()

    t1 = time.perf_counter()
    print(f"Kimodo: loaded in {t1 - t0:.1f}s")
    _model = model
    _loaded_variant = variant
    return _model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the motion to generate")
    num_frames: int = Field(default=150, description="Number of motion frames (30 fps)")
    num_denoising_steps: int = Field(default=100, description="Diffusion denoising steps")
    seed: int = Field(default=-1, description="Random seed (-1 = random)")
    cfg_weight: float | None = None
    post_processing: bool = False
    variant: str | None = Field(default=None, description="Override Kimodo variant")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": _model is not None,
        "variant": _loaded_variant,
        "default_variant": VARIANT,
        "model_root": MODEL_ROOT,
    }


@app.post("/load")
def load(variant: str | None = None):
    try:
        load_model(variant or VARIANT)
        return {"status": "ok", "variant": _loaded_variant}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {e}")


@app.post("/generate")
def generate(req: GenerateRequest):
    if _model is None:
        try:
            load_model(req.variant or VARIANT)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"model load failed: {e}")

    if req.seed >= 0:
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)

    gen_kwargs = {
        "prompts": req.prompt,
        "num_frames": req.num_frames,
        "num_denoising_steps": req.num_denoising_steps,
        "post_processing": req.post_processing,
        "progress_bar": lambda x: x,
    }
    if req.cfg_weight is not None:
        gen_kwargs["cfg_weight"] = req.cfg_weight

    t0 = time.perf_counter()
    try:
        # Float32 autocast: bfloat16 text encoder outputs → float32 denoiser
        with torch.autocast("cuda", dtype=torch.float32, enabled=True):
            output = _model(**gen_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 2)

    np_tensors = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            np_tensors[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_tensors[key] = value

    buf = io.BytesIO()
    np.savez(buf, **np_tensors)
    npz_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    gc.collect()

    return Response(
        content=npz_bytes,
        media_type="application/x-npz",
        headers={
            "X-Inference-Time-S": str(elapsed),
            "X-Num-Frames": str(req.num_frames),
            "X-Num-Steps": str(req.num_denoising_steps),
            "X-Tensor-Keys": ",".join(np_tensors.keys()),
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("KIMODO_PORT", "8098"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
